import jax
import triton
import triton.language as tl
import jax_triton as jt
from nimblegpt.fast_model import FSingleHeadCausalSelfAttention


@triton.jit
def resoftmax(y1, y2, m1, m2, l1, l2):
    """
    Combine two partial results which have been independently softmax-ed.

    Given:
        - y1 = softmax(x1) \cdot v1 and y2 = softmax(x2) \cdot v2
        - m1 = max(x1) and m2 = max(x2)
        - l1 = sum(exp(x1 - m1)) and l2 = sum(exp(x2 - m2)) (the softmax denominators).

    Define:
        - x = [x1, x2]
        - v = [v1, v2]
        - y = y1 + y2

    This function computes y = softmax(x) \cdot v` and returns y, m, l.
    """

    # New max.
    m = tl.where(m1 > m2, m1, m2)
    # New denominator.
    l = tl.exp(m1 - m) * l1 + tl.exp(m2 - m) * l2

    return (l1 / tl.exp(m - m1) * y1 + l2 / tl.exp(m - m2) * y2) / l, m, l


@triton.jit
def shcsa_block_kernel(
    q_ptr,
    K_ptr,
    V_ptr,
    seq_idx_ptr,
    out_ptr,
    SM_SCALE: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    N_FEAT: tl.constexpr,
    SUBSEQ_SIZE: tl.constexpr,
    SUBFEAT_SIZE: tl.constexpr,
):
    """
    Triton kernel implementing single-headed attention with causal masking for a single
    token embedding. We iterate over `SEBSEQ_SIZE` chunks of the attention vector, 
    flash-attention style. Each kernel cell computes `SUBFEAT_SIZE` elements of the output 
    vector.

    For clarity, we call subsets of the sequence axis 'subseqs' and subsets of the feature
    axis 'subfeats'. We call a tensor a 'block' when it has shape [SUBSEQ_SIZE,] or
    [SUBSEQ_SIZE, N_FEAT], and a 'chunk' when it has size [SUBFEAT_SIZE,] or 
    [SUBSEQ_SIZE, SUBFEAT_SIZE].

    Kernel cell i comuputes out[i * SUBFEAT_SIZE, (i+1)* SUBFEAT_SIZE], by multiplying `att` 
    with v[:, i * SUBFEAT_SIZE, (i+1)* SUBFEAT_SIZE] and summing over the sequence dimension.

    As with flash-attention, the output is computed interatively in blocks. The sequence
    is split into `SEQ_LEN // SUBSEQ_SIZE` blocks of size `SUBSEQ_SIZE`.

    Inputs
    ------
    q_ptr: [N_FEAT] - query vector for the current token.
    K_ptr: [SEQ_LEN, N_FEAT] - key matrix for the entire sequence.
    V_ptr: [SEQ_LEN, N_FEAT] - value matrix for the entire sequence.
    seq_idx_ptr: [] - index of the current token in the sequence.

    Paramters
    ---------
    SUBSEQ_SIZE: int
        Size of the subsequence blocks. Should be small enough that blocks fit in SRAM.
        Smaller values may increase warp occupancy, since more warps can be active at once.
    SUBFEAT_SIZE: int
        Number of output elements computed per kernel cell. Note that each kernel cell
        will independently compute the attention row. Small values increase parellelism
        (and therefore speed) but increase GPU utilisation super-linearly.

    Output
    ------
    out_ptr: [N_FEAT] - self attention output (`att @ v`)
    """
    seq_idx = tl.load(seq_idx_ptr)

    # This cell computes out[subfeat_start, subfeat_start: subfeat_start + SUBFEAT_SIZE].
    subfeat_start = tl.program_id(0) * SUBFEAT_SIZE

    seq_idxs = tl.arange(0, SEQ_LEN)
    feat_idxs = tl.arange(0, N_FEAT)

    subfeat_idxs = tl.arange(0, SUBFEAT_SIZE) + subfeat_start

    q = tl.load(q_ptr + feat_idxs)  # [N_FEAT,]

    # Tensor for accumulating the output chunk.
    out_chunk_acc = tl.zeros((SUBFEAT_SIZE, ), dtype=tl.float32)
    ## Running softmax - flash-attention style.
    # Running softmax maximum.
    m_acc = float("-inf")
    # Running softmax denoniator.
    l_acc = 0.0

    # Don't bother calculating attention and outputs for tokens which are causally masked.
    n_subseq = tl.cdiv(seq_idx + 1, SUBSEQ_SIZE)

    for subseq_i in range(0, n_subseq):
        # Each iteration, we load a [SEBSEQ_SIZE, N_FEAT] block of `K`` and compute a
        # [SUBSEQ_SIZE,] block of `att`. We mulitply this by a [SEBSEQ_SIZE, SUBFEAT_SIZE]
        # block of `V`` to compute a [SUBFEAT_SIZE,] partial-result block of `out.`

        # Index io tokens of the sequence which are processed in this block.
        # TODO: this should probably be subseq_i * SUBSEQ_SIZE.
        subseq_idxs = tl.arange(0, SUBSEQ_SIZE) + subseq_i
        # Causal mask for sequence tokens in this block.
        subseq_mask = subseq_idxs <= seq_idx

        # Index and mask into K. Sizes [SUBSEQ_SIZE, N_FEAT].
        block_idxs = subseq_idxs[:, None] * N_FEAT + feat_idxs[None, :]
        block_mask = tl.broadcast_to(subseq_mask[:, None],
                                     (SUBSEQ_SIZE, N_FEAT))

        K_block = tl.load(K_ptr + block_idxs, mask=block_mask,
                          other=0.0)  # [SUBSEQ_SIZE, N_FEAT]

        att_block = tl.sum(q[None, :] * K_block,
                           axis=1) * SM_SCALE  # [BLOCK_SIZE,]
        catt_block = tl.where(subseq_mask, att_block,
                              float("-inf"))  # [BLOCK_SIZE,]

        max_block = tl.max(catt_block, axis=0)
        # Softmax numerator of this block.
        sm_num_block = tl.exp(catt_block - max_block)
        # Softmax denominator of this block.
        sm_den_block = tl.sum(sm_num_block, axis=0)
        sm_att_block = sm_num_block / sm_den_block

        ## Load V[(subseq_i: subseq_i+1) * SUBSEQ_SIZE,
        #         subfeat_start: subfeat_start + SUBFEAT_SIZE]

        # Index and mask into V. Sizes [SUBSEQ_SIZE, SUBFEAT_SIZE].
        chunk_idxs = subseq_idxs[:, None] * N_FEAT + subfeat_idxs[None, :]
        chunk_mask = tl.broadcast_to(subseq_mask[:, None],
                                     (SUBSEQ_SIZE, SUBFEAT_SIZE))

        V_chunk = tl.load(V_ptr + chunk_idxs, mask=chunk_mask,
                          other=0.0)  # [SUBSEQ_SIZE, SUBFEAT_SIZE]

        # Partial result of a chunk of the output.
        # ([SUBSEQ_SIZE,] -> [SUBSEQ_SIZE, 1]) * [SUBSEQ_SIZE, SUBFEAT_SIZE]
        # -> [SUBSEQ_SIZE, SUBFEAT_SIZE] -{sum}-> [SUBFEAT_SIZE,]
        out_chunk_pr = tl.sum(sm_att_block[:, None] * V_chunk, axis=0)

        out_chunk_acc, m_acc, l_acc = resoftmax(out_chunk_acc, out_chunk_pr,
                                                m_acc, max_block, l_acc,
                                                sm_den_block)

    tl.store(out_ptr + subfeat_idxs, out_chunk_acc)


def shcsa_block(q,
                K,
                V,
                seq_idx,
                SUBSEQ_SIZE: int = 128,
                SUBFEAT_SIZE: int = 32):
    N_FEAT = q.shape[0]

    out_shape = jax.ShapeDtypeStruct((N_FEAT, ), q.dtype)
    grid = (N_FEAT // SUBFEAT_SIZE, )

    return jt.triton_call(q,
                          K,
                          V,
                          seq_idx,
                          kernel=shcsa_block_kernel,
                          out_shape=out_shape,
                          grid=grid,
                          SM_SCALE=1.0 / N_FEAT**0.5,
                          SEQ_LEN=K.shape[0],
                          N_FEAT=N_FEAT,
                          SUBSEQ_SIZE=SUBSEQ_SIZE,
                          SUBFEAT_SIZE=SUBFEAT_SIZE)


class SHCSABlock(FSingleHeadCausalSelfAttention):
    subseq_size: int = 128
    subfeat_size: int = 32

    def __call__(self, x: jax.Array, seq_idx: jax.Array):

        q, K, V = self.get_qKV(x, seq_idx)
        return shcsa_block(q,
                           K,
                           V,
                           seq_idx,
                           SUBSEQ_SIZE=self.subseq_size,
                           SUBFEAT_SIZE=self.subfeat_size)
