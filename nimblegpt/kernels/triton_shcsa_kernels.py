"""Single-headed causal self attention Triton kernels. Triton versions of the kernels in
nimblegpt/shcsa_kernels.py."""

import math
import jax
import triton
import triton.language as tl
import jax_triton as jt
import jax.numpy as jnp
import flax.linen as nn

next_pow2 = lambda x: int(math.pow(2, math.ceil(math.log(x, 2))))


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    # Substract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentials in Triton are fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def jt_softmax(x: jnp.ndarray) -> jnp.ndarray:
    out_shape = jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype)
    block_size = next_pow2(x.shape[1])
    strides = jt.strides_from_shape(x.shape)
    return jt.triton_call(
        x,
        kernel=softmax_kernel,
        out_shape=out_shape,
        input_row_stride=strides[0],
        output_row_stride=strides[0],
        n_cols=x.shape[1],
        grid=x.shape[0],
        BLOCK_SIZE=block_size,
    )


class SHCSATritonSoftmax(nn.Module):

    n_feat: int

    @nn.compact
    def __call__(self, x):
        T, C = x.shape  # sequence length, embedding dimensionality (n_embd)

        # [T, C] @ [C, 3 * n_feat] -> [T, 3 * n_feat] -> 3 * [T, n_feat]
        q, k, v = jnp.split(nn.Dense(features=3 * self.n_feat)(x), 3, axis=1)

        # [T, n_feat] @ [n_feat, T] -> [T, T].
        att = (q @ k.T) * (1.0 / jnp.sqrt(self.n_feat))
        causal_mask = jnp.tril(jnp.ones((T, T))).astype(bool)
        att = jnp.where(~causal_mask, -jnp.inf, att)
        att = jt_softmax(att)

        y = att @ v  # [T, T] @ [T, n_feat] -> [T, n_feat]

        return y


@triton.jit
def padded_softmax_kernel(
    att_ptr,
    p_ptr,
    output_ptr,
    att_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    n_padd = tl.load(p_ptr)

    att_row_num = tl.program_id(0)
    att_row_start_ptr = att_ptr + att_row_num * att_row_stride

    att_col_idxs = tl.arange(0, BLOCK_SIZE)
    att_ptrs = att_row_start_ptr + att_col_idxs

    valid_mask = att_col_idxs < n_cols
    causal_mask = att_col_idxs <= att_row_num
    padd_mask = (att_col_idxs >= n_padd) & (att_row_num >= n_padd)
    read_mask = valid_mask & causal_mask & padd_mask

    att_row = tl.load(att_ptrs, mask=read_mask, other=-float("inf"))

    numerator = tl.exp(att_row - tl.max(att_row, axis=0))
    sma_row = numerator / tl.sum(numerator, axis=0)

    output_row_start_ptr = output_ptr + att_row_num * output_row_stride
    output_ptrs = output_row_start_ptr + att_col_idxs

    tl.store(output_ptrs, sma_row, mask=att_col_idxs < n_cols)


def padded_softmax(att, n_padd):

    out_shape = jax.ShapeDtypeStruct(shape=att.shape, dtype=att.dtype)
    block_size = next_pow2(att.shape[1])
    strides = jt.strides_from_shape(att.shape)

    return jt.triton_call(
        att,
        jnp.array(n_padd),
        kernel=padded_softmax_kernel,
        out_shape=out_shape,
        att_row_stride=strides[0],
        output_row_stride=strides[0],
        n_cols=att.shape[1],
        grid=att.shape[0],
        BLOCK_SIZE=block_size,
    )


class SHCSATritonPaddedSoftmax(nn.Module):

    n_feat: int

    @nn.compact
    def __call__(self, x, n_padd: int = 0):
        T, C = x.shape  # sequence length, embedding dimensionality (n_embd)

        # [T, C] @ [C, 3 * n_feat] -> [T, 3 * n_feat] -> 3 * [T, n_feat]
        q, k, v = jnp.split(nn.Dense(features=3 * self.n_feat)(x), 3, axis=1)

        # [T, n_feat] @ [n_feat, T] -> [T, T].
        att = (q @ k.T) * (1.0 / jnp.sqrt(self.n_feat))
        att = padded_softmax(att, n_padd)

        y = att @ v  # [T, T] @ [T, n_feat] -> [T, n_feat]

        return y


@triton.jit
def padded_softmax_v_kernel(
    att_ptr,
    v_ptr,
    p_ptr,
    output_ptr,
    SEQ_LEN: tl.constexpr,
    N_FEAT: tl.constexpr,
    N_OCOLS: tl.constexpr,
):
    """
    Triton kernel for computing the softmax of an attention matrix which may have padding
    tokens, and then multiplying the result by a value matrix `v`.

    Kernel cell with coordinates (i, j) computes out[i, j*N_OCOLS: (j+1)*N_OCOLS]. To
    do this, it loads att[i, :] and v[:, j*N_OCOLS: (j+1)*N_OCOLS].

    Inputs
    ------
    att_ptr: [SEQ_LEN, SEQ_LEN]
    v_ptr: [SEQ_LEN, N_FEAT]

    Output
    ------
    out: [SEQ_LEN, N_FEAT]
        The output of self attention (`att @ v`).

    Constants
    ---------
    SEQ_LEN: (1024 for GPT-2)
    N_FEAT: (64 for GPT-2)
    N_OCOLS: Number of elements of output matrix computed per kernel instance.

    NOTE: Assumes all tensor sizes are powers of 2.
    """
    n_padd = tl.load(p_ptr)

    ## Load att[i, :] - with masking to avoid reading non-causal or padding tokens. ###
    seq_idxs = tl.arange(0, SEQ_LEN)
    att_row_num = tl.program_id(0)
    att_row_start_ptr = att_ptr + att_row_num * SEQ_LEN
    att_ptrs = att_row_start_ptr + seq_idxs

    att_causal_mask = seq_idxs <= att_row_num  # 0 for non-causal tokens.
    att_data_mask = (seq_idxs >= n_padd) & (
        att_row_num >= n_padd
    )  # 0 for padding tokens.
    att_read_mask = att_causal_mask & att_data_mask

    att_row = tl.load(att_ptrs, mask=att_read_mask, other=-float("inf"))

    ### Compute attention row softmax. ###

    numerator = tl.exp(att_row - tl.max(att_row, axis=0))
    sma_row = numerator / tl.sum(numerator, axis=0)  # [SEQ_LEN,]

    ### Load v[:, j*N_OCOLS: (j+1)*N_OCOLS] - with masking to avoid reading padding tokens. ###

    v_col_start = tl.program_id(1) * N_OCOLS
    v_block_start_ptr = v_ptr + v_col_start
    v_ptrs = v_block_start_ptr + (
        seq_idxs[:, None] * N_FEAT + tl.arange(0, N_OCOLS)[None, :]
    )

    # v[:n_padd, :] are values of padding tokens. The attention matrix already has zeros
    # for elements corresponding to these. We use a mask to avoid unnecessary reads.
    v_data_mask = (seq_idxs >= n_padd)[:, None] + tl.zeros((4,), dtype=tl.int1)[None, :]

    v_block = tl.load(v_ptrs, mask=v_data_mask, other=0.0)

    ### Compute output row-block. ###

    # We want to compute sma_row @ v_block, but Trition doesn't support doing this
    # directly, so we roll our own matrix-vector multiplication.
    # [SEQ_LEN,] -> [SEQ_LEN, N_OCOLS] (the same column copied N_OCOLS times).
    out = tl.sum(sma_row[:, None] * v_block, axis=0)  # [N_OCOLS,]

    ### Write output row-block. ###

    output_start_ptr = output_ptr + att_row_num * N_FEAT + v_col_start
    output_ptrs = output_start_ptr + tl.arange(0, N_OCOLS)

    out_mask = tl.zeros((N_OCOLS,), dtype=tl.uint8) + att_row_num >= n_padd

    # Don't bother writing outputs for padding tokens - just leave uninitialized.
    tl.store(output_ptrs, out, mask=out_mask)


def padded_softmax_v(att, v, n_padd, n_ocols: int = 4):

    out_shape = jax.ShapeDtypeStruct(shape=v.shape, dtype=v.dtype)
    grid = (att.shape[0], v.shape[1] // n_ocols)
    assert grid[1] * n_ocols == v.shape[1]

    return jt.triton_call(
        att,
        v,
        jnp.array(n_padd),
        kernel=padded_softmax_v_kernel,
        out_shape=out_shape,
        grid=grid,
        SEQ_LEN=att.shape[0],
        N_FEAT=v.shape[1],
        N_OCOLS=n_ocols,
    )


class SHCSATritonPaddedSoftmaxV(nn.Module):

    n_feat: int

    @nn.compact
    def __call__(self, x, n_padd: int = 0):
        T, C = x.shape  # sequence length, embedding dimensionality (n_embd)

        # [T, C] @ [C, 3 * n_feat] -> [T, 3 * n_feat] -> 3 * [T, n_feat]
        q, k, v = jnp.split(nn.Dense(features=3 * self.n_feat)(x), 3, axis=1)

        # [T, n_feat] @ [n_feat, T] -> [T, T].
        att = (q @ k.T) * (1.0 / jnp.sqrt(self.n_feat))

        y = padded_softmax_v(att, v, n_padd)

        return y


@triton.jit
def padded_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    p_ptr,
    out_ptr,
    SM_SCALE: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    N_FEAT: tl.constexpr,
    N_OCOLS: tl.constexpr,
):
    """
    Triton kernel implementing single-headed attention with causal masking, where some
    tokens may be padding.

    Kernel cell with coordinates i, j computes out[i, j*N_OCOLS: (j+1)*N_OCOLS] by
    multiplying att[i, :] by v[:, j*N_OCOLS: (j+1)*N_OCOLS].

    We compute att[i, :] by multiplying q[i, :] by k[:, :]^T.

    Inputs
    ------
    q_ptr: [SEQ_LEN, N_FEAT]
    k_ptr: [SEQ_LEN, N_FEAT]
    v_ptr: [SEQ_LEN, N_FEAT]

    Output
    ------
    out: [SEQ_LEN, N_FEAT]
        The output of self attention (`att @ v`).

    Constants
    ---------
    SEQ_LEN: (1024 for GPT-2)
    N_FEAT: (64 for GPT-2)
    N_OCOLS: Number of elements of output matrix computed per kernel instance.

    NOTE: Assumes all tensor sizes are powers of 2.
    """
    n_padd = tl.load(p_ptr)

    out_row_num = tl.program_id(0)
    out_col_start = tl.program_id(1) * N_OCOLS
    # This cell computes out[out_row_num, out_col_start: out_col_start + N_OCOLS].

    seq_idxs = tl.arange(0, SEQ_LEN)
    feat_idxs = tl.arange(0, N_FEAT)
    ocols_idxs = tl.arange(0, N_OCOLS)

    # Shape (1,) mask. 0 if this instance is computing only padding elements of `out`.
    data_row_num_mask = out_row_num >= n_padd
    # Shape (SEQ_LEN,) mask. 0 for tokens of the sequence which are padding.
    data_seq_mask = seq_idxs >= n_padd
    # Shape (SEQ_LEN, N_OCOLS) mask. 0 for features of v corresponding to padding tokens.
    data_v_block_mask = tl.broadcast_to(data_seq_mask[:, None], (SEQ_LEN, N_OCOLS))
    # Shape (SEQ_LEN, N_FEAT) mask into k. 0 for features corresponding to padding tokens.
    data_k_mat_mask = tl.broadcast_to(data_seq_mask[:, None], (SEQ_LEN, N_FEAT))
    # Shape (SEQ_LEN,). 0 for non-causal elements of `att`.
    causal_mask = seq_idxs <= out_row_num

    ### Compute the softmax of att[out_row_num, :]. ###
    # This requires loading one row of Q and all of K.

    q_row_start_ptr = q_ptr + out_row_num * N_FEAT
    q_ptrs = q_row_start_ptr + feat_idxs
    q_row = tl.load(q_ptrs, mask=data_row_num_mask, other=0.0)  # [N_FEAT,]

    k_ptrs = k_ptr + (seq_idxs[:, None] * N_FEAT + feat_idxs[None, :])
    k_mat = tl.load(k_ptrs, mask=data_k_mat_mask, other=0.0)  # [SEQ_LEN, N_FEAT]

    # ([N_FEAT,] -> [1, N_FEAT]) * [SEQ_LEN, N_FEAT] -> [SEQ_LEN, N_FEAT].
    att_row = tl.sum(q_row[None, :] * k_mat, axis=1) * SM_SCALE
    # padding and non-causal elements currenly have value 0. We need to set them to -inf
    # for the softmax.
    causal_att_row = tl.where(causal_mask & data_seq_mask, att_row, float("-inf"))
    sm_numerator = tl.exp(causal_att_row - tl.max(causal_att_row, axis=0))
    sm_att_row = sm_numerator / tl.sum(sm_numerator, axis=0)  # [seq_len,]

    ### Multiply att[out_row_num, :] by v[:, out_col_start: out_col_start + N_OCOLS]. ###

    v_block_start_ptr = v_ptr + out_col_start
    v_ptrs = v_block_start_ptr + (seq_idxs[:, None] * N_FEAT + ocols_idxs[None, :])
    v_block = tl.load(v_ptrs, mask=data_v_block_mask, other=0.0)  # [SEQ_LEN, N_OCOLS]

    out = tl.sum(sm_att_row[:, None] * v_block, axis=0)  # [N_OCOLS,]

    ### Write output row-block. ###

    out_row_start_ptr = out_ptr + out_row_num * N_FEAT
    out_ptrs = out_row_start_ptr + out_col_start + ocols_idxs

    tl.store(out_ptrs, out, mask=data_row_num_mask)


def padded_attention(q, k, v, n_padd, n_ocols: int = 4):

    out_shape = jax.ShapeDtypeStruct(shape=v.shape, dtype=v.dtype)
    grid = (q.shape[0], q.shape[1] // n_ocols)
    assert grid[1] * n_ocols == q.shape[1]

    return jt.triton_call(
        q,
        k,
        v,
        jnp.array(n_padd),
        kernel=padded_attention_kernel,
        out_shape=out_shape,
        grid=grid,
        SM_SCALE=1.0 / k.shape[1] ** 0.5,
        SEQ_LEN=q.shape[0],
        N_FEAT=q.shape[1],
        N_OCOLS=n_ocols,
    )


class SHCSATriton(nn.Module):

    n_feat: int

    @nn.compact
    def __call__(self, x, n_padd: int = 0):
        T, C = x.shape  # sequence length, embedding dimensionality (n_embd)

        # [T, C] @ [C, 3 * n_feat] -> [T, 3 * n_feat] -> 3 * [T, n_feat]
        q, k, v = jnp.split(nn.Dense(features=3 * self.n_feat)(x), 3, axis=1)

        y = padded_attention(q, k, v, n_padd)

        return y
