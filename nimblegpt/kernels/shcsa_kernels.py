"""Single-headed causal self attention Pallas/Triton kernels."""

import functools
import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax
from jax_triton import pallas as pl


def padded_softmax_kernel(att_ref, p_ref, o_ref, *, seq_len: int):
    """Pallas/Triton kernel which computes the row-softmax of an attention matrix which
    includes padding tokens.

    NOTE: Assumes inputs have power-of-two dimension sizes.
    """

    row_idx = pl.program_id(0)
    n_padd = p_ref[()]

    att_idx = jnp.arange(seq_len)
    row_idxs = (row_idx, att_idx)

    # Token i should only attend to tokens j <= i.
    causal_mask = att_idx <= row_idx

    # 1 in the bottom right corner of the matrix - where data tokens attend to data
    # tokens. 0 elsewhere.
    padd_mask = (att_idx >= n_padd) & (row_idx >= n_padd)

    read_mask = causal_mask & padd_mask
    row = pl.load(att_ref, row_idxs, mask=read_mask, other=-jnp.inf)

    row_minus_max = row - jnp.max(row, axis=0)
    numerator = jnp.exp(row_minus_max)
    denominator = jnp.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Only write back to rows corresponding to non-padding tokens. Padding tokens
    # may be uninitialized memory.
    pl.store(o_ref, row_idxs, softmax_output, mask=row_idx >= n_padd)


def padded_softmax(att, n_padd):

    # assert math.log2(att.shape[1]).is_integer()

    seq_len = att.shape[0]

    grid = (seq_len,)

    kernel = functools.partial(padded_softmax_kernel, seq_len=seq_len)

    out_shape = jax.ShapeDtypeStruct((seq_len, seq_len), jnp.float32)

    out = pl.pallas_call(kernel, grid=grid, out_shape=out_shape, interpret=True)(
        att, n_padd
    )

    return out


class PaddedSoftmaxSHCSA(nn.Module):

    n_feat: int

    @nn.compact
    def __call__(self, x, n_padd: int = 0):
        T, C = x.shape  # sequence length, embedding dimensionality (n_embd)

        # [T, C] @ [C, 3 * n_feat] -> [T, 3 * n_feat] -> 3 * [T, n_feat]
        q, k, v = jnp.split(nn.Dense(features=3 * self.n_feat)(x), 3, axis=1)

        # [T, n_feat] @ [n_feat, T] -> [T, T].
        # Row i of att tells us which tokens x[i] should attend to. att[i][j]
        # is high when token i should attend heavily to token j.
        att = (q @ k.T) * (1.0 / jnp.sqrt(self.n_feat))

        att = padded_softmax(att, n_padd)

        y = att @ v  # [T, T] @ [T, n_feat] -> [T, n_feat]

        return y


def padded_softmax_v_kernel(
    att_ref, v_ref, p_ref, o_ref, *, seq_len: int, n_ocols: int
):
    """Pallas/Triton kernel which computes the row-softmax of an attention matrix which
    include padding tokens, and multiplies the result by the value matrix `v`.

    NOTE: Assumes inputs have power-of-two dimension sizes.
    """
    # Row of attention matrix that this kernel instance will process.
    att_row_num = pl.program_id(0)
    # Start of the block of columns of the `v` matrix that this kernel instance will process.
    v_col_start = pl.program_id(1) * n_ocols
    n_padd = p_ref[()]

    ### Create indicies for reading memory. ###
    seq_idxs = jnp.arange(seq_len)

    att_idxs = (att_row_num, pl.dslice(None))

    ## [seq_len,] mask.
    # Token i should only attend to tokens j <= i.
    causal_mask = seq_idxs <= att_row_num
    padd_from_mask = (
        seq_idxs >= n_padd
    )  # 0 when padding tokens are attending to anything.
    padd_to_mask = (
        att_row_num >= n_padd
    )  # 0 when anything is attending to padding tokens.
    padd_mask = padd_from_mask & padd_to_mask
    seq_mask = causal_mask & padd_mask

    ## Index for v[:, (j:j+1)*n_ocols].
    v_col_idxs = pl.dslice(v_col_start, n_ocols)
    v_row_idxs = pl.dslice(0, seq_len)
    v_idxs = (v_row_idxs, v_col_idxs)

    ## Only read elements of `v` which will be multipled by non-padding tokens.
    v_row_mask = padd_from_mask
    v_mask = lax.broadcast_in_dim(
        jnp.expand_dims(v_row_mask, 1), (seq_len, n_ocols), (0, 1)
    )

    out_idxs = (att_row_num, pl.dslice(v_col_start, n_ocols))

    ### Compute attn row softmax. ###
    att_row = pl.load(att_ref, att_idxs, mask=seq_mask, other=-float("inf"))

    numerator = jnp.exp(att_row - jnp.max(att_row, axis=0))
    sma_row = numerator / jnp.sum(numerator, axis=0)

    ### Multiply attention by `v`. ###
    v_block = pl.load(v_ref, v_idxs, mask=v_mask, other=0)

    # We want to do `out = sma_row @ v_block` ([seq_len,] @ [seq_len, n_ocols] => [n_ocols,])
    # But Triton doesn't support matrix multiplication for small matrices.

    # Poor man's matrix multiplication (may be slowing us down since it doesn't use tensor cores).
    sma_mat = jnp.expand_dims(sma_row, 1)  # [seq_len, 1]
    # [seq_len, 1] * [seq_len, n_ocols] -> [seq_len, n_ocols] -[sum]-> [n_ocols,]
    out = jnp.sum(sma_mat * v_block, axis=0)

    ### Write output. ###
    pl.store(o_ref, out_idxs, out)


def padded_softmax_v(att, v, n_padd, n_ocols=4):

    # assert math.log2(att.shape[0]).is_integer()
    # assert math.log2(v.shape[1]).is_integer()

    seq_len = att.shape[0]
    n_feat = v.shape[1]

    assert n_feat % n_ocols == 0

    grid = (seq_len, n_feat // n_ocols)

    kernel = functools.partial(
        padded_softmax_v_kernel, seq_len=seq_len, n_ocols=n_ocols
    )

    out_shape = jax.ShapeDtypeStruct((seq_len, n_feat), jnp.float32)

    out = pl.pallas_call(kernel, grid=grid, out_shape=out_shape, interpret=True)(
        att, v, n_padd
    )

    return out


class PaddedSoftmaxVSHCSA(nn.Module):

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


def padded_attention_kernel(
    q_ref, k_ref, v_ref, p_ref, o_ref, *, seq_len: int, n_feat: int, n_ocols: int
):
    """
    Pallas/Triton kernel which computes attention given Q, K, and V matrices, which
    may include padding tokens.

    Inputs
    ------
    q_ref, k_rev, v_ref: references to the Q, K, and V matrices.
        All have shape shape [seq_len, n_feat].

    Each kernel instances computes out[out_row_num, out_col_start: out_col_start + n_ocols].
    This requires multiplying att[out_row_num, :] by v[:, out_col_start: out_col_start + n_ocols].

    To compute att[out_row_num, :], we multiply q[out_row_num, :] by
    k^T.
    """

    # Each instance computes out[out_row_num, out_col_start: out_col_start + n_ocols]
    out_row_num = pl.program_id(0)
    out_col_start = pl.program_id(1) * n_ocols
    n_padd = p_ref[()]

    seq_idxs = jnp.arange(seq_len)

    # Shape (1,) mask. 0 if this instance is computing a padding element of `out`.
    padd_row_mask = out_row_num >= n_padd
    # Shape (seq_len,) mask. 0 for tokens of the sequence that are padding.
    seq_mask = jnp.arange(seq_len) >= n_padd
    # Shape (seq_len, n_ocols) mask. 0 for elements corresponding to padding tokens.
    block_mask = lax.broadcast_in_dim(
        jnp.expand_dims(seq_mask, 1), (seq_len, n_ocols), (0, 1)
    )
    # Shape (seq_len, n_feat) mask. 0 for elements corresponding to padding tokens.
    mat_mask = lax.broadcast_in_dim(
        jnp.expand_dims(seq_mask, 1), (seq_len, n_feat), (0, 1)
    )
    # Token i should only atten to tokens j <= i. 0 for tokens j > i.
    causal_mask = seq_idxs <= out_row_num

    ### First we compute the softmax of row `out_row_num` of the attention matrix. ###
    # This requires loading one row of Q and all of K.

    q_idx = (out_row_num, pl.dslice(None))
    q_row = pl.load(q_ref, q_idx, mask=padd_row_mask, other=0)  # [n_feat]
    q_row = jnp.expand_dims(q_row, 0)  # [1, n_feat]

    k_idx = (pl.dslice(None), pl.dslice(None))
    k_mat = pl.load(k_ref, k_idx, mask=mat_mask, other=0)  # [seq_len, n_feat]

    # Compute att[out_row_num, :] - a single row of the full attention matrix.
    # [1, n_feat] . ([seq_len, n_feat] -[T]-> [n_feat, seq_len]) = [1, seq_len]
    att_row = pl.dot(q_row, k_mat, trans_b=True)
    att_row /= jnp.sqrt(n_feat)
    att_row = jnp.where(causal_mask & seq_mask, att_row, -jnp.inf)
    sm_numerator = jnp.exp(att_row - jnp.max(att_row))
    sm_att = sm_numerator / jnp.sum(sm_numerator, keepdims=True)  # [1, seq_len]

    v_idxs = (pl.dslice(None), pl.dslice(out_col_start, n_ocols))
    v_block = pl.load(v_ref, v_idxs, mask=block_mask, other=0)  # [seq_len, n_ocols]

    # [1, seq_len] . [seq_len, n_ocols] = [1, n_ocols]
    # out = pl.dot(sm_att, v_block) # [1, n_ocols]
    out = sm_att @ v_block

    # Store the result.
    out_idxs = (out_row_num, pl.dslice(out_col_start, n_ocols))
    pl.store(o_ref, out_idxs, out[0], mask=padd_row_mask)


def padded_attention(q, k, v, n_padd, n_ocols=4):

    seq_len, n_feat = q.shape

    assert n_feat % n_ocols == 0

    grid = (seq_len, n_feat // n_ocols)

    kernel = functools.partial(
        padded_attention_kernel, seq_len=seq_len, n_feat=n_feat, n_ocols=n_ocols
    )

    out_shape = jax.ShapeDtypeStruct((seq_len, n_feat), jnp.float32)

    out = pl.pallas_call(kernel, grid=grid, out_shape=out_shape, interpret=True)(
        q, k, v, n_padd
    )

    return out


class PaddedSHCSA(nn.Module):

    n_feat: int

    @nn.compact
    def __call__(self, x, n_padd: int = 0):
        # [T, n_feat] -> [T, n_feat].
        q, k, v = jnp.split(nn.Dense(features=3 * self.n_feat)(x), 3, axis=1)

        y = padded_attention(q, k, v, n_padd)

        return y
