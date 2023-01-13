import functools
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax
from ml_collections import ConfigDict
from jax_triton import pallas as pl


def GELU(x):
    """
    minGPT docstring
    ----------------
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))


def make_padded_softmax(block_size: int, num_warps: int = 1):
    """
    Returned kernel has signature:
    def padded_softmax_kernel(att: Array[block_size, block_size], n_padd: int):

    Such that `padded_softmax_kernel(att, n_padd)` only returns a correct result for
    rows which don't correspond to padding tokens.
    """
    # grid = block_size => one kernel instance per row of the input matrix.
    @functools.partial(
        pl.pallas_call,
        out_shape=jax.ShapeDtypeStruct((block_size, block_size), jnp.float32),
        grid=block_size,
        num_warps=num_warps,
    )
    def padded_softmax_kernel(x_ref, p_ref, o_ref):
        row_idx = pl.program_id(0)
        n_padd = p_ref[()]

        x_idx = jnp.arange(block_size)
        row_idxs = (row_idx, x_idx)

        # 1 for valid elements of `x_ref`, 0 elsewhere (i.e. out of bounds).
        valid_mask = x_idx < x_ref.shape[1]

        # Token i should only attend to tokens j <= i.
        causal_mask = x_idx <= row_idx

        # 1 in the bottom right corner of the matrix - where data tokens attend to data
        # tokens. 0 elsewhere.
        padd_mask = (x_idx >= n_padd) & (row_idx >= n_padd)

        read_mask = valid_mask & causal_mask & padd_mask
        row = pl.load(x_ref, row_idxs, mask=read_mask, other=-float("inf"))

        row_minus_max = row - jnp.max(row, axis=0)
        numerator = jnp.exp(row_minus_max)
        denominator = jnp.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # TODO: For some reason this crashes triton.
        # Zero out attention matrix for masked tokens.
        # full_output = jnp.where(
        #     causal_mask & padd_mask,
        #     softmax_output,
        #     jnp.zeros(
        #         block_size,
        #     ),
        # )

        pl.store(o_ref, row_idxs, softmax_output, mask=valid_mask)
        # pl.store(
        #     o_ref,
        #     row_idxs,
        #     jnp.zeros(block_size),
        #     mask=valid_mask & ~causal_mask & ~padd_mask,
        # )

    return padded_softmax_kernel


def make_2d_mask(
    n_row: int, n_col: int, row_start: int, row_stop: int, col_start: int, col_stop: int
):
    """
    Equivalent to
    ```
    m = jnp.zeros((n_row, n_col))
    m.at[row_start: row_stop, col_start: col_stop].set(1)
    ```
    Except jittable! (as long as `n_row` and `n_col` are static).
    """
    mat = jnp.arange(n_row * n_col).reshape((n_row, n_col))
    row_mask = (mat >= n_col * row_start) & (mat <= n_col * row_stop - 1)
    col_mask = (mat % n_col >= col_start) & (mat % n_col < col_stop)
    return row_mask & col_mask


class TSingleHeadCausalSelfAttention(nn.Module):
    """
    Inference only (no dropout) single headed attention.

    minGPT docstring
    ----------------
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

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

        att = make_padded_softmax(T)(att, n_padd)

        y = att @ v  # [T, T] @ [T, n_feat] -> [T, n_feat]

        return y


class TCausalSelfAttention(nn.Module):
    n_head: int

    @nn.compact
    def __call__(self, x, n_padd: int = 0):
        T, C = x.shape  # sequence length, embedding dimensionality (n_embd)
        n_feat = C // self.n_head  # Features per q/k/v per head.

        # [T, C] -> [T, n_head, n_feat]
        # TODO: mapping over [n_padd] * n_head is to avoid an issue in pallas.
        y = nn.vmap(
            TSingleHeadCausalSelfAttention,
            in_axes=(
                None,
                0,
            ),  # Don't map over `x` - each `SingleHead CausalSelfAttention` gets the full `x`.
            axis_size=self.n_head,
            out_axes=1,
            variable_axes={"params": 0},  # 0th axis of params should be the vmap axis.
            split_rngs={"params": True},
        )(n_feat=n_feat)(x, jnp.ones((self.n_head)) * n_padd)
        y = jnp.reshape(y, (T, C))  # [T, n_head, n_feat] -> [T, C]

        y = nn.Dense(features=C)(y)
        return y


class TBlock(nn.Module):
    n_head: int

    @nn.compact
    def __call__(self, x, n_padd: int = 0):
        T, C = x.shape  # Sequence length, embedding dimensionality.

        y = nn.LayerNorm()(x)
        y = TCausalSelfAttention(n_head=self.n_head)(y, n_padd=n_padd)
        x = x + y

        y = nn.LayerNorm()(x)
        y = nn.Dense(features=4 * C)(y)
        y = GELU(y)
        y = nn.Dense(features=C)(y)

        x = x + y

        return x


class TGPT(nn.Module):
    C: ConfigDict

    @nn.compact
    def __call__(self, indices, n_padd: int = 0):
        """
        Parameters
        ----------
        indicies : jnp.ndarray
            Array of token indices of shape (T,). See 'bpe.py' for how text is converted
            into indices.
        n_padd : int
            Number of padding tokens before the data tokens in `indices`.
        """
        (T,) = indices.shape  # One index per token in the sequence.

        # Rotate positions so that first non-padding token has position 0.
        pos = (jnp.arange(0, T) - n_padd) % self.C.block_size

        # Token embeddings of shape [T, n_embd].
        tok_emb = nn.Embed(num_embeddings=self.C.vocab_size, features=self.C.n_embd)(
            indices
        )
        # Position embeddings of shape [T, n_embd].
        pos_emb = nn.Embed(num_embeddings=self.C.block_size, features=self.C.n_embd)(
            pos
        )

        x = tok_emb + pos_emb

        for _ in range(self.C.n_layer):
            x = TBlock(n_head=self.C.n_head)(x, n_padd=n_padd)

        x = nn.LayerNorm()(x)
        logits = nn.Dense(features=self.C.vocab_size, use_bias=False)(x)

        return logits
