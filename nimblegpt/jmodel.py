from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax
from ml_collections import ConfigDict


def GELU(x):
    """
    minGPT docstring
    ----------------
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * x**3)))


# Copied from https://jax.readthedocs.io/en/latest/_modules/jax/_src/nn/functions.html#softmax
def softmax(
    x: jax.Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[jax.Array] = None,
) -> jax.Array:
    r"""Softmax function.

    Computes the function which rescales elements to the range :math:`[0, 1]`
    such that the elements along :code:`axis` sum to :math:`1`.

    .. math ::
      \mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    Args:
      x : input array
      axis: the axis or axes along which the softmax should be computed. The
        softmax output summed across these dimensions should sum to :math:`1`.
        Either an integer or a tuple of integers.
      where: Elements to include in the :code:`softmax`.
      initial: The minimum value used to shift the input array. Must be present
        when :code:`where` is not None.
    """
    x_max = jnp.max(x, axis, where=where, initial=-jnp.inf, keepdims=True)
    unnormalized = jnp.exp(x - lax.stop_gradient(x_max))
    return unnormalized / jnp.sum(unnormalized, axis, where=where, keepdims=True)


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


class JSingleHeadCausalSelfAttention(nn.Module):
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

        # Token i should not attend to token j for any j > i. We set att to -inf
        # for any position above the diagonal - i.e. where j > i.
        causal_mask = ~jnp.tril(jnp.ones((T, T))).astype(bool)
        att = jnp.where(causal_mask, -jnp.inf, att)

        # Data tokens should not attend to padding tokens. For any data token
        # i > n_padd, set the attention values to padding tokens to -inf.
        # Equivalent to: `att = att.at[n_padd:, :n_padd].set(-jnp.inf)`.
        padd_mask = make_2d_mask(T, T, n_padd, T, 0, n_padd)
        att = jnp.where(padd_mask, -jnp.inf, att)

        att = softmax(att, axis=-1)

        y = att @ v  # [T, T] @ [T, n_feat] -> [T, n_feat]

        return y


class JCausalSelfAttention(nn.Module):
    n_head: int

    @nn.compact
    def __call__(self, x, n_padd: int = 0):
        T, C = x.shape  # sequence length, embedding dimensionality (n_embd)
        n_feat = C // self.n_head  # Features per q/k/v per head.

        # [T, C] -> [T, n_head, n_feat]
        y = nn.vmap(
            JSingleHeadCausalSelfAttention,
            in_axes=None,  # Don't map over `x` - each `SingleHead CausalSelfAttention` gets the full `x`.
            axis_size=self.n_head,
            out_axes=1,
            variable_axes={"params": 0},  # 0th axis of params should be the vmap axis.
            split_rngs={"params": True},
        )(n_feat=n_feat)(x, n_padd)
        y = jnp.reshape(y, (T, C))  # [T, n_head, n_feat] -> [T, C]

        y = nn.Dense(features=C)(y)
        return y


class JBlock(nn.Module):
    n_head: int

    @nn.compact
    def __call__(self, x, n_padd: int = 0):
        T, C = x.shape  # Sequence length, embedding dimensionality.

        y = nn.LayerNorm()(x)
        y = JCausalSelfAttention(n_head=self.n_head)(y, n_padd=n_padd)
        x = x + y

        y = nn.LayerNorm()(x)
        y = nn.Dense(features=4 * C)(y)
        y = GELU(y)
        y = nn.Dense(features=C)(y)

        x = x + y

        return x


class JGPT(nn.Module):
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
            x = JBlock(n_head=self.C.n_head)(x, n_padd=n_padd)

        x = nn.LayerNorm()(x)
        logits = nn.Dense(features=self.C.vocab_size, use_bias=False)(x)

        return logits
