"""Implementation of GPT optimized for speed over readability.

To help with experimentation, each module is 'pluggable' in the sense that the caller
chooses the implementation of each submodule.
"""

from functools import partial
from typing import Callable, Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax
from ml_collections import ConfigDict

from nimblegpt.base_model import (BaseBlock, BaseCausalSelfAttention, BaseGPT,
                                  BaseSingleHeadCausalSelfAttention)
from nimblegpt.model import GELU, softmax


class FSingleHeadCausalSelfAttention(BaseSingleHeadCausalSelfAttention):
    n_cntx: int
    n_feat: int

    def __call__(self, x: jax.Array, seq_idx: jax.Array):

        # q : [n_feat], K, V : [n_cntx, n_feat]
        q, K, V = self.get_qKV(x, seq_idx)

        # [n_feat] @ [n_feat, n_cntx] -> [n_cntx].
        # Attention for token `idx`. att[i] is high when token `idx` should attend
        # heavily to token i.
        att = (K @ q) * (1.0 / jnp.sqrt(self.n_feat))

        # Causal masking. Token `seq_idx` should not attend to token i for any i > idx.
        att = jnp.where(jnp.arange(self.n_cntx) > seq_idx, -jnp.inf, att)

        att = softmax(att)

        # [n_cntx] @ [n_cntx, n_feat] -> [n_feat]
        y = att @ V

        return y

    # Note: We implement this as a method instead of a separate module so that the
    # variables are scoped under 'SingleHeadCausalSelfAttention'.
    @nn.compact
    def get_qKV(self, x: jax.Array, seq_idx: jax.Array):
        """
        Compute Q, K, V matrices for a single head.

        This module processes a single token embedding at a time, and builds up a cache
        of K and V matrices for the entire sequence. The caching implementation is based on:
        https://flax.readthedocs.io/en/latest/_modules/flax/linen/attention.html#MultiHeadDotProductAttention

        Parameters
        ----------
        x : jax.Array
            Shape [n_embd]. The token embedding for the next token in the sequence.
        seq_idx : jax.Array
            Shape []. The index of `x` in the sequence. Due to caching, this function 
            behaves statefully. When called with `seq_idx=i`, the function will only
            return correct results if it has previously been called with `seq_idx=j` 
            for all j < i.

        Returns
        -------
        Q, K, V, idx : Tuple
            Q : [n_feat] - The query vector for `x`.
            K, V : [n_cntx, n_feat] - The key and value matrices for the entire context.
            idx : int - The index of `x` in the context.
        """
        # Attention q, k, v vectors for token embedding `x`. Shape [n_feat].
        q, k, v = jnp.split(nn.Dense(features=3 * self.n_feat)(x), 3, axis=0)

        is_initialized = self.has_variable("cache", "cached_keys")

        # Cached K and V matrices. Shape [n_cntx, n_feat].
        cached_keys = self.variable("cache", "cached_keys", jnp.zeros,
                                    (self.n_cntx, self.n_feat), k.dtype)
        cached_values = self.variable("cache", "cached_values", jnp.zeros,
                                      (self.n_cntx, self.n_feat), v.dtype)

        if is_initialized:

            K = lax.dynamic_update_slice(cached_keys.value,
                                         jnp.expand_dims(k, axis=0),
                                         (seq_idx, 0))
            V = lax.dynamic_update_slice(cached_values.value,
                                         jnp.expand_dims(v, axis=0),
                                         (seq_idx, 0))

            cached_keys.value = K
            cached_values.value = V

        return q, cached_keys.value, cached_values.value


class FCausalSelfAttention(BaseCausalSelfAttention):
    SingleHeadCausalSelfAttention: BaseSingleHeadCausalSelfAttention
    n_cntx: int
    n_head: int

    @nn.compact
    def __call__(self, x, seq_idx):
        C, = x.shape  # Embedding dimensionality (n_embd)
        n_feat = C // self.n_head  # Features per q/k/v per head.

        # [C,] -> [n_head, n_feat]
        y = nn.vmap(
            self.SingleHeadCausalSelfAttention,
            in_axes=
            None,  # Don't map over `x` - each `SingleHead CausalSelfAttention` gets the full `x`.
            axis_size=self.n_head,
            out_axes=0,
            variable_axes={
                "params": 0,
                "cache": 0
            },  # 0th axis of params should be the vmap axis.
            split_rngs={"params": True},
        )(n_feat=n_feat, n_cntx=self.n_cntx)(x, seq_idx)
        y = jnp.reshape(y, (C, ))  # [n_head, n_feat] -> [C,]

        y = nn.Dense(features=C)(y)
        return y


class FBlock(BaseBlock):
    CausalSelfAttention: BaseCausalSelfAttention
    n_cntx: int
    n_head: int

    @nn.compact
    def __call__(self, x, seq_idx):
        C, = x.shape  # Embedding dimensionality.

        y = nn.LayerNorm()(x)
        y = self.CausalSelfAttention(n_cntx=self.n_cntx,
                                     n_head=self.n_head)(y, seq_idx)
        x = x + y

        y = nn.LayerNorm()(x)
        y = nn.Dense(features=4 * C)(y)
        y = GELU(y)
        y = nn.Dense(features=C)(y)

        x = x + y

        return x


class FGPT(BaseGPT):
    C: ConfigDict
    Block: BaseBlock

    @staticmethod
    def MakeWithSHCSA(C: ConfigDict, shcsa: BaseSingleHeadCausalSelfAttention):

        csa = partial(FCausalSelfAttention,
                      SingleHeadCausalSelfAttention=shcsa)
        block = partial(FBlock, CausalSelfAttention=csa)

        return FGPT(C=C, Block=block)

    @staticmethod
    def Make(C: ConfigDict):
        return FGPT.MakeWithSHCSA(C, FSingleHeadCausalSelfAttention)

    @nn.compact
    def __call__(self, tok_idx: jax.Array, seq_idx: jax.Array):
        """
        Parameters
        ----------
        tok_idx : jax.Array
            The index of the current token in the vocabulary.
        seq_idx: jax.Array
            The index of the current token in the sequence.
        """
        # Token embeddings of shape [T, n_embd].
        tok_emb = nn.Embed(num_embeddings=self.C.vocab_size,
                           features=self.C.n_embd)(tok_idx)
        # Position embeddings of shape [T, n_embd].
        pos_emb = nn.Embed(num_embeddings=self.C.block_size,
                           features=self.C.n_embd)(seq_idx)

        x = tok_emb + pos_emb

        for _ in range(self.C.n_layer):
            x = self.Block(n_cntx=self.C.block_size,
                           n_head=self.C.n_head)(x, seq_idx)

        x = nn.LayerNorm()(x)
        logits = nn.Dense(features=self.C.vocab_size, use_bias=False)(x)

        return logits

    def __hash__(self):
        return id(self)

    def init_vars(self, params) -> Dict:
        """Initializes (and returns) the model's variables (i.e. the KV cache).
        
        Example usage::

            model = FGPT.Make(C)
            vars = model.init_vars({"params": gpt_params})
            logits, vars = model.apply({"params": gpt_params, **vars}, tok_idx, seq_idx)

        """
        _, vars = self.apply(params,
                             jnp.array(0),
                             jnp.array(0),
                             mutable="cache")
        return vars

    @partial(jax.jit, static_argnames=("self", "logit_sampler", "max_new_tokens"))
    def generate(self,
                 rng: jax.random.PRNGKey,
                 variables: Dict,
                 prompt_idxs: jax.Array,
                 logit_sampler: Callable[
                     [jax.random.PRNGKey, jax.Array],
                     jax.Array] = lambda _, logits: jnp.argmax(logits),
                 max_new_tokens: int = 10):
        """Auto-regressively generate tokens.

        Parameters
        ----------
        variables : Dict
            Variable dict for `apply` - should include the KV cache and model params.
        prompt_idxs : jax.Array
            Prompt sequence as indicies into the vocabulary. E.g. [59423, 233, 921].
        logit_sampler : Callable
            A function which accepts an array of logits of shape [n_embd] and return a
            single token index.
        max_new_tokens : int
            Max number of new tokens to generate.

        Returns
        -------
        token_idx : jax.Array
            The generated sequence of tokens - as vocab indicies. Shape 
            [len(prompt_idx) + max_new_tokens].

        Example usage::
            
                model = FGPT.Make(C)
                vars = model.init_vars({"params": gpt_params})
                token_idx = model.generate(vars, prompt_idx)
        """
        assert len(prompt_idxs) > 0

        # For loop carry state.
        init_val = {
            "rng": rng,
            # "variables" : { "params": ..., "cache": ... }
            "variables": variables,
            "seq": jnp.pad(prompt_idxs, (0, max_new_tokens),
                           constant_values=0),
        }

        def body(seq_idx: int, val: Dict):
            logits, cache = self.apply(val["variables"],
                                       val["seq"][seq_idx],
                                       seq_idx,
                                       mutable="cache")

            # Sample the next token based on the logits - unless we are still processing
            # the prompt - then use the next token in the prompt. We start sampling when
            # seq_idx == len(prompt_idxs) - 1 - i.e. the current token is the last of the
            # prompt.
            next_tok_idx = lax.cond(
                seq_idx >= len(prompt_idxs) - 1,
                lambda: logit_sampler(val["rng"], logits),
                lambda: val["seq"][seq_idx + 1],
            )

            # Only advance the rng if it's used - for consistent results with other
            # implementations.
            rng = lax.cond(
                seq_idx >= len(prompt_idxs) - 1,
                lambda: jax.random.split(val["rng"])[0],
                lambda: val["rng"],
            )

            return {
                "rng": rng,
                "variables": {
                    **val["variables"],
                    **cache,
                },
                "seq": val["seq"].at[seq_idx + 1].set(next_tok_idx)
            }

        val = lax.fori_loop(0, len(prompt_idxs) + max_new_tokens - 1, body, init_val)

        return val["seq"]
