"""Utilities jitted text generating with nimbleGPT."""

from typing import Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.random import KeyArray

from .bpe import get_encoder
from .jmodel import JGPT


def sample_token(rng, logits: jnp.array, temperature=1.0, top_k=40):
    """
    Parameters
    ----------
    logits
      1D array of length `config.vocab_size` (50257 for gpt2).
    """
    logits = logits / temperature
    top_logits, top_indices = jax.lax.top_k(logits, k=top_k)
    samp_idx = jax.random.categorical(rng, top_logits)  # Index into `top_indices`.
    return top_indices[samp_idx]


def jitted_text_generator(
    config, module: nn.Module = JGPT, params=None, temperature=1.0, top_k=40
) -> Callable[[KeyArray, Array, int], Array]:
    """
    Creates a function with the signature:
        generate_text(rng: KeyArray, prompt: str, max_new_tokens: int = 20)
    which can be used to generate text from a trained model.

    If `params = None`, the signature is:
        generate_text(rng: KeyArray, params, prompt: str, max_new_tokens: int = 20)

    Passing `params` to this function is convenient but uses more GPU memory.
    """
    # Only gpt2 is supported for now.
    assert config.model_type == "gpt2", config.model_type

    encoder = get_encoder()

    @jax.jit
    def generate_tokens(
        rng: KeyArray, params, pad_tok_idxs: Array, n_padd: int, max_new_tokens=20
    ):
        def body(i: int, rng_and_tok_idxs: Tuple[KeyArray, Array]):
            rng, tok_idxs = rng_and_tok_idxs

            logits = module(config).apply(params, tok_idxs, n_padd=n_padd + i)
            next_tok_idx = sample_token(
                rng, logits[-1], temperature=temperature, top_k=top_k
            )

            rng = jax.random.split(rng)[0]
            tok_idxs = jnp.roll(tok_idxs, -1).at[-1].set(next_tok_idx)
            return rng, tok_idxs

        return jax.lax.fori_loop(0, max_new_tokens, body, (rng, pad_tok_idxs))[1]

    def generate_text(rng: KeyArray, params, prompt: str, max_new_tokens: int = 20):
        prompt_idxs = jnp.array(encoder.encode(prompt))
        n_padd = config.block_size - len(prompt_idxs)
        pad_prompt_idxs = jnp.pad(prompt_idxs, (n_padd, 0))
        pad_tok_idxs = generate_tokens(
            rng, params, pad_prompt_idxs, n_padd, max_new_tokens
        )
        text = encoder.decode(np.array(pad_tok_idxs[n_padd - max_new_tokens :]))
        return text

    if params is not None:
        return lambda rng, prompt, max_new_tokens: generate_text(
            rng, params, prompt, max_new_tokens
        )

    return generate_text
