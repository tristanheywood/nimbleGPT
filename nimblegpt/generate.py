"""Utilities for generating text with nimbleGPT."""

import jax
import jax.numpy as jnp
import numpy as np

from .bpe import get_encoder


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


def generate_tokens(rng, model, token_indices, max_new_tokens=20, temperature=1):

    for _ in range(max_new_tokens):
        logits = model(token_indices)

        idx_next = sample_token(rng, logits[-1], temperature=temperature)
        rng = jax.random.split(rng)[0]

        token_indices = jnp.concatenate(
            (token_indices, jnp.expand_dims(idx_next, axis=0)), axis=-1
        )

    return token_indices


def generate_text(rng, model, prompt, max_new_tokens=20, temperature=1):
    encoder = get_encoder()
    prompt_indices = jnp.array(encoder.encode(prompt))

    generated_indices = generate_tokens(
        rng, model, prompt_indices, max_new_tokens, temperature
    )

    generated_text = encoder.decode(np.array(generated_indices))
    return generated_text
