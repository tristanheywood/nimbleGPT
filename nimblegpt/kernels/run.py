"""CLI script for executing specific parts of nimbleGPT. Designed to be used with Nsight
Systems, since Nsight must profile a single process."""

import argparse

import jax
import jax.numpy as jnp

from nimblegpt.config import get_config_for
from nimblegpt.jgenerate import jitted_text_generator
from nimblegpt.kernels.pmodel import PGPT, BaseSingleHeadCausalSelfAttention
from nimblegpt.kernels.triton_shcsa_kernels import (
    SHCSATriton,
    SHCSATritonPaddedSoftmax,
    SHCSATritonPaddedSoftmaxV,
    SHCSATritonSoftmax,
)

TRITON_SHCSAS = [
    SHCSATritonSoftmax,
    SHCSATritonPaddedSoftmax,
    SHCSATritonPaddedSoftmaxV,
    SHCSATriton,
]

TRITON_NAME_TO_SHCSA = {k.__name__: k for k in TRITON_SHCSAS}


def run(SHCSAModule: BaseSingleHeadCausalSelfAttention):

    config = get_config_for("gpt2")
    rng = jax.random.PRNGKey(0)

    gpt_module = PGPT.MakeWithSHCSA(config, SHCSAModule)
    params = gpt_module.init(rng, jnp.zeros((config.block_size,), jnp.int), n_padd=0)
    prompt = "The"
    generate_text = jitted_text_generator(config, gpt_module)

    text = generate_text(
        rng, params, prompt, max_new_tokens=1000
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--triton-shcsa",
        help=f"Triton single-headed causal self-attention kernel to execute. Must be one of {TRITON_NAME_TO_SHCSA.keys()}.",
        type=str,
    )

    args = parser.parse_args()
    module = TRITON_NAME_TO_SHCSA[args.triton_shcsa]

    run(module)


if __name__ == "__main__":
    main()
