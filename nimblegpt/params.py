"""Utilities for restructuring pretrained parameters into the format/shapes required for
nimbleGPT."""
from typing import Dict

import flaxmodels.gpt2 as fm_gpt2
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict


def make_qkv_param_tensor(qkv_tensor: jnp.array, n_head):
    """
    Restructure a flaxmodels qkv parameter tensor so it can be used by
    `CausalSelfAttention`.
    """
    # [q1, ..., qn, k1, ..., kn, v1, ..., vn]
    qsksvs = jnp.array(qkv_tensor.split(n_head * 3, axis=-1))
    # [[q1, ..., qn], [k1, ..., kn], [v1, ..., vn]]
    qkv = jnp.array(qsksvs.split(3))
    # [3, n_head, ... , n_feat] -> [n_head, ... , 3 * n_feat]
    params = jnp.concatenate(qkv, axis=-1)
    return params


def make_attention_param_dict(fm_attn_params: Dict, config: ConfigDict):
    return {
        "Dense_0": fm_attn_params["Dense_1"],
        "VmapSingleHeadCausalSelfAttention_0": {
            "Dense_0": {
                "bias":
                make_qkv_param_tensor(fm_attn_params["Dense_0"]["bias"],
                                      config.n_head),
                "kernel":
                make_qkv_param_tensor(fm_attn_params["Dense_0"]["kernel"],
                                      config.n_head),
            }
        },
    }


def make_block_param_dict(fm_block_params: Dict, config: ConfigDict):
    return {
        "CausalSelfAttention_0":
        make_attention_param_dict(fm_block_params["GPT2SelfAttention_0"],
                                  config),
        **fm_block_params["GPT2MLP_0"],  # Dense_0, Dense_1
        **{k: v
           for k, v in fm_block_params.items() if "Layer" in k},  # "LayerNorm_0, LayerNorm_1"
    }


def make_gpt_param_dict(fm_gpt_params: Dict, config: ConfigDict):
    return {
        # Embed_0, Embed_1, LayerNorm_0
        **{
            k: v
            for k, v in fm_gpt_params["GPT2Model_0"].items() if "Block" not in k
        },
        "Dense_0": fm_gpt_params["Dense_0"],
        **{
            f"Block_{i}": make_block_param_dict(
                fm_gpt_params["GPT2Model_0"][f"GPT2Block_{i}"], config)
            for i in range(config.n_layer)
        },
    }


def get_flaxmodels_gpt2_params():

    fm_gpt_module = fm_gpt2.GPT2LMHeadModel(pretrained="gpt2")
    fm_gpt_params = fm_gpt_module.init(jax.random.PRNGKey(0),
                                       jnp.zeros((1, ),
                                                 dtype=jnp.int32))["params"]

    return fm_gpt_params
