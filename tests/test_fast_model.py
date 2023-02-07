from absl.testing import absltest
import jax
import flaxmodels.gpt2 as fm_gpt2
import jax.numpy as jnp
from nimblegpt.config import get_config_for
from nimblegpt.fast_model import SingleHeadQKV
from nimblegpt.params import make_gpt_param_dict
import flax.linen as nn


class FastModelTest(absltest.TestCase):
    def setUp(self):
        super().setUp()

        self.config = get_config_for("gpt2")
        self.n_cntx = self.config.block_size
        self.n_embd = self.config.n_embd
        self.n_feat = self.n_embd // self.config.n_head

        self.rng = jax.random.PRNGKey(0)
        self.fm_gpt_module = fm_gpt2.GPT2LMHeadModel(pretrained="gpt2")
        self.fm_gpt_params = self.fm_gpt_module.init(self.rng,
                                                     input_ids=jnp.array(
                                                         [0]))["params"]
        self.gpt_params = make_gpt_param_dict(self.fm_gpt_params, self.config)

        # self.X = jax.random.randint(self.rng, (5, ), 0, self.config.vocab_size)

    def test_SingleHeadQKV(self):

        X = jax.random.uniform(self.rng, (10, self.n_embd))
        shqkv_module = SingleHeadQKV(self.n_cntx, self.n_feat)

        # Params for the first casual self attention block.
        sa_0_params = self.gpt_params["Block_0"]["CausalSelfAttention_0"][
            "VmapSingleHeadCausalSelfAttention_0"]
        # Params for the first head of the first casual self attention block.
        sh_0_params = jax.tree_map(lambda x: x[0], sa_0_params)

        # [10, 3 * n_feat]. Compute QKV for the whole sequence in one hit.
        QKV = nn.Dense(features=3 * self.n_feat).apply(
            {"params": sh_0_params["Dense_0"]}, X)

        # Init the KV cache.
        vars = shqkv_module.init(self.rng, X[0])

        for i in range(10):
            qKVi, vars = shqkv_module.apply(
                {
                    "cache": vars["cache"],
                    "params": sh_0_params
                },
                X[i],
                mutable="cache")

        # q, k, v for the last token in the sequence.
        q = qKVi[0]
        k = qKVi[1][9]
        v = qKVi[2][9]

        # q, k, v for the last token in the sequence is close.
        assert (QKV[9] - jnp.concatenate([q, k, v])).max() < 1e-5

        K = jnp.split(QKV, 3, axis=1)[1]
        V = jnp.split(QKV, 3, axis=1)[2]

        # K, V for the whole sequence is close.
        assert (K - qKVi[1][:10]).max() < 1e-5
        assert (V - qKVi[2][:10]).max() < 1e-5
