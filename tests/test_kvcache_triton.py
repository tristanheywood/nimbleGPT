from absl.testing import absltest
import jax
import flaxmodels.gpt2 as fm_gpt2
import jax.numpy as jnp

from nimblegpt.config import get_config_for
from nimblegpt.fast_model import FSingleHeadCausalSelfAttention
from nimblegpt.kernels.kvcache_triton_kernels import SHCSABlock
from nimblegpt.model import SingleHeadCausalSelfAttention
from nimblegpt.params import make_gpt_param_dict


class TritonSHCSAcacheKVTest(absltest.TestCase):
    def setUp(self):
        super().setUp()

        self.config = get_config_for("gpt2")
        self.n_cntx = self.config.block_size
        self.n_embd = self.config.n_embd
        self.n_feat = self.config.n_embd // self.config.n_head

        self.rng = jax.random.PRNGKey(0)
        self.fm_gpt_module = fm_gpt2.GPT2LMHeadModel(pretrained="gpt2")
        self.fm_gpt_params = self.fm_gpt_module.init(self.rng,
                                                     input_ids=jnp.array(
                                                         [0]))["params"]
        self.gpt_params = make_gpt_param_dict(self.fm_gpt_params, self.config)
        self.shcsa_x = jax.random.uniform(self.rng, (self.n_feat, ))

    def test_block(self):

        X = jax.random.uniform(self.rng, (5, self.n_embd))
        module = SHCSABlock(self.n_feat, self.n_cntx)

        # Params for the first casual self attention block.
        sa_0_params = self.gpt_params["Block_0"]["CausalSelfAttention_0"][
            "VmapSingleHeadCausalSelfAttention_0"]
        # Params for the first head of the first casual self attention block.
        sh_0_params = jax.tree_map(lambda x: x[0], sa_0_params)

        Y = SingleHeadCausalSelfAttention(n_feat=self.n_feat).apply(
            {"params": sh_0_params}, X)

        # Init the KV cache.
        vars = module.init(self.rng, X[0], jnp.array(0))

        ## Test first element of the sequence.
        fy, _ = module.apply(
            {
                "cache": vars["cache"],
                "params": sh_0_params
            },
            x=X[0],
            seq_idx=0,
            mutable="cache",
        )

        assert (Y[0] - fy).max() < 1e-5

        ## Test whole sequence.
        fys = []
        for i in range(5):
            fy, vars = module.apply(
                {
                    "cache": vars["cache"],
                    "params": sh_0_params
                },
                x=X[i],
                seq_idx=i,
                mutable="cache",
            )
            fys.append(fy)

        assert (Y - jnp.array(fys)).max() < 1e-5

