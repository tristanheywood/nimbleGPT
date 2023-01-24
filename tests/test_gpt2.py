"""Test that various implementations of GPT2 are equivalent."""

import flaxmodels.gpt2 as fm_gpt2
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from nimblegpt import GPT, get_config_for, make_gpt_param_dict, JGPT


class GPT2Test(absltest.TestCase):
    """Test nimbleGPT GPT2 against flaxmodels GPT2."""

    def setUp(self):
        super().setUp()

        self.config = get_config_for("gpt2")

        self.rng = jax.random.PRNGKey(0)
        self.fm_gpt_module = fm_gpt2.GPT2LMHeadModel(pretrained="gpt2")
        self.fm_gpt_params = self.fm_gpt_module.init(
            self.rng, input_ids=jnp.array([0])
        )["params"]

        self.x = jax.random.randint(self.rng, (3,), 0, self.config.vocab_size)
        self.n_padd = 2
        self.padded_x = jnp.pad(
            self.x, (0, self.n_padd), constant_values=self.config.vocab_size - 1
        )

        self.fm_y = self.fm_gpt_module.apply({"params": self.fm_gpt_params}, self.x)[
            "logits"
        ]

        self.gpt_params = make_gpt_param_dict(self.fm_gpt_params, self.config)
        self.jgpt_params = make_gpt_param_dict(
            self.fm_gpt_params, self.config, prepend="J"
        )

    def test_model_against_fm(self):
        """Test that nimbleGPT GPT2 is identical (or at least close to) to flaxmodels GPT2."""

        y = GPT(self.config).apply({"params": self.gpt_params}, self.x)

        # Difference is possibly due to the activation function.
        assert (y - self.fm_y).max() < 1e-3

    def test_jmodel_basic(self):
        """Test model against jmodel with no padding and no jit."""

        y = GPT(self.config).apply({"params": self.gpt_params}, self.x)
        jy = JGPT(self.config).apply({"params": self.jgpt_params}, self.x)

        assert (y - jy).max() == 0

    def test_jmodel_jitted(self):
        """Test model against jmodel with no padding and jit."""

        y = GPT(self.config).apply({"params": self.gpt_params}, self.x)
        jy = jax.jit(JGPT(self.config).apply)({"params": self.jgpt_params}, self.x)

        # Jitting gives slightly different results.
        assert (y - jy).max() < 1e-4

    def test_jmodel_padded(self):
        """Test that jmodel returns the same logits regardless of padding tokens."""

        jy = JGPT(self.config).apply({"params": self.jgpt_params}, self.x)
        pjy = JGPT(self.config).apply(
            {"params": self.jgpt_params}, self.padded_x, n_padd=self.n_padd
        )

        assert (jy - pjy[self.n_padd:]).max() == 0


if __name__ == "__main__":
    absltest.main()
