"""Test that various implementations of GPT2 are equivalent."""

import flaxmodels.gpt2 as fm_gpt2
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from nimblegpt import GPT, get_config_for, make_gpt_param_dict, JGPT


class ModelTest(absltest.TestCase):
    """Test nimbleGPT GPT2 against flaxmodels GPT2."""
    def setUp(self):
        super().setUp()

        self.config = get_config_for("gpt2")

        self.rng = jax.random.PRNGKey(0)
        self.fm_gpt_module = fm_gpt2.GPT2LMHeadModel(pretrained="gpt2")
        self.fm_gpt_params = self.fm_gpt_module.init(self.rng,
                                                     input_ids=jnp.array(
                                                         [0]))["params"]

        self.x = jax.random.randint(self.rng, (3, ), 0, self.config.vocab_size)
        self.fm_y = self.fm_gpt_module.apply({"params": self.fm_gpt_params},
                                             self.x)["logits"]

        self.gpt_params = make_gpt_param_dict(self.fm_gpt_params, self.config)

    def test_model_against_fm(self):
        """Test that nimbleGPT GPT2 is identical (or at least close to) to flaxmodels GPT2."""

        y = GPT(self.config).apply({"params": self.gpt_params}, self.x)

        # Difference is possibly due to the activation function.
        assert (y - self.fm_y).max() < 0.1


if __name__ == "__main__":
    absltest.main()
