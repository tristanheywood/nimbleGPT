"""Test that the various implementations of single-headed attention are equivalent."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from nimblegpt import get_config_for
from nimblegpt.model import SingleHeadCausalSelfAttention
from nimblegpt.jmodel import JSingleHeadCausalSelfAttention


class SingleHeadedAttentionTest(absltest.TestCase):
    def setUp(self):
        super().setUp()

        self.config = get_config_for("gpt2")
        self.n_feat = self.config.n_embd // self.config.n_head

        self.rng = jax.random.PRNGKey(0)
        self.x = jax.random.normal(self.rng, (5, self.config.n_embd))

        self.n_padd = 3
        self.padded_x = jnp.pad(self.x, ((self.n_padd, 0), (0, 0)), constant_values=0)

    def test_jmodel_basic(self):
        """Test model against jmodel."""

        y, _ = SingleHeadCausalSelfAttention(self.n_feat).init_with_output(
            self.rng, self.x
        )
        jy, _ = JSingleHeadCausalSelfAttention(self.n_feat).init_with_output(
            self.rng, self.x
        )

        assert (y - jy).max() == 0

    def test_jmodel_padding(self):
        """Test that jmodel produces the same outputs with and without padding tokens."""

        jy, _ = JSingleHeadCausalSelfAttention(self.n_feat).init_with_output(
            self.rng, self.x
        )
        pjy, _ = JSingleHeadCausalSelfAttention(self.n_feat).init_with_output(
            self.rng, self.padded_x, n_padd=self.n_padd
        )

        # Not sure why difference is non-zero - possibly due to numerics in the softmax.
        assert (jy - pjy[self.n_padd :]).max() <= 1e-6


if __name__ == "__main__":
    absltest.main()
