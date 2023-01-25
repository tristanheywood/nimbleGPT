from absl.testing import absltest

import jax
import jax.numpy as jnp
from absl.testing import absltest

from nimblegpt import get_config_for
from nimblegpt.model import SingleHeadCausalSelfAttention
from nimblegpt.triton_shcsa_kernels import (
    SHCSATritonSoftmax,
    SHCSATritonPaddedSoftmax,
    SHCSATritonPaddedSoftmaxV,
    SHCSATriton
)


class TritonSHCSATest(absltest.TestCase):
    def setUp(self):
        super().setUp()

        self.config = get_config_for("gpt2")
        self.n_feat = self.config.n_embd // self.config.n_head

        self.rng = jax.random.PRNGKey(0)
        self.x = jax.random.normal(self.rng, (4, self.config.n_embd))

        self.n_padd = 4
        self.padded_x = jnp.pad(self.x, ((self.n_padd, 0), (0, 0)), constant_values=0)

    def test_triton_softmax(self):

        y, _ = SingleHeadCausalSelfAttention(self.n_feat).init_with_output(
            self.rng, self.x
        )
        ty, _ = SHCSATritonSoftmax(self.n_feat).init_with_output(self.rng, self.x)
        jit_ty, _ = jax.jit(SHCSATritonSoftmax(self.n_feat).init_with_output)(
            self.rng, self.x
        )

        assert (y - ty).max() < 1e-6
        assert (y - jit_ty).max() < 1e-6

    def test_triton_padded_softmax(self):

        y, _ = SingleHeadCausalSelfAttention(self.n_feat).init_with_output(
            self.rng, self.x
        )
        ty, _ = SHCSATritonPaddedSoftmax(self.n_feat).init_with_output(self.rng, self.x)
        pty, _ = SHCSATritonPaddedSoftmax(self.n_feat).init_with_output(
            self.rng, self.padded_x, n_padd=self.n_padd
        )
        jit_pty, _ = jax.jit(SHCSATritonPaddedSoftmax(self.n_feat).init_with_output)(
            self.rng, self.padded_x, n_padd=self.n_padd
        )

        assert (y - ty).max() < 1e-6
        assert (y - pty[self.n_padd :]).max() < 1e-6
        assert (y - jit_pty[self.n_padd :]).max() < 1e-6

    def test_triton_padded_softmax_v(self):

        y, _ = SingleHeadCausalSelfAttention(self.n_feat).init_with_output(
            self.rng, self.x
        )
        ty, _ = SHCSATritonPaddedSoftmaxV(self.n_feat).init_with_output(
            self.rng, self.x
        )
        pty, _ = SHCSATritonPaddedSoftmaxV(self.n_feat).init_with_output(
            self.rng, self.padded_x, n_padd=self.n_padd
        )
        jit_pty, _ = jax.jit(SHCSATritonPaddedSoftmaxV(self.n_feat).init_with_output)(
            self.rng, self.padded_x, n_padd=self.n_padd
        )

        assert (y - ty).max() < 1e-6
        assert (y - pty[self.n_padd :]).max() < 1e-6
        assert (y - jit_pty[self.n_padd :]).max() < 1e-6

    def test_triton_shcsa(self):

        y, _ = SingleHeadCausalSelfAttention(self.n_feat).init_with_output(
            self.rng, self.x
        )
        ty, _ = SHCSATriton(self.n_feat).init_with_output(self.rng, self.x)
        pty, _ = SHCSATriton(self.n_feat).init_with_output(
            self.rng, self.padded_x, n_padd=self.n_padd
        )
        jit_pty, _ = jax.jit(SHCSATriton(self.n_feat).init_with_output)(
            self.rng, self.padded_x, n_padd=self.n_padd
        )

        assert (y - ty).max() < 1e-6
        assert (y - pty[self.n_padd :]).max() < 1e-6
        assert (y - jit_pty[self.n_padd :]).max() < 1e-6

    
