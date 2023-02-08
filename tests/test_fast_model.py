from absl.testing import absltest
import jax
import flaxmodels.gpt2 as fm_gpt2
import jax.numpy as jnp
from nimblegpt.config import get_config_for
from nimblegpt.fast_model import FGPT, FCausalSelfAttention, FSingleHeadCausalSelfAttention
from nimblegpt.params import make_gpt_param_dict
import flax.linen as nn
from nimblegpt.model import GPT, CausalSelfAttention, SingleHeadCausalSelfAttention


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

    def test_get_qKV(self):
        """Test that get_qKV (which processes one token embedding at a time and 
        caches K and V) works the same as computing QKV in one hit from the whole 
        sequence."""

        X = jax.random.uniform(self.rng, (5, self.n_embd))
        module = FSingleHeadCausalSelfAttention(n_cntx=self.n_cntx,
                                                n_feat=self.n_feat)

        # Params for the first casual self attention block.
        sa_0_params = self.gpt_params["Block_0"]["CausalSelfAttention_0"][
            "VmapSingleHeadCausalSelfAttention_0"]
        # Params for the first head of the first casual self attention block.
        sh_0_params = jax.tree_map(lambda x: x[0], sa_0_params)

        # [5, 3 * n_feat]. Compute QKV for the whole sequence in one hit.
        QKV = nn.Dense(features=3 * self.n_feat).apply(
            {"params": sh_0_params["Dense_0"]}, X)

        # Init the KV cache.
        vars = module.init(self.rng, X[0], jnp.array(0), method="get_qKV")

        for i in range(5):
            qKV, vars = module.apply(
                {
                    "cache": vars["cache"],
                    "params": sh_0_params
                },
                x=X[i],
                seq_idx=i,
                mutable="cache",
                method="get_qKV",
            )

        # q, k, v for the last token in the sequence.
        q = qKV[0]
        k = qKV[1][4]
        v = qKV[2][4]

        # q, k, v for the last token in the sequence is close.
        assert (QKV[4] - jnp.concatenate([q, k, v])).max() < 1e-5

        K = jnp.split(QKV, 3, axis=1)[1]
        V = jnp.split(QKV, 3, axis=1)[2]

        # K, V for the whole sequence is close.
        assert (K - qKV[1][:5]).max() < 1e-5
        assert (V - qKV[2][:5]).max() < 1e-5

    def test_shcsa(self):
        """Test FSingleHeadCausalSelfAttention."""

        X = jax.random.uniform(self.rng, (5, self.n_embd))
        module = FSingleHeadCausalSelfAttention(n_cntx=self.n_cntx,
                                                n_feat=self.n_feat)

        # Params for the first casual self attention block.
        sa_0_params = self.gpt_params["Block_0"]["CausalSelfAttention_0"][
            "VmapSingleHeadCausalSelfAttention_0"]
        # Params for the first head of the first casual self attention block.
        sh_0_params = jax.tree_map(lambda x: x[0], sa_0_params)

        Y = SingleHeadCausalSelfAttention(n_feat=self.n_feat).apply(
            {"params": sh_0_params}, X)

        # Init the KV cache.
        vars = module.init(self.rng, X[0], jnp.array(0))

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

        # Outputs are close to `model.py`.
        assert (Y - jnp.array(fys)).max() < 1e-5

    def test_csa(self):
        """Test FCausalSelfAttention."""

        X = jax.random.uniform(self.rng, (5, self.n_embd))
        module = FCausalSelfAttention(
            n_cntx=self.n_cntx,
            n_head=self.config.n_head,
            SingleHeadCausalSelfAttention=FSingleHeadCausalSelfAttention)

        # Params for the first casual self attention block.
        sa_0_params = self.gpt_params["Block_0"]["CausalSelfAttention_0"]

        Y = CausalSelfAttention(n_head=self.config.n_head).apply(
            {"params": sa_0_params}, X)

        # Init the KV cache.
        vars = module.init(self.rng, X[0], jnp.array(0))

        fys = []
        for i in range(5):
            fy, vars = module.apply(
                {
                    "cache": vars["cache"],
                    "params": sa_0_params
                },
                x=X[i],
                seq_idx=i,
                mutable="cache",
            )
            fys.append(fy)

        # Outputs are close to `model.py`.
        assert (Y - jnp.array(fys)).max() < 1e-4

    def test_fgpt_logits(self):
        """Test logit output of FGPT - for a single token."""

        tok_idx = jax.random.randint(self.rng, (), 0, self.config.vocab_size)
        module = FGPT.Make(self.config)

        logits = GPT(self.config).apply({"params": self.gpt_params},
                                        jnp.expand_dims(tok_idx, axis=0))

        # Init the KV cache.
        vars = module.init(self.rng, tok_idx, jnp.array(0))

        flogits, _ = module.apply(
            {
                "cache": vars["cache"],
                "params": self.gpt_params
            },
            tok_idx=tok_idx,
            seq_idx=jnp.array(0),
            mutable="cache",
        )

        # Logits are close to `model.py`.
        assert (logits - flogits).max() < 1e-4

    def test_fgpt_generation(self):
        """Test FGPT auto-regressive generation."""

        prompt_idx = jax.random.randint(self.rng, (3, ), 0,
                                        self.config.vocab_size)
        gpt_module = GPT(self.config)
        module = FGPT.Make(self.config)

        seq = prompt_idx
        for _ in range(3):
            logits = gpt_module.apply({"params": self.gpt_params}, seq)
            seq = jnp.concatenate([seq, logits.argmax(axis=-1)[-1:]], axis=-1)

        # Init the KV cache.
        vars = module.init_vars({"params": self.gpt_params})

        fseq = prompt_idx
        for seq_idx in range(5):
            flogits, vars = module.apply(
                {
                    "params": self.gpt_params,
                    **vars
                },
                tok_idx=fseq[seq_idx],
                seq_idx=jnp.array(seq_idx),
                mutable="cache",
            )
            if seq_idx >= 2:
                fseq = jnp.concatenate(
                    [fseq, jnp.expand_dims(flogits.argmax(), axis=0)], axis=-1)

        # Sequence is identical to `model.py`.
        assert (seq == fseq).all()

    def test_fgpt_jitted_generation(self):
        """Test FGPT auto-regressive generation with jit."""

        prompt_idxs = jax.random.randint(self.rng, (3, ), 0,
                                         self.config.vocab_size)

        gpt_module = GPT(self.config)
        module = FGPT.Make(self.config)

        seq = prompt_idxs
        for _ in range(3):
            logits = gpt_module.apply({"params": self.gpt_params}, seq)
            seq = jnp.concatenate([seq, logits.argmax(axis=-1)[-1:]], axis=-1)

        vars = module.init_vars({"params": self.gpt_params})
        fseq = module.generate(self.rng, {
            "params": self.gpt_params,
            **vars
        }, prompt_idxs, max_new_tokens=3)

        # Sequence is identical to `model.py`.
        assert (seq == fseq).all()
