from ml_collections import ConfigDict


def get_config_for(model_name: str) -> ConfigDict:
    if model_name == "gpt2":
        C = ConfigDict()
        C.model_type = "gpt2"
        C.n_layer = 12
        C.n_head = 12
        C.n_embd = 768

        C.vocab_size = 50257
        C.block_size = 1024

        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C
    else:
        raise ValueError(f"Unknown model name: {model_name}")
