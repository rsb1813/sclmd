# Small random Gemma 4 models from transformers, shared by the tests.
import pytest
import torch

TEXT = dict(
    vocab_size=300, hidden_size=48, intermediate_size=96, num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
    head_dim=8, global_head_dim=16, num_global_key_value_heads=1, sliding_window=5, max_position_embeddings=512,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2, attention_k_eq_v=True,
    hidden_size_per_layer_input=0, final_logit_softcapping=30.0, pad_token_id=0, eos_token_id=1, bos_token_id=2,
)


def randomize(model, seed: int):
    """Random values everywhere, also in the norms and layer scalars that start at one."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        scalars = [(n, b) for n, b in model.named_buffers() if n.endswith("layer_scalar")]  # other buffers are constants
        for name, p in list(model.named_parameters()) + scalars:
            if p.dim() == 1:
                p.copy_(0.5 + torch.rand(p.shape, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    return model


@pytest.fixture(scope="session")
def small():
    """(transformers model, its text config as a dict), float32 on the CPU."""
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    config = Gemma4TextConfig(**TEXT)
    model = randomize(Gemma4ForCausalLM(config).eval(), 0)
    model.lm_head.weight = model.model.embed_tokens.weight
    return model, config.to_dict()


@pytest.fixture(scope="session")
def small_drafter():
    """(transformers drafter for the `small` model, its text config as a dict)."""
    from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4TextConfig

    text = dict(TEXT, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_kv_shared_layers=4, final_logit_softcapping=None,
                layer_types=["sliding_attention"] * 3 + ["full_attention"], vocab_size_per_layer_input=0)
    config = Gemma4AssistantConfig(text_config=Gemma4TextConfig(**text), backbone_hidden_size=TEXT["hidden_size"])
    model = randomize(Gemma4AssistantForCausalLM(config).eval(), 1)
    model.lm_head.weight = model.model.embed_tokens.weight
    return model, config.text_config.to_dict()
