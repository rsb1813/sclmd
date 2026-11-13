# The reference decoder against transformers on a small random Gemma 4.
import torch

from sclmd import model as M


def setup(small):
    hf, cdict = small
    cfg = M.Config.from_text_config(cdict)
    return hf, cfg, hf.state_dict()


def test_config_reads_both_layer_types(small):
    _, cfg, _ = setup(small)
    assert cfg.head_dim == {M.SLIDING: 8, M.FULL: 16} and cfg.kv_heads == {M.SLIDING: 2, M.FULL: 1}
    assert cfg.last_of(M.SLIDING) == 4 and cfg.last_of(M.FULL) == 5


def test_whole_sequence_matches_transformers(small):
    hf, cfg, state = setup(small)
    ids = torch.randint(3, cfg.vocab, (2, 24), generator=torch.Generator().manual_seed(1))  # longer than the window of 5
    with torch.no_grad():
        want = hf(input_ids=ids, output_hidden_states=True)
        logits, last = M.forward(cfg, state, "model.", ids)
    assert torch.allclose(logits, want.logits, atol=2e-4, rtol=1e-4)
    assert torch.allclose(last, want.hidden_states[-1], atol=2e-4, rtol=1e-4)  # what the drafter is given


def test_token_by_token_matches_whole_sequence(small):
    _, cfg, state = setup(small)
    ids = torch.randint(3, cfg.vocab, (1, 20), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        whole, _ = M.forward(cfg, state, "model.", ids)
        cache, steps = M.Cache(), []
        steps.append(M.forward(cfg, state, "model.", ids[:, :7], cache)[0])  # a prompt, then one token at a time
        for t in range(7, 20):
            steps.append(M.forward(cfg, state, "model.", ids[:, t : t + 1], cache)[0])
    assert torch.allclose(torch.cat(steps, dim=1), whole, atol=2e-4, rtol=1e-4)


def test_cropped_cache_continues_as_if_never_extended(small):
    _, cfg, state = setup(small)
    ids = torch.randint(3, cfg.vocab, (1, 16), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        whole, _ = M.forward(cfg, state, "model.", ids)
        cache = M.Cache()
        M.forward(cfg, state, "model.", ids[:, :10], cache)
        M.forward(cfg, state, "model.", torch.randint(3, cfg.vocab, (1, 4)), cache)  # rejected draft tokens
        cache.crop(10)
        rest, _ = M.forward(cfg, state, "model.", ids[:, 10:], cache)
    assert torch.allclose(rest, whole[:, 10:], atol=2e-4, rtol=1e-4)


def test_shared_states_match_transformers(small):
    hf, cfg, state = setup(small)
    ids = torch.randint(3, cfg.vocab, (1, 12), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        want = hf(input_ids=ids, return_shared_kv_states=True).shared_kv_states
        shared = {}
        M.forward(cfg, state, "model.", ids, shared=shared)
    for t in (M.SLIDING, M.FULL):
        assert torch.allclose(shared[t][0], want[t][0], atol=1e-5) and torch.allclose(shared[t][1], want[t][1], atol=1e-5)
