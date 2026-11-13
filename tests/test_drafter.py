# The drafter against transformers' Gemma4Assistant on small random models.
import torch

from sclmd import model as M
from sclmd.drafter import Drafter


def setup(small, small_drafter, n=14):
    hf, cdict = small
    hf_d, ddict = small_drafter
    cfg, dcfg = M.Config.from_text_config(cdict), M.Config.from_text_config(ddict)
    ids = torch.randint(3, cfg.vocab, (1, n + 1), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        out = hf(input_ids=ids[:, :n], output_hidden_states=True, return_shared_kv_states=True)
    return hf, hf_d, cfg, Drafter.from_state(dcfg, hf_d.state_dict()), ids, out


def test_drafter_config_shares_every_layer(small, small_drafter):
    *_, drafter, _, _ = setup(small, small_drafter)
    assert drafter.cfg.kv_shared == drafter.cfg.n_layers == 4


def test_two_guesses_match_transformers(small, small_drafter):
    hf, hf_d, cfg, drafter, ids, out = setup(small, small_drafter)  # 14 keys: more than the drafter's window of 5 + 1
    n = ids.shape[1] - 1
    embed = hf.get_input_embeddings()
    kv = out.shared_kv_states
    shared = {t: (kv[t][0], kv[t][1], torch.arange(n)) for t in kv}
    token, hidden = ids[:, n:], out.hidden_states[-1][:, -1:]
    for _ in range(2):
        with torch.no_grad():
            want = hf_d(inputs_embeds=torch.cat([embed(token), hidden], dim=-1), position_ids=torch.tensor([[n]]),
                        shared_kv_states=dict(kv), attention_mask=None)
            logits, nxt = drafter.step(embed(token), hidden, torch.tensor([n - 1]), shared)
        assert torch.allclose(logits, want.logits, atol=2e-4, rtol=1e-4)
        assert torch.allclose(nxt, want.last_hidden_state, atol=2e-4, rtol=1e-4)
        token, hidden = want.logits.argmax(-1), want.last_hidden_state


def test_many_places_at_once_match_one_at_a_time(small, small_drafter):
    hf, _, cfg, drafter, ids, out = setup(small, small_drafter)
    n = ids.shape[1] - 1
    embed = hf.get_input_embeddings()
    kv = out.shared_kv_states
    last = torch.tensor([6, 9, n - 1])
    hidden = out.hidden_states[-1][:, last]
    tokens = ids[:, last + 1]
    with torch.no_grad():
        together, _ = drafter.step(embed(tokens), hidden, last, {t: (kv[t][0], kv[t][1], torch.arange(n)) for t in kv})
        for j, p in enumerate(last.tolist()):
            cut = {t: (kv[t][0][:, :, : p + 1], kv[t][1][:, :, : p + 1], torch.arange(p + 1)) for t in kv}  # only what was there then
            alone, _ = drafter.step(embed(tokens[:, j : j + 1]), hidden[:, j : j + 1], torch.tensor([p]), cut)
            assert torch.allclose(together[:, j], alone[:, 0], atol=1e-5)
