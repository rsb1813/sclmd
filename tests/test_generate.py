# Generation with drafts gives the text generation without drafts gives.
import torch

from sclmd import model as M
from sclmd.drafter import Drafter
from sclmd.generate import Picker, RefTarget, drafter_guess, generate


def target(small):
    hf, cdict = small
    return RefTarget(M.Config.from_text_config(cdict), hf.state_dict())


PROMPT = [2, 17, 45, 201, 9]


def oracle(truth: list, n_prompt: int, right: list, depth: int = 4):
    """A guesser that knows the real continuation and gets `right[i]` tokens of its i-th guess right."""
    calls = []

    def guess(token, hidden, length):
        at = length - n_prompt + 1  # index in `truth` of the first guessed token
        good = right[len(calls) % len(right)]
        calls.append(length)
        return [truth[at + j] if j < good and at + j < len(truth) else 3 + (truth[min(at + j, len(truth) - 1)] + 1) % 250 for j in range(depth)]

    return guess, calls


def test_greedy_text_is_the_same_with_any_mix_of_right_and_wrong_guesses(small):
    truth = generate(target(small), PROMPT, 40, Picker())
    for right in ([0], [4], [1, 0, 3, 4, 2]):
        guess, calls = oracle(truth, len(PROMPT), right)
        stats = {}
        assert generate(target(small), PROMPT, 40, Picker(), guess, stats=stats) == truth
        assert stats["passes"] == len(calls)
    assert len(calls) < 40 / 2  # guesses were kept: far fewer passes than tokens


def test_sampled_text_is_the_same_with_guesses(small):
    pick = Picker(temperature=1.0, top_k=64, top_p=0.95, seed=7)
    truth = generate(target(small), PROMPT, 30, pick)
    assert truth != generate(target(small), PROMPT, 30, Picker())  # sampling does change the text
    guess, _ = oracle(truth, len(PROMPT), [2, 0, 4, 1])
    assert generate(target(small), PROMPT, 30, pick, guess) == truth


def test_stop_token_ends_the_text_whether_guessed_or_not(small):
    truth = generate(target(small), PROMPT, 40, Picker())
    stop = truth[11]
    cut = truth[: truth.index(stop) + 1]
    assert generate(target(small), PROMPT, 40, Picker(), stop=(stop,)) == cut
    guess, _ = oracle(truth, len(PROMPT), [4])
    assert generate(target(small), PROMPT, 40, Picker(), guess, stop=(stop,)) == cut


def test_with_the_real_drafter_the_text_is_unchanged(small, small_drafter):
    hf_d, ddict = small_drafter
    drafter = Drafter.from_state(M.Config.from_text_config(ddict), hf_d.state_dict())
    truth = generate(target(small), PROMPT, 24, Picker())
    t = target(small)
    assert generate(t, PROMPT, 24, Picker(), drafter_guess(drafter, t, 6)) == truth
