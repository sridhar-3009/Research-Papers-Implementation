import itertools
import random

import torch
import torch.nn.functional as F

from instructgpt import (LM, IDX, collect_rankings, correct_response, labeler_utility, make_instruction, pretrain,
                         prompt_tokens, ranking_loss, sft, train_reward_model, web_document, ppo, benchmark)


def test_tasks_and_utility():
    assert correct_response("SORT", list("dbca")) == list("abcd")
    assert correct_response("FIRST2", list("dbca")) == list("db") and correct_response("LAST2", list("dbca")) == list("ca")
    assert labeler_utility("SORT", list("dbca"), list("abcd")) == 3.0
    assert labeler_utility("SORT", list("dbca"), list("abc")) == 2 * 3 / 4 - 0.3
    rng = random.Random(0)
    docs = [web_document(rng) for _ in range(500)]
    instr = [d for d in docs if "SEP" in d]
    assert 0.1 < len(instr) / 500 < 0.3
    for d in instr:                                                              # web text never answers correctly
        task, args = d[1], d[2:d.index("SEP")]
        assert d[d.index("SEP") + 1:-1] != correct_response(task, args) or len(args) < 2


def test_ranking_loss_is_mean_over_all_pairs():
    torch.manual_seed(0)
    rm = LM(scalar=True)
    prompt = ("SORT", list("cab"))
    ranked = [list("abc"), list("acb"), list("cab"), list("c")]
    loss = ranking_loss(rm, prompt, ranked)
    from instructgpt import score
    r = score(rm, [prompt] * 4, ranked)
    manual = -torch.stack([F.logsigmoid(r[i] - r[j]) for i, j in itertools.combinations(range(4), 2)]).mean()
    assert torch.allclose(loss, manual, atol=1e-6)
    assert len(list(itertools.combinations(range(9), 2))) == 36                   # K = 9 -> C(9, 2) = 36 pairs


def test_pipeline_runs_and_ptx_term_is_used():
    base = pretrain(steps=5, batch=8)
    s = sft(base, steps=3, n_demos=8, batch=4)
    rk = collect_rankings(s, 4, K_range=(4, 5))
    assert all(4 <= len(r) <= 5 for _, r in rk)
    rm = train_reward_model(s, rk, epochs=1)
    p0, h0 = ppo(s, rm, beta=0.1, gamma_ptx=0.0, iters=1, batch=8, epochs=1)
    p1, h1 = ppo(s, rm, beta=0.1, gamma_ptx=1.0, iters=1, batch=8, epochs=1)
    assert h0[0][2] == 0.0 and 0.0 <= benchmark(p1, n=20) <= 1.0
    assert prompt_tokens("SORT", ["a"]) == ["BOS", "SORT", "a", "SEP"] and IDX["PAD"] == 0
