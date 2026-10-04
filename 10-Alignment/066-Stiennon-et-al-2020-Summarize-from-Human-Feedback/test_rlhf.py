import random

import torch
import torch.nn.functional as F

from rlhf import (IDX, LM, POST_LEN, collect_comparisons, encode, logprobs, make_post, ppo, reference_summary,
                  reward, sample, train_reward_model, train_sft, true_quality)


def test_task_and_true_quality():
    rng = random.Random(0)
    post, top = make_post(rng)
    assert len(post) == POST_LEN and post.count(top[0]) == 4 and post.count(top[1]) == 3
    best = true_quality(post, top, [top[0], top[1]])
    assert best == 2.3
    assert true_quality(post, top, [top[0]]) == 1.0
    assert true_quality(post, top, [top[0], top[0]]) < true_quality(post, top, [top[0]])  # repetition penalised
    missing = next(f"T{i}" for i in range(16) if f"T{i}" not in post)
    assert true_quality(post, top, [top[0], top[1], missing]) == best - 1.0                # hallucination -1
    refs = [len(reference_summary(top, rng)) for _ in range(1000)]
    assert 0.45 < refs.count(1) / 1000 < 0.55


def test_bradley_terry_loss_matches_paper_formula():
    ra, rb = torch.tensor([2.0, -1.0]), torch.tensor([0.5, 0.0])
    y = torch.tensor([1.0, 0.0])                                                  # first: A preferred; second: B
    paper = -(torch.log(torch.sigmoid(ra[0] - rb[0])) + torch.log(torch.sigmoid(rb[1] - ra[1]))) / 2
    assert torch.allclose(F.binary_cross_entropy_with_logits(ra - rb, y), paper)


def test_encode_sample_logprobs_shapes():
    rng = random.Random(1)
    post, top = make_post(rng)
    ids = encode(post, [top[0]])
    assert ids[0] == IDX["POST"] and ids[POST_LEN + 1] == IDX["SUM"] and ids[-1] == IDX["EOS"]
    torch.manual_seed(0)
    m = LM()
    out = sample(m, [post, post], temperature=1.0, gen=torch.Generator().manual_seed(0))
    assert len(out) == 2 and all(isinstance(t, str) for s in out for t in s)
    lp, mask = logprobs(m, [ids])
    assert lp.shape == mask.shape and mask.sum() == 2                              # one summary token + EOS
    assert (lp <= 0).all()


def test_pipeline_runs_end_to_end():
    sft = train_sft(steps=5, batch=8)
    data = collect_comparisons(sft, 16)
    assert all(d[3] in (0.0, 1.0) for d in data)
    rm = train_reward_model(sft, data, steps=3, batch=8)
    rng = random.Random(9)
    refs = [make_post(rng) for _ in range(256)]
    with torch.no_grad():
        mean_ref = float(reward(rm, [p for p, _ in refs], [reference_summary(t, rng) for _, t in refs]).mean())
    assert abs(mean_ref) < 0.5                                                     # normalised so references ~ 0
    policy, hist = ppo(sft, rm, beta=0.1, iters=2, batch=8, epochs=1)
    assert len(hist) == 2 and hist[0][2] == 0.0                                    # first batch: policy == SFT, KL 0
