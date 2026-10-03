import math
import random
import re

import torch

from gpt3 import (PFS_DAY, TABLE_2_1, TABLE_2_2, arithmetic_example, bayes_optimal_nll, build_prompt, choose, dedup,
                  dirichlet_sequences, epochs_elapsed, estimated_jaccard, is_dirty, jaccard, layer_mask, minhash,
                  ngram_set, param_estimate, pareto_keep, sample_source, scramble, shingles, training_flops)


def test_table_2_1_parameter_counts():
    for name, (n, L, d, h, dh, _, _) in TABLE_2_1.items():
        assert abs(param_estimate(L, d) / n - 1) < 0.02, name                    # within 2% of the paper's labels
        assert h * dh == d or name in ("XL", "13B")       # the paper's table: XL 24 x 128 = 3072 != 2048; 13B 5120 != 5140


def test_compute_175b():
    f = training_flops(175e9, 300e9)
    assert abs(f - 3.15e23) < 1e21
    assert abs(f / PFS_DAY - 3640) < 20                                           # 'about 3,640 petaflop/s-days'


def test_alternating_dense_and_banded_attention():
    dense, banded = layer_mask(0, 8), layer_mask(1, 8, window=3)
    assert torch.equal(dense, torch.tril(torch.ones(8, 8, dtype=torch.bool)))
    assert banded[7].nonzero().flatten().tolist() == [5, 6, 7]
    assert not torch.any(torch.triu(banded, 1))


def test_data_mixture():
    e = epochs_elapsed()
    assert abs(e["Common Crawl (filtered)"] - 0.44) < 0.01 and abs(e["Books2"] - 0.44) < 0.01
    assert abs(sum(w for _, w in TABLE_2_2.values()) - 1.01) < 1e-9                  # the paper's weights round to 101%
    rng = random.Random(0)
    draws = [sample_source(rng) for _ in range(20000)]
    assert abs(draws.count("Common Crawl (filtered)") / 20000 - 0.60 / 1.01) < 0.02


def test_pareto_filter_keeps_by_quality():
    rng = random.Random(0)
    keep = lambda s: sum(pareto_keep(s, rng) for _ in range(20000)) / 20000
    assert abs(keep(0.9) - 1.1 ** -9) < 0.02                                       # P(Lomax(9) > 0.1)
    assert abs(keep(0.1) - 1.9 ** -9) < 0.003
    assert keep(1.0) == 1.0


def test_minhash_estimates_jaccard_and_dedups():
    a = "the quick brown fox jumps over the lazy dog again and again in the park today"
    b = a.replace("today", "tonight")
    c = "an entirely different sentence about cooking pasta with fresh tomatoes and basil leaves"
    j = jaccard(shingles(a), shingles(b))
    assert abs(estimated_jaccard(minhash(a, 400), minhash(b, 400)) - j) < 0.08
    assert dedup([a, b, c]) == [a, c]


def test_contamination_ngrams():
    train = ngram_set("one two three four five six seven eight nine ten eleven twelve thirteen fourteen", 13)
    assert is_dirty("zero one two three four five six seven eight nine ten eleven twelve thirteen", train)
    assert not is_dirty("one two three four five six seven eight nine ten eleven twelve", train)


def test_prompts_and_multiple_choice_scoring():
    demos = [("2 + 2", "4"), ("3 + 5", "8")]
    assert build_prompt("Add the numbers.", demos, "1 + 6", 0) == "Add the numbers.\n\n1 + 6 =>"
    assert build_prompt("", demos, "1 + 6", 2) == "2 + 2 => 4\n\n3 + 5 => 8\n\n1 + 6 =>"
    table = {("ctx", "a"): (-2.0, 1), ("ctx", "b b b"): (-3.0, 3), ("Answer:", "a"): (-1.0, 1),
             ("Answer:", "b b b"): (-6.0, 3)}
    lp = lambda c, x: table[(c, x)]
    assert choose(lp, "ctx", ["a", "b b b"], "sum") == 0                           # -2 > -3
    assert choose(lp, "ctx", ["a", "b b b"], "per_token") == 1                     # -1 per token beats -2
    assert choose(lp, "ctx", ["a", "b b b"], "unconditional") == 1                 # -3 + 6 = 3 beats -2 + 1 = -1


def test_synthetic_tasks():
    rng = random.Random(0)
    q, a = arithmetic_example(rng, 2, "+")
    x, y = map(int, re.findall(r"\d+", q))
    assert a == f"A: {x + y}"
    w = "inevitably"
    assert sorted(scramble(w, "CL", rng)) == sorted(w)
    a1 = scramble(w, "A1", rng)
    assert a1[0] == w[0] and a1[-1] == w[-1] and sorted(a1) == sorted(w)
    assert scramble(w, "RW", rng) == w[::-1]
    assert scramble(w, "RI", rng).replace(" ", "")[::2].startswith("i")


def test_bayes_optimal_predictor_is_the_floor():
    torch.manual_seed(0)
    x = dirichlet_sequences(2000, 20, 5, 0.5)
    b = bayes_optimal_nll(x, 5, 0.5)
    assert abs(b[:, 0].mean().item() - math.log(5)) < 0.05                          # nothing seen yet: uniform
    assert b[:, -1].mean() < b[:, 0].mean() - 0.2                                    # more context, better prediction

