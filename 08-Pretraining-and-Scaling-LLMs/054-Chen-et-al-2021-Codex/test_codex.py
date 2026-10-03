import random
from math import comb

import numpy as np

from codex import (BUILDING_BLOCKS, MINI_HUMANEVAL, TABLE_1, bleu, check_correctness, code_loss, nucleus, pass_at_k,
                   pass_at_k_comb, pass_at_k_naive, plain_tokens, synthetic_problem, truncate, whitespace_tokens)


def test_stable_estimator_equals_binomial_form():
    for n, c, k in [(200, 10, 1), (200, 10, 100), (200, 0, 5), (20, 19, 2), (10, 3, 8), (5, 5, 5)]:
        assert abs(pass_at_k(n, c, k) - pass_at_k_comb(n, c, k)) < 1e-9
    assert abs(pass_at_k(200, 37, 1) - 37 / 200) < 1e-12                       # pass@1 = c/n


def test_estimator_is_unbiased_and_naive_is_biased_low():
    """Appendix A: average over c ~ Binomial(n, p) exactly."""
    n, k, p = 20, 10, 0.1
    w = [comb(n, c) * p ** c * (1 - p) ** (n - c) for c in range(n + 1)]
    truth = 1 - (1 - p) ** k
    unbiased = sum(wi * pass_at_k(n, c, k) for c, wi in enumerate(w))
    naive = sum(wi * pass_at_k_naive(n, c, k) for c, wi in enumerate(w))
    assert abs(unbiased - truth) < 1e-12 and naive < truth - 0.05


def test_execution_harness_pass_fail_timeout_crash():
    for p in MINI_HUMANEVAL:
        assert check_correctness(p["prompt"], p["canonical"], p["test"], p["entry_point"]) == "passed"
    p = MINI_HUMANEVAL[1]
    assert check_correctness(p["prompt"], "    return l\n", p["test"], p["entry_point"]) == "failed: assertion"
    assert check_correctness(p["prompt"], "    return l[\n", p["test"], p["entry_point"]) == "failed: SyntaxError"
    assert check_correctness(p["prompt"], "    while True: pass\n", p["test"], p["entry_point"], timeout=0.3) == "timed out"


def test_stop_sequences():
    s = "    return x + 1\n\ndef helper():\n    pass\n"
    assert truncate(s) == "    return x + 1\n"
    assert truncate("    return 1\n\nprint(f())") == "    return 1\n"
    assert truncate("    return 1") == "    return 1"


def test_nucleus_keeps_smallest_top_set():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    q = nucleus(p, 0.9)                                                          # needs 0.5 + 0.3 + 0.15
    assert q[3] == 0 and abs(q.sum() - 1) < 1e-12 and abs(q[0] - 0.5 / 0.95) < 1e-12


def test_whitespace_runs_shrink_indented_code():
    code = "def f(x):\n        if x:\n            return 1\n        return 0\n"
    assert "".join(whitespace_tokens(code)) == code == "".join(plain_tokens(code))
    assert len(whitespace_tokens(code)) < 0.6 * len(plain_tokens(code))


def test_bleu_can_prefer_a_wrong_program():
    ref = MINI_HUMANEVAL[1]["canonical"]
    wrong = "    return [x - 1 for x in l]\n"                                       # one token off, functionally wrong
    right = "    out = []\n    for v in l:\n        out.append(v + 1)\n    return out\n"
    p = MINI_HUMANEVAL[1]
    assert check_correctness(p["prompt"], wrong, p["test"], p["entry_point"]) != "passed"
    assert check_correctness(p["prompt"], right, p["test"], p["entry_point"]) == "passed"
    assert bleu(wrong, ref) > bleu(right, ref)
    assert abs(bleu(ref, ref) - 1) < 1e-12


def test_building_blocks_and_synthetic_problems():
    assert len(BUILDING_BLOCKS) == 13
    rng = random.Random(0)
    for n in (1, 3, 6):
        pr = synthetic_problem(n, rng)
        assert len(pr["blocks"]) == n and pr["prompt"].count("\n    -") == n
        assert check_correctness(pr["prompt"], pr["canonical"], pr["test"], pr["entry_point"]) == "passed"


def test_paper_numbers():
    assert TABLE_1["Codex-12B"] == (28.81, 46.81, 72.31)
    assert all(a <= b <= c for a, b, c in TABLE_1.values())                    # pass@k grows with k
    assert abs(code_loss(5.92e7) - 1) < 1e-12 and code_loss(12e9) < code_loss(12e6)
