import random

from verifiers import (Verifier, calculator_fix, calculator_step, final_answer, majority_vote, sample_solutions,
                       select_by_verifier, strip_annotations, coverage_at_n, toy_correct, toy_problem, train_generator)

SOL = ("Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
       "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72")


def test_gsm8k_format_and_calculator():
    assert final_answer(SOL) == "72" and final_answer("no answer") is None
    assert final_answer("#### 1,250") == "1250"
    wrong = SOL.replace("<<48+24=72>>", "<<48+24=71>>")
    assert calculator_fix(wrong) == SOL
    assert calculator_step("She has 3*4 = <<3*4=") == "12" and calculator_step("no calc") is None
    assert calculator_step("<<7/2=") == "3.5"
    assert "<<" not in strip_annotations(SOL)


def test_selection_rules():
    correct = [[False, True, False], [False, False, False], [True, True, True]]
    assert coverage_at_n(correct, 1) == 1 / 3 and coverage_at_n(correct, 2) == 2 / 3
    answers, scores = ["5", "7", "7", "9"], [0.9, 0.8, 0.7, 0.1]
    assert select_by_verifier(answers, scores) == "5"
    assert select_by_verifier(answers, scores, top_k=3) == "7"                   # the top 3 vote: 5, 7, 7
    assert majority_vote(["1", None, "2", "2"]) == "2"


def test_toy_problem_and_checker():
    rng = random.Random(0)
    for _ in range(50):
        p = toy_problem(rng)
        assert toy_correct(p, p["gold"])
        assert p["q"].startswith("Q") and p["q"].endswith("=") and f"t{p['target']}" in p["q"]
    p = {"digits": [1, 2, 3, 4, 5, 9], "target": 10}
    assert toy_correct(p, "1+9;") and not toy_correct(p, "5+5;") and not toy_correct(p, "3+7;") and not toy_correct(p, "x;")


def test_generator_sampling_and_verifier_run():
    gen = train_generator(steps=20, batch=16)
    ps = [toy_problem(random.Random(i)) for i in range(4)]
    sols = sample_solutions(gen, [p["q"] for p in ps], n=3)
    assert len(sols) == 4 and all(len(s) == 3 and s[0].endswith(";") for s in sols)
    data = [(p["q"], a, toy_correct(p, a)) for p, ss in zip(ps, sols) for a in ss]
    ver = Verifier(gen).train(data, steps=3, batch=8)
    sc = ver.score(ps[0]["q"], sols[0])
    assert len(sc) == 3
    ver2 = Verifier(gen, token_level=False, joint=False).train(data, steps=2, batch=8)
    assert len(ver2.score(ps[1]["q"], sols[1])) == 3
