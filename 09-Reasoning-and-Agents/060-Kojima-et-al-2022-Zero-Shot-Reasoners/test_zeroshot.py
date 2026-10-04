import random

from zeroshot import (ANSWER_TRIGGERS, TABLE_1, TABLE_4, TRIGGER, answer_prompt, cleanse, reasoning_prompt,
                      self_consistency, toy_answer, toy_document, toy_problem, train_toy, zero_shot_cot,
                      zero_shot_prompt)


def test_two_stage_prompts():
    q = "Joe throws 25 punches per minute. A fight lasts 5 rounds of 3 minutes. How many punches?"
    x1 = reasoning_prompt(q)
    assert x1 == f"Q: {q}\nA: Let's think step by step."
    x2 = answer_prompt(x1, "In 3 minutes, 75 punches. In 5 rounds, 375.")
    assert x2.startswith(x1) and x2.endswith("Therefore, the answer (arabic numerals) is")
    assert zero_shot_prompt(q).endswith("The answer (arabic numerals) is")


def test_cleansing_takes_the_first_fitting_piece():
    assert cleanse(" probably 375 and 376.") == "375"
    assert cleanse(" $1,250.") == "1250"
    assert cleanse(" B, C, and D.", "multiple choice") == "B"
    assert cleanse(" Yes, because...", "yes/no") == "yes"
    assert cleanse(" none", "number") is None


def test_pipeline_with_a_fake_model_and_self_consistency():
    calls = []

    def fake(prompt):
        calls.append(prompt)
        return " 3 * 25 = 75. 5 * 75 = 375." if prompt.endswith(TRIGGER) else " 375."
    ans, z, y = zero_shot_cot(fake, "How many punches?")
    assert ans == "375" and len(calls) == 2 and ANSWER_TRIGGERS["number"] in calls[1] and z in calls[1]
    assert self_consistency(["7", "3", "7", None, "2"]) == "7" and self_consistency([None]) is None


def test_table_4_ordering():
    best = max(TABLE_4, key=lambda r: r[2])
    assert best[1] == "Let's think step by step." and best[2] == 78.7
    instructive = [a for c, _, a in TABLE_4 if c == "instructive"]
    others = [a for c, _, a in TABLE_4 if c in ("misleading", "irrelevant")]
    assert min(instructive) > max(others)
    assert TABLE_1["MultiArith"] == (17.7, 78.7) and TABLE_1["CommonsenseQA"][1] < TABLE_1["CommonsenseQA"][0]


def test_toy_corpus_and_pipeline_shapes():
    rng = random.Random(0)
    q, run = toy_problem(rng, 4)
    assert sum(int(c) for c in q[1:-1]) % 10 == run[-1]
    kinds = {toy_document(rng)[1][0] for _ in range(400)}
    assert kinds == {">", "T", "R", "X"}
    m, loss = train_toy(steps=40, d=32, batch=16)
    ans, z = toy_answer(m, q, "T")
    assert len(ans) <= 1 and loss < 3.5
