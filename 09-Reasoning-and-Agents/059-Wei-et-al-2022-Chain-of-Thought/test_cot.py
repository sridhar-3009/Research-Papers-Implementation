import random

from cot import (EXEMPLARS, FORMATS, TABLE_2_GSM8K, chain_is_correct, cot_prompt, evaluate_toy, extract_answer,
                 standard_prompt, toy_answer, toy_item, train_toy)


def test_prompts_differ_only_by_the_chain():
    q = "A farmer has 3 cows and buys 4 more. How many cows?"
    s, c = standard_prompt(EXEMPLARS, q), cot_prompt(EXEMPLARS, q)
    assert s.endswith(f"Q: {q}\nA:") and c.endswith(f"Q: {q}\nA:")
    assert "5 + 6 = 11." in c and "5 + 6 = 11." not in s
    assert s.count("The answer is") == c.count("The answer is") == 2


def test_answer_extraction():
    assert extract_answer(" 2 cans of 3 is 6. 5 + 6 = 11. The answer is 11.") == "11"
    assert extract_answer("so 1,250 + 3 = 1,253. The answer is $1,253.") == "1253"
    assert extract_answer("It costs 4.0 dollars") == "4"
    assert extract_answer("no numbers") is None


def test_toy_formats_have_the_promised_shapes():
    rng = random.Random(0)
    outs = {f: toy_item(random.Random(1), f, 4) for f in FORMATS}
    q, a, ans, run = outs["chain of thought"]
    assert q.startswith("Q") and q.endswith("=") and len(q) == 6
    assert a == "".join(map(str, run[:-1])) + f">{ans};" and run[-1] == ans
    assert len(outs["variable compute (dots)"][1]) == len(a)                     # same length, no content
    assert outs["reasoning after answer"][1].startswith(str(ans))
    assert outs["standard"][1] == f"{ans};"
    assert sum(int(c) for c in q[1:-1]) % 10 == ans
    assert toy_answer("chain of thought", a) == str(ans) and chain_is_correct(a, run)
    assert toy_answer("reasoning after answer", outs["reasoning after answer"][1]) == str(ans)


def test_toy_training_runs_and_learns_something():
    m, loss = train_toy("chain of thought", steps=60, d=32, batch=32)
    acc, _, samples = evaluate_toy(m, "chain of thought", 2, n=20)
    assert loss < 2.0 and samples and 0 <= acc <= 1


def test_paper_numbers():
    s, c = TABLE_2_GSM8K["PaLM 540B"]
    assert (s, c) == (17.9, 56.9) and c > 3 * s
    assert all(c > s for s, c in TABLE_2_GSM8K.values())


def test_gsm8k_exemplars_are_consistent():
    from cot import GSM8K_EXEMPLARS
    assert len(GSM8K_EXEMPLARS) == 8
    for e in GSM8K_EXEMPLARS:
        assert extract_answer(e["chain"]) == e["answer"] or e["answer"] in e["chain"]
