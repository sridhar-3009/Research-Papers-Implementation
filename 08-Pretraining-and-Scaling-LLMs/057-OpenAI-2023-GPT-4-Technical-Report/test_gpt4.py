import random

import numpy as np

from gpt4 import (TABLE_2, difficulty_buckets, eligible_problems, expected_calibration_error, fit_capability,
                  fit_power_law_offset, hindsight_neglect_item, is_contaminated, mean_log_pass_rate, predict,
                  rbrm_reward, reliability_table, strip, toy_rbrm_classify)


def test_power_law_with_offset_recovers_and_extrapolates():
    C = np.logspace(-8, -4, 8)                                                  # compute normalised so the target is 1
    L = 0.5 * C ** -0.1 + 1.2
    a, b, c = fit_power_law_offset(C, L, n_grid=2000)
    assert abs(b + 0.1) < 0.01 and abs(c - 1.2) < 0.02
    assert abs(predict((a, b, c), 1.0) - 1.7) < 0.02                            # 10,000x extrapolation


def test_capability_metric_selection_and_fit():
    counts = np.array([[3, 0, 5], [7, 1, 9], [10, 4, 20]])
    assert list(eligible_problems(counts)) == [0, 2]
    assert abs(mean_log_pass_rate([0.5, 0.25]) - (np.log(2) + np.log(4)) / 2) < 1e-12
    C = np.logspace(-6, -3, 5)
    alpha, k = fit_capability(C, 0.2 * C ** -0.3)
    assert abs(alpha - 0.2) < 1e-9 and abs(k - 0.3) < 1e-9
    buckets = difficulty_buckets(np.linspace(0, 1, 75), n_buckets=6, exclude_hardest=15)
    assert sum(len(b) for b in buckets) == 60 and 74 in buckets[0] and 15 in buckets[-1]


def test_ece():
    assert expected_calibration_error([0.9] * 10, [1] * 9 + [0]) < 1e-12        # 90% confident, 90% right
    assert abs(expected_calibration_error([0.9] * 10, [1] * 5 + [0] * 5) - 0.4) < 1e-12
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.25, 1, 20000)
    calibrated = rng.random(20000) < conf
    assert expected_calibration_error(conf, calibrated) < 0.02
    over = np.clip(conf + 0.15, 0, 1)                                            # overconfident
    assert expected_calibration_error(over, calibrated) > 0.1
    assert sum(r[2] for r in reliability_table(conf, calibrated)) == 20000


def test_contamination_substring_check():
    q = ("A 30-year-old man presents with fever, a stiff neck, and photophobia for two days. Which test should be "
         "performed first to confirm the most likely diagnosis?")
    assert strip("Hello, World! 42") == "helloworld42"
    train_hit = ["Some forum post... a 30 year old man presents with FEVER; a stiff-neck & photophobia for two days. "
                 "Which test should be performed first, to confirm the most likely diagnosis? ... answers below"]
    assert is_contaminated(q, train_hit, random.Random(0))                      # formatting differences are ignored
    assert not is_contaminated(q, ["An unrelated training document about cooking pasta."], random.Random(0))
    paraphrase = ["A man aged thirty has had a fever, neck stiffness and light sensitivity for 2 days; what do you test?"]
    assert not is_contaminated(q, paraphrase, random.Random(0))                 # a known false negative


def test_rbrm_rewards():
    assert toy_rbrm_classify("I can't help with that, but I can share safety resources.") == "A"
    assert toy_rbrm_classify("As an AI model developed by OpenAI, I cannot. It is important to note...") == "B"
    assert toy_rbrm_classify("Step 1: [redacted-instructions]") == "C"
    assert rbrm_reward(True, "I can't help with that.") > rbrm_reward(True, "Sure! Here you go.")
    assert rbrm_reward(False, "Sure! Here are some cheaper options.") > rbrm_reward(False, "I cannot help with that.")


def test_hindsight_neglect_labels_follow_expected_value():
    rng = random.Random(0)
    for _ in range(50):
        text, label, ev = hindsight_neglect_item(rng)
        assert (label == "Y") == (ev > 0)
        assert ("wins" in text) == (ev < 0)                                     # the outcome always misleads


def test_reported_numbers():
    assert TABLE_2["MMLU (5-shot)"][0] == 86.4
    beats_sota = [k for k, (g4, _, _, sota) in TABLE_2.items() if g4 > sota]
    assert len(beats_sota) == 6 and "DROP F1 (3-shot)" not in beats_sota       # all except DROP
