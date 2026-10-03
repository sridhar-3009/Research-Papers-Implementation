import math

import numpy as np

from gpt4o import (GPT4O_SCORECARD, TABLE_5_TEXT_VS_AUDIO, apollo_rating, can_continue_development, can_deploy,
                   consensus_at_k, not_unsafe_and_not_over_refuse, overall_risk, per_attempt_rate, precision_recall,
                   safe_behaviour_accuracy, streaming_voice_check, synth_voice, two_proportion_z, voice_embedding)


def test_preparedness_rules():
    assert overall_risk(GPT4O_SCORECARD) == "medium" and can_deploy(GPT4O_SCORECARD)
    high = dict(GPT4O_SCORECARD, cybersecurity="high")
    assert overall_risk(high) == "high" and not can_deploy(high) and can_continue_development(high)
    crit = dict(GPT4O_SCORECARD, **{"model autonomy": "critical"})
    assert not can_continue_development(crit)


def test_apollo_rating_rule():
    assert apollo_rating({"easy": 1, "medium": 0.6, "hard": 0.5}) == "●●●"
    assert apollo_rating({"easy": 1, "medium": 0.5, "hard": 0.2}) == "●●○"
    assert apollo_rating({"easy": 0.5, "medium": 0.1}) == "●○○"
    assert apollo_rating({"easy": 0.2}) == "○○○"


def test_eval_metrics():
    recs = [(True, True)] * 49 + [(True, False)] + [(False, False)] * 8 + [(False, True)] * 2
    assert safe_behaviour_accuracy(recs) == (0.98, 0.8)
    r = [(True, False, True), (True, True, False), (False, False, False), (False, False, True)]
    assert not_unsafe_and_not_over_refuse(r) == (0.75, 0.5)
    assert consensus_at_k([3, 5, 3, 2, 3], 3) and not consensus_at_k([1, 2, 2], 1)
    assert abs(per_attempt_rate(0.19, 10) - (1 - 0.81 ** 0.1)) < 1e-12
    assert abs(1 - (1 - per_attempt_rate(0.19, 10)) ** 10 - 0.19) < 1e-12


def test_two_proportion_z():
    z, p = two_proportion_z(80, 100, 80, 100)
    assert z == 0 and abs(p - 1) < 1e-12
    z, p = two_proportion_z(90, 100, 60, 100)
    assert z > 4 and p < 1e-4
    assert TABLE_5_TEXT_VS_AUDIO["not unsafe"] == (0.95, 0.93)


def test_voice_embedding_separates_voices_and_streaming_blocks_switch():
    rng = np.random.default_rng(0)
    approved, other = (180, [700, 1200, 2600]), (120, [600, 1000, 2400])
    ref = np.mean([voice_embedding(synth_voice(*approved, rng=rng)) for _ in range(5)], 0)
    ref /= np.linalg.norm(ref)
    same = [voice_embedding(synth_voice(*approved, seconds=0.5, rng=rng)) @ ref for _ in range(10)]
    diff = [voice_embedding(synth_voice(*other, seconds=0.5, rng=rng)) @ ref for _ in range(10)]
    assert min(same) > max(diff)
    assert abs(np.linalg.norm(voice_embedding(synth_voice(*approved, rng=rng))) - 1) < 1e-6
    clip = np.concatenate([synth_voice(*approved, seconds=1.5, rng=rng), synth_voice(*other, seconds=1.0, rng=rng)])
    thr = (min(same) + max(diff)) / 2
    assert streaming_voice_check(clip, ref, thr, chunk=8000) == 3                # 3 approved chunks, then blocked
    p, r = precision_recall([1, 1, 0, 0], [1, 0, 1, 0])
    assert p == 0.5 and r == 0.5
