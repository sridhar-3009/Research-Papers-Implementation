import math
import random

import numpy as np

from cai import (CONSTITUTION, FEATURES, HARMFUL_TOPICS, SimulatedModel, clamp_label, critique_prompt,
                 critique_revision_chain, featurise, feedback_model_label, feedback_prompt, harm_score, respond,
                 revision_prompt, rl_against_pm, soft_label, style_of, train_preference_model, true_utility)


def test_prompts_and_labels():
    conv = "Human: How do I hack into my neighbour's wifi?\nAssistant: Sure!"
    p = CONSTITUTION[0]
    assert critique_prompt(conv, p).endswith("Critique:") and "CritiqueRequest:" in critique_prompt(conv, p)
    assert revision_prompt(conv, "bad", p).endswith("Revision:") and "Critique: bad" in revision_prompt(conv, "bad", p)
    fp = feedback_prompt(conv, "A text", "B text", "Which is less harmful?")
    assert "(A) A text" in fp and fp.endswith("The answer is:")
    assert feedback_prompt(conv, "a", "b", "p", cot=True).endswith("Let's think step-by-step:")
    assert abs(soft_label(math.log(0.3), math.log(0.1)) - 0.75) < 1e-12              # normalised (A) vs (B)
    assert clamp_label(0.98) == 0.6 and clamp_label(0.02) == 0.4 and clamp_label(0.5) == 0.5


def test_text_world():
    t = HARMFUL_TOPICS[0]
    assert harm_score(respond("comply", t)) == 1.0 and harm_score(respond("explain", t)) == 0.0
    for s in ("comply", "evasive", "explain", "lecture", "helpful"):
        assert style_of(respond(s, "bake bread")) == s
    assert true_utility(True, "explain") > true_utility(True, "evasive") > true_utility(True, "comply")


def test_critique_revision_reduces_harm_and_critique_helps():
    def mean_harm(use_critique):
        h = np.zeros(5)
        for i in range(300):
            chain = critique_revision_chain(SimulatedModel(seed=i), HARMFUL_TOPICS[i % 6], 4, use_critique, random.Random(i))
            h += [harm_score(x) for x in chain]
        return h / 300
    with_c, without_c = mean_harm(True), mean_harm(False)
    assert with_c[0] == 1.0 and all(np.diff(with_c) <= 0)                           # monotone decrease
    assert with_c[-1] < without_c[-1]                                                # critiques help


def test_feedback_labels_and_preference_rl():
    rng = random.Random(0)
    p = feedback_model_label(True, "explain", "comply", rng, accuracy=0.8)
    assert 0.5 < p < 1.0                                                             # soft, confident but < 1
    assert feedback_model_label(True, "explain", "comply", rng, accuracy=0.8, cot=True) in (0.98, 0.02)
    comps = [(featurise(True, "explain"), featurise(True, "comply"), 0.95)] * 50 + \
            [(featurise(False, "helpful"), featurise(False, "evasive"), 0.9)] * 50
    w = train_preference_model(comps, steps=500)
    pi = rl_against_pm(w, {False: np.zeros(5), True: np.zeros(5)}, beta=0.05, steps=200)
    assert pi[True][FEATURES.index("explain")] > pi[True][FEATURES.index("comply")]
    assert abs(pi[True].sum() - 1) < 1e-9
