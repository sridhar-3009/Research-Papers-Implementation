import math

import numpy as np

from webgpt import (Browser, LocalWeb, answer_features, best_of_n, bradley_terry_loss, elo_preference,
                    format_answer, kl_penalised_return, make_web, policy_episode, policy_logp, sigmoid,
                    train_reward_model, true_quality)

PAGES = [{"title": "Tides", "domain": "encyclopedia.org",
          "text": "Tides are the rise and fall of sea levels. They are caused by the Moon. The Sun also contributes."},
         {"title": "Moon facts", "domain": "someblog.net", "text": "The Moon is bright. It orbits Earth."}]


def test_browser_commands():
    web = LocalWeb(PAGES)
    b = Browser(web, "What causes tides?", max_actions=10)
    b.step("Search tides moon")
    assert b.results[0] == 0 and "[0] Tides (encyclopedia.org)" in b.view()
    b.step("Clicked on link 0")
    assert b.view().startswith("Tides are the rise")
    b.step("Find in page: Sun")
    assert b.view().startswith("The Sun also contributes.")
    b.step("Quote: They are caused by the Moon.")
    assert b.quotes and b.quotes[0]["domain"] == "encyclopedia.org"
    b.step("Quote: this text is not on the page")
    assert len(b.quotes) == 1
    b.step("Back")
    assert b.page is None
    b.step("dance")                                                                 # invalid, but counts
    assert b.actions == 7
    b.step("End: Answer")
    assert b.done and "Episode is over" in b.step("Top")
    assert "[1] Tides (encyclopedia.org)" in format_answer("Tides come from the Moon [1].", b.quotes)


def test_reward_model_math():
    assert abs(elo_preference(1.0) - 0.731) < 1e-3
    assert abs(bradley_terry_loss(0.0, 0.0, 0.5) - math.log(2)) < 1e-9
    assert bradley_terry_loss(2.0, 0.0, 1.0) < bradley_terry_loss(0.0, 2.0, 1.0)
    rng = np.random.default_rng(0)
    true_w = np.array([0.0, 1.5, -1.0])
    A, B = rng.normal(size=(4000, 3)), rng.normal(size=(4000, 3))
    y = (rng.random(4000) < sigmoid((A - B) @ true_w)).astype(float)
    w = train_reward_model(A, B, y, steps=3000)
    assert np.allclose(w[1:], true_w[1:], atol=0.2)
    assert best_of_n([1, 5, 3], lambda x: -abs(x - 4)) == 5 or best_of_n([1, 5, 3], lambda x: -abs(x - 4)) == 3


def test_kl_penalty():
    assert kl_penalised_return(2.0, [-1.0, -1.0], [-1.0, -1.0], beta=0.5) == 2.0     # same policy: no penalty
    assert kl_penalised_return(2.0, [-0.1], [-2.1], beta=0.5) == 1.0                  # 2 nats more likely: -1


def test_toy_world_policy_and_quality():
    web, truth = make_web()
    rng = np.random.default_rng(0)
    theta = {"precise": 0.999, "top": 0.999, "find": 0.999, "fill": 1e-6}
    text, quotes, logp, b = policy_episode(web, "topic3", theta, rng)
    assert truth["topic3"] in text and quotes and quotes[0]["domain"] == "encyclopedia.org"
    assert true_quality(text, quotes, "topic3", truth) == 2.5
    f = answer_features(text, quotes, "topic3")
    assert list(f[1:]) == [1.0, 1.0, 1.0, 0.0]
    assert abs(policy_logp(b.choices, b.n_fill, theta) - logp) < 1e-9
    wrong = text.replace(truth["topic3"], "valueX")
    assert true_quality(wrong, quotes, "topic3", truth) < 0
