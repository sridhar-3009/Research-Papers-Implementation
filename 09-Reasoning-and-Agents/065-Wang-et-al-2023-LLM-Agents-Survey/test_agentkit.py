import random

from agentkit import (FRAMEWORK, HybridMemory, SimLLM, UnifiedMemory, analytic, dataset_aligned_profiles, evaluate,
                      generated_profiles, handcrafted_profile, key_door_episode, reflect_memories, reflexion,
                      self_consistency, single_path, tree_of_thoughts, with_environment_feedback)


def test_profiles():
    assert handcrafted_profile("Ada", "coder", ["careful"]) == "You are Ada, a coder. You are careful."
    ps = generated_profiles(3, random.Random(0))
    assert len(ps) == 3 and all(p.startswith("You are agent") for p in ps)
    assert dataset_aligned_profiles([{"age": 34, "occupation": "nurse", "state": "Ohio"}]) == ["You are a 34-year-old nurse from Ohio."]


def test_memory_structures_and_operations():
    u = UnifiedMemory(window=3)
    for t in ["a1", "a2", "a3", "a4"]:
        u.write(t)
    assert u.records == ["a2", "a3", "a4"]                                         # overflow forgets the oldest
    h = HybridMemory(window=2, capacity=4)
    h.write("note says key3 opens door", importance=8)
    for t in ["saw x", "saw y", "saw x", "saw z", "saw w"]:
        h.write(t)
    texts = [r["text"] for r in h.long]
    assert texts.count("saw x") == 1                                               # duplicates are merged
    assert len(h.long) == 4 and "note says key3 opens door" in texts               # eviction keeps the important one
    assert "saw y" not in texts                                                    # ... and drops the least important, oldest
    assert "key3" in " ".join(h.read("key door"))
    assert reflect_memories(["Bob ran", "Bob ate", "Bob slept", "Amy ran"]) == ["Bob occurs 3 times"]


def test_planning_strategies_match_closed_forms():
    p, L = 0.8, 6
    s, calls = evaluate(single_path, p, L, n=4000)
    assert abs(s - analytic("single_path", p, L)) < 0.03 and calls < L + 0.01
    e, _ = evaluate(with_environment_feedback, p, L, n=4000, retries=2)
    assert abs(e - analytic("environment_feedback", p, L, retries=2)) < 0.02
    perfect = SimLLM(p=1.0, q=1.0)
    steps = ["a", "b", "c"]
    assert single_path(perfect, steps) and self_consistency(perfect, steps) and tree_of_thoughts(perfect, steps)
    assert reflexion(perfect, steps) and with_environment_feedback(perfect, steps)
    hopeless = SimLLM(p=0.0, q=1.0)
    assert not single_path(hopeless, steps) and not with_environment_feedback(hopeless, steps)


def test_reflexion_learns_across_trials():
    llm = SimLLM(p=0.5, seed=1)
    wins = sum(reflexion(llm, ["a", "b"], trials=2) for _ in range(4000)) / 4000
    # trial 1: both right 0.25; fail at a (0.5) -> trial 2 needs b (0.5); fail at b (0.25) -> trial 2 needs a (0.5)
    assert abs(wins - (0.25 + 0.5 * 0.5 + 0.25 * 0.5)) < 0.03


def test_key_door_memory_task():
    rng = random.Random(0)
    assert sum(key_door_episode(UnifiedMemory(8), 20, rng) for _ in range(50)) == 0
    assert sum(key_door_episode(HybridMemory(8), 20, rng) for _ in range(50)) == 50
    assert len(FRAMEWORK) == 9
