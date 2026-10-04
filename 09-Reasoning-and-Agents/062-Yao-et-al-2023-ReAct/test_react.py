from react import (TABLE_1, ToyWorld, WikiEnv, ablate, cot_then_react, majority, react_loop, react_then_cot, run_toy,
                   train_toy)

PAGES = {"Velmora": "Velmora is a 1998 film. It was directed by Ana Kord. It won two awards.",
         "Ana Kord": "Ana Kord is a film director. She was born in Brisk. Brisk is a coastal town."}


def test_wiki_env_actions():
    env = WikiEnv(PAGES)
    assert env.search("Velmora").startswith("Velmora is a 1998 film.")
    assert "Similar" in env.search("Velmorra") and "Velmora" in env.search("Velmorra")
    env.search("Ana Kord")
    assert env.lookup("Brisk").endswith("She was born in Brisk.")
    assert env.lookup("Brisk").endswith("Brisk is a coastal town.")
    assert env.lookup("Brisk") == "No more results."
    obs, done = env.step("finish[Brisk]")
    assert done and env.answer == "Brisk"
    assert env.step("dance[now]")[0].startswith("Invalid")


def test_react_loop_with_a_scripted_model():
    script = iter([" I need the director of Velmora.\nAction 1: search[Velmora]",
                   " Ana Kord directed it. Find where she was born.\nAction 2: search[Ana Kord]",
                   " She was born in Brisk.\nAction 3: finish[Brisk]"])
    answer, transcript = react_loop(lambda prompt, stop: next(script), "Where was the director of Velmora born?",
                                    WikiEnv(PAGES))
    assert answer == "Brisk"
    assert "Observation 1: Velmora is a 1998 film." in transcript and "Observation 2: Ana Kord is" in transcript
    assert transcript.count("Thought") == 3


def test_ablations_and_backoff():
    traj = [("thought", "t"), ("action", "search[x]"), ("observation", "o"), ("answer", "y")]
    assert ablate(traj, "standard") == [("answer", "y")]
    assert [k for k, _ in ablate(traj, "cot")] == ["thought", "answer"]
    assert [k for k, _ in ablate(traj, "act")] == ["action", "observation", "answer"]
    assert majority(["a", "b", "a", None]) == ("a", 2)
    assert react_then_cot(None, ["x", "x", "y"]) == ("x", "cot-sc") and react_then_cot("z", ["x"]) == ("z", "react")
    assert cot_then_react(["a", "b", "c", "a"], lambda: "r") == ("a", "cot-sc")    # 2 of 4 >= n/2
    assert cot_then_react(["a", "b", "c"], lambda: "r") == ("r", "react")
    assert TABLE_1["ReAct -> CoT-SC"][0] > TABLE_1["CoT-SC (21 samples)"][0]


def test_toy_world_trajectories_and_run():
    w = ToyWorld()
    f = w.films[0]
    t = w.trajectory(f, "react")
    assert t[:4] == ["Q", f, "T1", "S"] and t[-2] == w.answer(f)
    assert w.trajectory(f, "cot")[-2] == w.answer(f) and "S" not in w.trajectory(f, "cot")
    w.outage = True
    assert w.page(f) == []
    w.outage = False
    m, loss = train_toy(w, "react", steps=5, batch=8)
    ans, seq = run_toy(w, m, f, "react")
    assert seq[:2] == ["Q", f]
