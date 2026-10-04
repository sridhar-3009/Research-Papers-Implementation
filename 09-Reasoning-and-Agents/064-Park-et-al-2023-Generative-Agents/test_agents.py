import numpy as np

from agents import (DECAY, MemoryStream, embed, minmax, rate_importance, reflect, reflection_depth, run_town,
                    salient_questions)


def test_importance_and_embedding_standins():
    assert rate_importance("brushing teeth") == 1
    assert rate_importance("asking your crush out on a date") > rate_importance("cleaning up the room")
    assert 1 <= rate_importance("a breakup") <= 10
    a, b, c = embed("Klaus is writing a research paper"), embed("Klaus writes research"), embed("eating breakfast")
    assert abs(np.linalg.norm(a) - 1) < 1e-9 and a @ b > a @ c


def test_retrieval_score_is_sum_of_minmax_scaled_terms():
    ms = MemoryStream()
    ms.add("ate breakfast in the kitchen", t=0, importance=1)
    ms.add("asked Maria about the research project", t=10, importance=6)
    ms.add("Maria is planning a party", t=20, importance=4)
    total, (rec, imp, rel) = ms.scores("research project with Maria", t=20)
    assert np.allclose(rec, minmax([DECAY ** 20, DECAY ** 10, 1.0]))
    assert np.allclose(imp, [0, 1, 0.6])
    assert np.allclose(total, rec + imp + rel)
    top = ms.retrieve("research project with Maria", t=30, k=1)
    assert top[0].text.startswith("asked Maria") and top[0].last_access == 30   # retrieval refreshes recency
    w = MemoryStream(weights=(1, 0, 0)); w.records = ms.records
    assert w.retrieve("anything", t=30, k=1)[0].last_access == 30


def test_reflection_trigger_and_tree():
    ms = MemoryStream()
    for i in range(30):
        ms.add(f"Klaus is reading about gentrification research {i}", t=i, importance=6)
    assert ms.should_reflect()                                                     # 30 x 6 = 180 > 150
    qs = salient_questions(ms.records)
    assert len(qs) == 3 and any("klaus" in q or "gentrification" in q or "research" in q for q in qs)
    refl = reflect("Klaus", ms, t=31)
    assert len(refl) == 3 and all(r.kind == "reflection" and r.evidence for r in refl)
    assert "because of" in refl[0].text and not ms.should_reflect()
    assert reflection_depth(refl[0]) == 1 and reflection_depth(ms.records[0]) == 0


def test_town_runs_and_measures():
    r = run_town(seed=0, hours=30)
    assert r["party"] >= 1 and r["mayor"] >= 1 and 0 <= r["attended"] <= r["party"]
    d0, d1 = r["density"]
    assert 0 < d0 <= d1 <= 1
