import numpy as np

import mre as M


def test_gradient_matches_finite_differences():
    m = M.MRE(4, 3, N=2, seed=1)
    basic = np.array([(m.rel(0), 1, 2), (m.rel(2), 3, 0)])
    higher = np.array([(m.rel(1), m.rel(0), m.rel(2)), (m.rel(1), 2, m.rel(0))])
    c, g = m.cost_grad(m.E, basic, higher)
    num = np.zeros_like(m.E)
    for idx in np.ndindex(m.E.shape):
        Ep, Em = m.E.copy(), m.E.copy()
        Ep[idx] += 1e-6; Em[idx] -= 1e-6
        num[idx] = (m.cost_grad(Ep, basic, higher)[0] - m.cost_grad(Em, basic, higher)[0]) / 2e-6
    assert np.abs(num - g).max() < 1e-5


def test_task_definitions():
    names, cases = M.arithmetic_task()
    assert len(cases) == 288 and (3, 11, 2) in cases and (12 + 5, 7, 11) in cases   # 11+3=2, 7*5=35=11 (mod 12)
    people, fam = M.family_task()
    assert len(people) == 24 and len(fam) == 112
    facts = M.family_facts(M.ENGLISH)
    assert ("Colin", "has_father", "James") in facts and ("Colin", "has_uncle", "Arthur") in facts
    assert ("Charlotte", "has_aunt", "Margaret") in facts                            # aunt by marriage


def test_learns_arithmetic_and_generalises():
    test_err, train_err = M.run_arithmetic(30, seed=0)
    assert train_err == 0 and test_err <= 2


def test_incremental_learning_keeps_frozen_matrices():
    m = M.MRE(3, 2, N=2, seed=0)
    before = m.E.copy()
    m.fit(np.zeros((0, 3), int), [(m.rel(0), 0, m.rel(1))], maxiter=200, free=[m.rel(1)])
    changed = np.abs(m.E - before).sum((1, 2)) > 0
    assert list(np.where(changed)[0]) == [m.rel(1)]
