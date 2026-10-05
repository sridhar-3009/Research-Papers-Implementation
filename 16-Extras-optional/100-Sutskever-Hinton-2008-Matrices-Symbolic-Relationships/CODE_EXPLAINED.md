# The code, explained simply

How the code in this folder implements Matrix Relational Embedding (Sutskever & Hinton 2008).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `mre.py` | the MRE model (all entities as N × N matrices), the Eq. 1 + Eq. 2 + weight-decay cost with an exact vectorised gradient, conjugate-gradient training (with optional frozen entities), the arithmetic and family-tree tasks, and the higher-order and incremental experiments |
| `experiments.py` | Tables 1–5 with 5 runs each, plus a matrix-size sweep (E1–E5) |
| `demo.py` | gradient check, Tables 1–2 analogues, higher-order arithmetic (Eq. 2 vs discriminative vs incremental), higher-order family (~27 seconds) |
| `test_mre.py` | 4 quick tests (~0.7 seconds) |

**Run it** (from `16-Extras-optional/100-Sutskever-Hinton-2008-Matrices-Symbolic-Relationships`; needs scipy):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `mre.py`

### The model
| Name | What it does |
|---|---|
| `MRE(n_objects, n_relations, N)` | one array E of shape (entities, N, N): objects first, then relations (`rel(r)` gives a relation's entity index); unit-Gaussian initialisation; weight decay 0.01 |
| `cost_grad(E, basic, higher, free)` | **basic** rows (relation, A, B-object): M = RA for all rows at once; logits −‖M − C‖² over all objects; log-sum-exp cost; gradients dM = 2(Σ p_c C − B) and dC = 2(p_c − [c = B])(M − C), scattered back to R and A with `np.add.at`. **higher** rows (R̃, A, B as any entities): squared error ‖R̃A − B‖² and its gradient. Then weight decay (only on the `free` entities when some are frozen) |
| `fit(basic, higher, maxiter, free)` | `scipy.optimize.minimize(method="CG")` on the flattened matrices, or only on the `free` ones |
| `answer_ranking`, `correct` | rank objects by distance to RA; correct if B is among the k closest, where k = number of right answers |

### Tasks
| Name | What it does |
|---|---|
| `arithmetic_task(base, with_mult)` | +k and ×k relations mod 12 (288 cases) |
| `evaluate(model, cases, train_cases)` | counts errors, grouping all known answers of each (relation, A) query |
| `run_arithmetic(n_test)`, `run_family(n_test)` | random held-out split, training, then test and training errors |
| `ENGLISH`, `ITALIAN_NAMES`, `family_facts`, `family_task` | the tree (couples, children, genders) and a rename for the isomorphic Italian copy; relations are derived (aunts and uncles include spouses of parents' siblings; nephews and nieces include children of the spouse's siblings); 112 cases |
| `higher_order_arithmetic(k, incremental, discriminative_higher)` | basic +0…+11 without +k, plus 36 plus / minus / inverse facts; Eq. 2 (default), a discriminative softmax over relation matrices (`_DiscHigher`, the ablation), or incremental (learn the rest, then only +k with everything frozen) |
| `higher_order_family(relation, incremental)` | the same with the 12 higher_oppsex facts |

`REPORTED` holds Tables 1–5 and the training set-up, checked against the PDF.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | arithmetic with 30 / 60 / 90 held out × N ∈ {2…6} × 5 runs |
| `e2` | family trees with 10 / 20 / 30 held out × 5 runs |
| `e3` | higher-order +1 / +4 / +6 / +10: Eq. 2 vs discriminative, 5 runs |
| `e4` | higher-order family for 4 relations, 5 runs |
| `e5` | incremental versions of E3 and E4 |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_gradient_matches_finite_differences` | the full cost's gradient is exact |
| `test_task_definitions` | 288 arithmetic facts (11 + 3 ≡ 2, 7 × 5 ≡ 11 mod 12); 24 people and 112 family cases; facts like Colin's father James, his uncle Arthur, and Charlotte's aunt by marriage Margaret |
| `test_learns_arithmetic_and_generalises` | zero training errors and at most 2 of 30 test errors |
| `test_incremental_learning_keeps_frozen_matrices` | with `free=[…]`, only that matrix changes |

---

## 5. Try it yourself

1. Implement LRE (vector objects) on the same tasks and compare generalisation and parameter counts.
2. Add "×k" relations to the higher-order task with a "times" higher-order relation.
3. Inspect the learned +k matrices: are they close to powers of the +1 matrix?
4. Use 2 × 2 matrices on mod-4 arithmetic and plot the learned number matrices (circle structure?).
5. Teach a brand-new relation such as "has_grandfather" from a definition: (has_father · has_father) — what would the higher-order fact look like?
