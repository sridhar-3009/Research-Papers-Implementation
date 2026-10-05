# The code, explained simply

How the code in this folder turns two arguments of *The Malicious Use of AI* (Brundage et al. 2018) into defender-side experiments.
Read [EXPLAINED.md](EXPLAINED.md) first. Everything here is abstract or uses a toy digit classifier; there is no attack tooling.

---

## 1. The files

| File | What it is |
|---|---|
| `security.py` | the abstract cost-benefit model (targets, attempt choice, harm, capable actors, the defence needed), a softmax classifier on digits, FGSM, adversarial training, label-flip poisoning, loss-based sanitisation, and the red-team report |
| `experiments.py` | cost-model sensitivity, robustness curves, poisoning rates, a trusted-set audit (E1–E4) |
| `demo.py` | the cost-model table and the red-team report (~1.3 seconds) |
| `test_security.py` | 3 quick tests (~1 second) |

**Run it** (from `16-Extras-optional/102-Brundage-et-al-2018-Malicious-Use-of-AI`; needs scikit-learn):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `security.py`

### The cost-benefit model
| Name | What it does |
|---|---|
| `attack_economics(tailoring_cost, …, defence_factor)` | lognormal target values; per target, the better of generic (0.002·v − 0.01) and tailored (0.05·v − c) if positive; both success rates scaled by `defence_factor`; returns the shares attacked and tailored, the expected harm and the cost |
| `capable_actors(cost, budgets)` | the share of budgets ≥ the cost of one tailored attempt |
| `defence_needed(new_cost, old_cost)` | bisection on `defence_factor` until the harm at the new cost equals the harm at the old cost |

### The classifier and its vulnerabilities
| Name | What it does |
|---|---|
| `load_digits`, `softmax`, `train_softmax`, `accuracy` | multinomial logistic regression by gradient descent on scikit-learn digits; with `adv_eps` > 0, each step trains on clean + FGSM examples |
| `fgsm(W, b, X, y, eps)` | input gradient (P − Y)Wᵀ, signed, scaled by ε, clipped to [0, 1] |
| `poison_labels(y, frac, rng, targeted)` | random flips (always to a different class), or relabel 7 → 1 |
| `sanitize(X, y, drop_frac)` | train, drop the highest-loss fraction, retrain |
| `red_team_report` | clean and FGSM accuracy for a standard vs an adversarially trained model (ε = 0.1); random flips with and without sanitisation; the targeted relabelling with and without sanitisation |

`REPORTED` holds the report's framework (authors, three changes, three domains, AI properties, four recommendations, four research areas), checked against the PDF.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | tailoring cost × tailored success probability |
| `e2` | FGSM accuracy curves for standard and three adversarial-training strengths |
| `e3` | random flips 0–50%, with and without sanitisation |
| `e4` | per-class accuracy on 100 trusted test images and training-label frequencies after the targeted attack |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_cheaper_tailoring_expands_threats_and_defence_restores` | cheaper tailoring raises tailored attempts > 10× and harm > 5×; the computed defence factor restores the old harm within 2%; capable-actor counting |
| `test_fgsm_increases_loss_and_respects_bounds` | perturbations stay within ε and [0, 1], and cut accuracy by > 10 points |
| `test_poisoning` | targeted relabelling changes only 7s; random flips always change the label; a targeted-poisoned model calls > 90% of test 7s "1" |

---

## 5. Try it yourself

1. Add a defence that raises the attacker's cost instead of lowering success (e.g. a per-attempt verification step). Which is more effective in the model?
2. Replace FGSM with a multi-step (projected gradient) perturbation and compare the robustness of the adversarially trained model.
3. Poison only 20% of the 7s. Does the per-class audit still catch it with 100 trusted images?
4. Train an MLP instead of a linear model. Is random label noise still harmless?
5. Write a responsible-disclosure note for the targeted-poisoning finding: what would you tell the model owner, and what fix would you suggest?
