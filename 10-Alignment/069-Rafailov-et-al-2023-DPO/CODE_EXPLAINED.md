# The code, explained simply

How the code in this folder implements DPO (Rafailov et al. 2023).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `dpo.py` | DPO loss (the paper's Appendix B code) and gradient weight; unlikelihood loss; closed-form optimal policy; exact bandit trainer; controlled-sentiment toy (vocabulary, generator, true reward, exact optimal frontier); tiny GRU LM; SFT, preference sampling, DPO / Preferred-FT / Unlikelihood, Bradley–Terry reward model, RLHF policy gradient, Best-of-N, evaluation |
| `experiments.py` | frontier sweep, data size, label noise, ablations, real DPO on Anthropic HH |
| `demo.py` | loss example, exact bandit, sentiment frontier table, implicit-reward accuracy (~10 seconds) |
| `test_dpo.py` | 3 quick tests (~1 second) |

**Run it** (from `10-Alignment/069-Rafailov-et-al-2023-DPO`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~10 seconds
python3 experiments.py --quick
```

---

## 2. `dpo.py`

### The loss
| Function | What it does |
|---|---|
| `dpo_loss(pi_w, pi_l, ref_w, ref_l, beta)` | −log σ(β[(π_w − π_l) − (ref_w − ref_l)]) on summed log-probs; also returns the implicit rewards β(log π − log π_ref) |
| `dpo_grad_weight(r_w, r_l)` | σ(r_l − r_w), the per-pair weight in the gradient |
| `unlikelihood_loss` | −(log π(y_w) − α log π(y_l)): no σ weight and no reference |
| `optimal_policy(ref, r, beta)` | Eq. 4 |
| `kl` | the KL divergence |

### The bandit
`bandit_expected_dpo(ref, r, beta, method)` minimises the **exact expected** loss over all pairs (i, j) ~ π_ref × π_ref, with Bradley–Terry label probabilities σ(r_i − r_j). It runs Adam on the policy's logits. `method` is DPO, unlikelihood or Preferred-FT.

### The sentiment toy
- **Vocabulary:** 0 = BOS, 1–3 positive, 4–6 negative, 7–15 neutral (`SENT` holds +1 / −1 / 0).
- **`generator_probs(q)`:** the "real reviews" distribution. Words are i.i.d.; the prompt's polarity q tilts positive vs negative mass.
- **`true_reward`:** the number of positive words minus the number of negative words.
- **`optimal_frontier(beta)`:** the exact (KL, reward) of π* = generator·e^(s/β)/Z per word, times 6 words, averaged over all 225 prompts.
- **`optimal_reward_at_kl`:** inverts the frontier by bisection on β.
- **`LM`:** embedding → GRU → logits.
  - `seq_logp` sums the completion's log-probs.
  - `sample` generates step by step with the GRU state.
- **`train_sft`:** maximum likelihood on generator samples. This model is π_ref.
- **`make_preferences`:** two SFT samples per prompt; the winner is drawn with probability σ(r₁ − r₂).
- **`train_offline(method)`:** DPO / Unlikelihood / Preferred-FT on the fixed pairs. The reference log-probs are pre-computed once.
- **`train_reward_model`:** a Bradley–Terry linear model on word counts (and counts × prompt tone).
- **`train_rlhf`:** REINFORCE with a mean baseline on r_RM − β(log π − log π_ref), the same KL-penalised reward PPO uses. It is simpler than PPO (no clipping or value net).
- **`evaluate`:** mean true reward and **sequence-level KL** (the sum of exact per-step KLs along sampled sequences, as in the paper's footnote).
- **`best_of_n`:** the highest-scoring of N SFT samples under the RM. Its KL is the standard log N − (N−1)/N.
- **`implicit_reward`, `pair_accuracy`:** "the LM is secretly a reward model".

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | frontier: DPO / RLHF over β ∈ {0.05, 0.1, 0.3, 1, 3}, Unlikelihood α ∈ {0.05, 0.1, 0.5, 1}, Preferred-FT, Best-of-{4, 16, 64}; 3 seeds; efficiency = reward / best reward at that KL |
| `e2` | 250 … 16,000 pairs, DPO vs RLHF |
| `e3` | 0–40% flipped labels |
| `e4` | DPO vs unweighted gradient vs no reference model |
| `e5` | real DPO on Anthropic HH (Qwen2.5-0.5B-Instruct, β = 0.1, RMSprop 1e-6, 150-step warm-up, batch 64 via accumulation); held-out implicit-reward accuracy |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_dpo_loss_matches_formula_and_gradient` | the loss value −log σ(1); autograd gradient = ∓β·σ(r_l − r_w) |
| `test_bandit_dpo_recovers_closed_form_optimum` | DPO equals π_ref·e^(r/β)/Z; log π*/π_ref recovers r up to a constant (Eq. 5) |
| `test_frontier_and_lm_shapes` | smaller β gives more reward and more KL; LM log-prob shapes; reward counting |

---

## 5. Try it yourself

1. Give DPO 8000 pairs instead of 2000 (`make_preferences(..., 8000)`). Does it close the gap to RLHF?
2. Train DPO for 2000 steps instead of 400. Does it overfit, with KL rising past its target?
3. Make the reward model the wrong family (e.g. only count positive words). Does RLHF lose its edge?
4. Set β = 0.05 in the bandit. How close to the argmax does DPO get, and what happens to KL?
5. Run `experiments.py --only e4` and compare the unweighted update's KL with DPO's.
