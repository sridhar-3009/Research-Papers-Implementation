# The code, explained simply

How the code in this folder implements WebGPT (Nakano et al. 2021).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `webgpt.py` | a local web with TF-IDF and authority search, the Table 1 text browser with its state summary, reference formatting, Bradley–Terry / Elo reward-model maths and a linear RM trainer, best-of-n, the KL-penalised return, a toy web, a stochastic browsing policy, a simulated labeler, REINFORCE with KL, the paper's numbers |
| `experiments.py` | a real RM on `openai/webgpt_comparisons`, best-of-n with an open model, toy RM-data / best-of-n / β sweeps (heavy, not run here) |
| `demo.py` | a browser session, the toy RM, the best-of-n over-optimisation curve, RL with and without KL, the paper's numbers (~8 seconds) |
| `test_webgpt.py` | 4 quick tests (~0.2 seconds) |

**Run it** (from `09-Reasoning-and-Agents/063-Nakano-et-al-2021-WebGPT`):
```
python3 -m pytest -q             # ~0.2 seconds
python3 demo.py                  # ~8 seconds
python3 experiments.py --quick
```

---

## 2. `webgpt.py`

### The browser
- **`LocalWeb(pages)`:** pages are `{title, domain, text, authority?}`. `search(query, k)` ranks by TF-IDF (normalised by page length) × authority.
- **`Browser(web, question, max_actions)`:** `step(command)` understands the Table 1 commands:
  - `Search`, `Clicked on link`, `Find in page:`, `Quote:` (accepted only if the text is on the page; it stores title, domain and extract);
  - `Scrolled down/up n`, `Top`, `Back`, `End: Answer`, `End: Nonsense/Controversial`;
  - anything else is an invalid action that still counts towards the limit.
  - **`summary()`** is the text the model sees.
- **`format_answer(text, quotes)`:** the answer followed by a numbered reference list.

### The reward model and optimisation
| Name | What it is |
|---|---|
| `elo_preference(delta)` | σ(delta); 1 → 73% |
| `bradley_terry_loss(r_a, r_b, label)` | cross-entropy with soft tie labels |
| `train_reward_model(FA, FB, y)` | a linear r = w·features fitted by gradient descent on the BT loss |
| `best_of_n(candidates, score_fn)` | rejection sampling |
| `kl_penalised_return(rm, logp_pi, logp_bc, beta)` | RM − β Σ(log π − log π_BC) |

### The toy
- **`make_web()`:** an encyclopedia page (true fact, authority 1.5) and a blog page (wrong fact) per topic.
- **`policy_episode(web, topic, theta, rng)`:** a stochastic browsing policy that runs real `Browser` commands. Its four parameters are the chance of a precise query, of clicking the top result, of using Find, and the expected number of filler sentences. It returns the answer, quotes, log-probability and browser (with the recorded choices).
- **`policy_logp(choices, n_fill, theta)`:** the log-probability of the same episode under another policy (for KL).
- **`answer_features`:** what the RM can see (cites, supported, encyclopedia domain, filler count).
- **`true_quality`:** the labeler's utility. It is non-linear in filler: good up to 2 sentences, bad beyond.
- **`labeler_compare`:** a noisy comparison with σ((q_a − q_b)/noise) and 10% ties.
- **`rl_finetune(web, truth, theta_bc, reward_fn, beta)`:** REINFORCE with exact score-function gradients (x − p for Bernoulli choices, n − λ for Poisson length), a mean baseline, and the KL-penalised return.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | a real RM: a small LM + scalar head on `openai/webgpt_comparisons`, BT loss with ties; held-out pairwise accuracy; saves `rm.pt` |
| `e2` | best-of-1/4/16 answers from an open instruction model, scored by the E1 reward model |
| `e3` | toy: RM accuracy vs number of comparisons (250 … 16,000) |
| `e4` | toy: the true-quality-vs-n curve for RMs trained on 500 / 3,000 / 16,000 comparisons |
| `e5` | toy: RL with β ∈ {0, 0.03, 0.1, 0.3, 1}; RM score vs true quality |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_browser_commands` | search ranking, click, find, quote (only real text), back, invalid action counting, end, reference formatting |
| `test_reward_model_math` | σ(1) = 73%; tie loss = log 2; the BT trainer recovers true weights; best-of-n |
| `test_kl_penalty` | no penalty for identical policies; 0.5 × 2 nats = 1 |
| `test_toy_world_policy_and_quality` | a near-deterministic good policy cites the encyclopedia and gets utility 2.5; features; log-prob consistency; a wrong value scores below 0 |

---

## 5. Try it yourself

1. Give the RM a non-linear feature (`min(fill, 2)`) as well as `fill`. Does the best-of-n curve stop turning down?
2. Train the RM on only 300 comparisons. Where does the best-of-n peak move?
3. Set the blog pages' authority to 2.0 (the unreliable site ranks first). How do BC, best-of-n and RL change?
4. Run `rl_finetune` with β = 0.03 and 1.0. Plot RM score vs true quality over training.
5. Add an `End: Nonsense` option to the policy for unanswerable questions, and give the labeler a reward for using it correctly.
