# The code, explained simply

How the code in this folder implements InstructGPT's three steps (Ouyang et al. 2022).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `instructgpt.py` | a toy world (web text, instruction tasks, labeler utility), tiny LLaMA-style models, pre-training, the pre-training benchmark, SFT on demonstrations, sampling, K-way rankings, the per-prompt ranking loss (Eq. 1), RM training with normalisation, PPO / PPO-ptx (Eq. 2), evaluation and win rates |
| `experiments.py` | RM strength vs RL gain, per-prompt vs shuffled pairs, γ and β sweeps, SFT data size (heavy, not run here) |
| `demo.py` | base model → SFT → RM → PPO vs PPO-ptx, with the alignment tax (~18 seconds) |
| `test_instructgpt.py` | 3 quick tests (~1 second) |

**Run it** (from `10-Alignment/067-Ouyang-et-al-2022-InstructGPT`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~18 seconds
python3 experiments.py --quick
```

---

## 2. `instructgpt.py`

### The world
| Name | What it is |
|---|---|
| `make_instruction(rng)` | a task from SORT / FIRST2 / LAST2, plus 3–5 letters |
| `correct_response`, `labeler_utility` | the gold answer; 2 × position accuracy + 1 for exact − 0.3 × length error |
| `web_document(rng)` | pre-training text: 80% periodic patterns, 20% instruction + unrelated continuation |
| `benchmark(model)` | next-token accuracy on held-out periodic documents (the "public NLP benchmark") |

### Training
- **`pretrain(steps)`:** LM loss on web documents.
- **`sft(base, steps, n_demos)`:** fine-tune on a fixed set of demonstrations; the loss covers the response only.
- **`respond(model, prompts, temperature)`:** batched generation; task tokens are disallowed in answers.

### Reward model
- **`collect_rankings(policy, n, K_range=(4, 9))`:** K samples per prompt at temperature 1.5 (diversity), sorted by noisy labeler utility.
- **`ranking_loss(rm, prompt, ranked)`:** Eq. 1 for one prompt: K forward passes, then the mean of −log σ(r_i − r_j) over all C(K, 2) ordered pairs.
- **`train_reward_model(sft_model, rankings)`:** RM = the SFT body + a scalar head; batches of 8 prompts; then a bias so demonstrations score 0.
- **`score(rm, prompts, responses)`:** the normalised reward.

### RL and evaluation
- **`ppo(sft_model, rm, beta, gamma_ptx, ...)`:**
  - **Per iteration:** sample responses; reward = −β·KL per token plus the RM score at the end; returns-to-go; advantages against a value net initialised from the RM; clipped PPO updates.
  - **With `gamma_ptx` > 0:** adds γ × the LM loss on fresh web documents to every update (PPO-ptx).
- **`evaluate(model, temperature)`:** utility, exact match, benchmark.
- **`win_rate(a, b)`:** simulated labeler preference.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | 300 … 8000 ranked prompts: RM accuracy and PPO utility vs SFT |
| `e2` | Eq. 1 per-prompt batching vs shuffled independent pairs: validation accuracy per epoch |
| `e3` | γ ∈ {0, 0.1, 0.3, 1, 3}: utility vs benchmark (the alignment tax) |
| `e4` | β ∈ {0, 0.01, 0.05, 0.2}: KL, RM score, utility |
| `e5` | 50 … 2000 SFT demonstrations: utility and benchmark |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_tasks_and_utility` | task answers; utility values; web text has ~20% instruction-like documents whose continuations are not the answer |
| `test_ranking_loss_is_mean_over_all_pairs` | `ranking_loss` equals the explicit mean over all C(4, 2) pairs; C(9, 2) = 36 |
| `test_pipeline_runs_and_ptx_term_is_used` | pre-training → SFT → rankings (K in range) → RM → PPO and PPO-ptx run; KL starts at 0 |

---

## 5. Try it yourself

1. Train the RM on 5,000 ranked prompts with a wider model (d = 96). Does PPO now beat SFT?
2. Put REVERSE back as a task and measure the RM's accuracy on it alone.
3. Set γ = 3. Does instruction following suffer?
4. Replace `ranking_loss` with shuffled independent pairs (as in E2) and watch for overfitting.
5. Give the base model 3 few-shot demonstrations in the prompt ("few-shot GPT-3"). How far does that get without fine-tuning?
