# The code, explained simply

How the code in this folder implements learning to summarise from human feedback (Stiennon et al. 2020).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `rlhf.py` | a toy summarisation task with a known human utility, noisy references, simulated labelers; tiny LLaMA-style networks (policy, reward model, value function); SFT; sampling and per-token log-probabilities; comparisons; Bradley–Terry RM training with reference normalisation; PPO with per-token KL, clipping, GAE and a separate value network; evaluation |
| `experiments.py` | β sweep, RM scaling, best-of-n vs PPO, ROUGE-like reward vs RM, a real RM on `openai/summarize_from_feedback` (heavy, not run here) |
| `demo.py` | the task, SFT, RM, PPO with β = 0 and 0.2, results table, the paper's numbers (~28 seconds) |
| `test_rlhf.py` | 4 quick tests (~1 second) |

**Run it** (from `10-Alignment/066-Stiennon-et-al-2020-Summarize-from-Human-Feedback`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~28 seconds
python3 experiments.py --quick
```

---

## 2. `rlhf.py`

### The task
| Name | What it is |
|---|---|
| `make_post(rng)` | 14 tokens: topics ×4, ×3, ×2, plus filler, shuffled; returns (post, top-3 topics) |
| `reference_summary(top, rng)` | noisy TL;DRs: 50% main topic only, 20% both main topics, 15% wrong second topic, 15% both + filler |
| `true_quality(post, top, summary)` | the labelers' utility |
| `labeler_prefers(qa, qb, rng, noise)` | noisy Bradley–Terry choice |
| `encode(post, summary)` | ids: POST post… SUM summary… EOS |

`true_quality` scores: +1 per main topic, +0.3 for the right order, −1 per hallucinated token, −0.5 per filler, −0.6 per repetition, −0.4 per token beyond 3.

### Models
- **`LM(d, layers, scalar)`:** paper 056's LLaMA body. With `scalar=True`, a linear head gives one number per position (the RM or the value function).
- **`sample(policy, posts, temperature)`:** batched generation of summaries (it never emits PAD, POST or SUM).
- **`logprobs(model, seqs)`:** per-token log-probabilities, plus a mask covering the summary tokens and EOS.

### The pipeline
1. **`train_sft(steps, d)`:** cross-entropy on references; the loss covers summary tokens only.
2. **`collect_comparisons(policy, n)`:** two samples per post (25% of the time the second is a reference), labelled by the simulated labelers.
3. **`train_reward_model(sft, data)`:** copies the SFT body and adds a scalar head read at the last token.
   - The loss is BCE-with-logits on r_A − r_B, which equals −log σ(r_chosen − r_rejected).
   - Afterwards it sets `rm.bias` so references average 0.
   - **`reward(rm, posts, summaries)`:** the normalised score; **`rm_accuracy`:** held-out agreement.
4. **`ppo(sft, rm, beta, ...)`:** for each iteration:
   - sample summaries from the current policy;
   - per-token reward = −β(log π − log π_SFT), plus r_RM at the last token (or `reward_fn` if given);
   - GAE advantages from the value network (initialised from the RM), normalised;
   - 4 epochs of the clipped PPO objective plus the value loss;
   - it records (RM score, true quality, KL).

### Evaluation
- **`evaluate_policy(policy, rm)`:** greedy summaries on 400 fresh posts. It reports the mean true quality, the mean RM score, the preference rate against references (σ of the quality difference), the mean length, and an example.

---

## 3. `experiments.py`

| Function | Reproduces |
|---|---|
| `e1` | Figure 5: β ∈ {0, 0.01, 0.03, 0.1, 0.3, 1}: KL, RM score, true quality, preference |
| `e2` | Figure 6: RM accuracy for 250 … 8000 comparisons and widths 32 / 64 / 128 |
| `e3` | best-of-n (n = 1 … 64) from SFT, ranked by the RM, vs PPO |
| `e4` | PPO on a ROUGE-1-like overlap with one fixed noisy reference per post vs PPO on the RM |
| `e5` | a real RM: a small LM + head on `openai/summarize_from_feedback`; held-out accuracy |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_task_and_true_quality` | post composition; ideal 2.3, single topic 1.0; repetition and hallucination penalties; ~50% one-topic references |
| `test_bradley_terry_loss_matches_paper_formula` | BCE on r_A − r_B equals −log σ(r_chosen − r_rejected) |
| `test_encode_sample_logprobs_shapes` | sequence layout; sampling; the log-prob mask covers the summary + EOS |
| `test_pipeline_runs_end_to_end` | SFT → comparisons → RM (references normalised to ~0) → PPO (KL 0 on the first batch) |

---

## 5. Try it yourself

1. Run `ppo` with β = 0.02, 0.05, 0.5 and plot true quality against KL. Where is the peak?
2. Make labelers less noisy (`noise=0.2`). How much better do the RM and the final policy get?
3. Give the RM only 300 comparisons. Does PPO still beat SFT, or does it hack the weaker RM?
4. Remove the value network (use the batch-mean return as the baseline). Is training noisier?
5. Make references better (60% ideal). Does RLHF still help, and by how much?
