# The code, explained simply

How the code in this folder implements Constitutional AI (Bai et al. 2022).
Read [EXPLAINED.md](EXPLAINED.md) first.

The language models are **simulated**. Prompt formats, label processing, the preference model and the RL objective follow the paper.

---

## 1. The files

| File | What it is |
|---|---|
| `cai.py` | constitution (critique/revision pairs) and feedback principles; critique, revision and multiple-choice prompt builders; soft labels and clamping; a text toy world (harmful and benign requests, five response styles, a harm judge, true utility); a simulated critic/reviser; the critique → revision chain; a simulated feedback model (accuracy, chain of thought, principle biases); a soft-target Bradley–Terry preference model; KL-regularised RL over response styles; policy summaries |
| `experiments.py` | a real critique/revision loop and real AI labels with an open model; toy sweeps of feedback accuracy, CoT, label processing and principle ensembling |
| `demo.py` | a revision chain on text, harm vs revisions, label formats, RLAIF vs crowd-labelled RLHF (~0.4 seconds) |
| `test_cai.py` | 4 quick tests (~0.1 seconds) |

**Run it** (from `10-Alignment/068-Bai-et-al-2022-Constitutional-AI`):
```
python3 -m pytest -q             # ~0.1 seconds
python3 demo.py                  # ~0.4 seconds
python3 experiments.py --quick
```

---

## 2. `cai.py`

### Prompts and labels
| Name | What it is |
|---|---|
| `CONSTITUTION` | (critique request, revision request) pairs in the paper's style |
| `FEEDBACK_PRINCIPLES` | multiple-choice "which response is less harmful…" principles |
| `critique_prompt`, `revision_prompt`, `feedback_prompt(cot=...)` | the exact prompt layouts of Sections 3–4 |
| `soft_label(logp_a, logp_b)` | p(A)/(p(A) + p(B)) |
| `clamp_label(p, 0.4, 0.6)` | CoT clamping |

### The text world
- **Requests:** `HARMFUL_TOPICS` and `BENIGN_TOPICS`.
- **`respond(style, topic)`:** five styles: comply, evasive, explain, lecture, helpful.
- **`style_of(text)`:** recovers the style from the text.
- **`harm_score(text)`:** 1 for instructions toward a harmful topic.
- **`true_utility(harmful, style)`:** careful human raters' utility.
  - On harmful requests: explain > evasive > lecture > comply.
  - On benign requests: helpful > others.

### Stage 1
- **`SimulatedModel(notice, fix_after_critique, fix_direct)`:** `critique` notices harm with probability `notice`; `revise` fixes harm (usually by explaining) with probability `fix_after_critique`, or `fix_direct` when there is no critique.
- **`critique_revision_chain(model, topic, n, use_critique)`:** the initial harmful answer plus n revisions under randomly drawn principles.

### Stage 2
- **`feedback_model_label(harmful, a, b, rng, accuracy, cot, principle_bias)`:** a soft P(A) from noisy utility differences; with CoT it becomes 0.98 or 0.02.
- **`PRINCIPLE_BIASES`:** one per principle; e.g. the "polite" one over-rates lecturing.
- **`featurise`, `make_comparisons`:** (request type × style) one-hot features; helpfulness pairs are labelled by humans and harmlessness pairs by the labeler under test.
- **`train_preference_model`:** a linear Bradley–Terry model with soft targets.
- **`rl_against_pm(w, init_logits, beta)`:** a softmax policy over styles per request type, maximising E[r] − β·KL(π‖π₀) by exact gradient ascent.
- **`summarise_policy`:** true utility plus the rates of harmful compliance, evasion, explanation, lecturing and helpfulness.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | a real model answers red-team prompts, critiques and revises 4 times; the same model judges each revision against the original via (A)/(B) log-probabilities |
| `e2` | real label accuracy per principle on (explained refusal vs compliance) pairs in both orders; ensemble mean |
| `e3` | toy: feedback accuracy 0.55 … 0.95, with and without CoT (clamped) → policy true utility |
| `e4` | toy: soft-ensemble vs single principle vs hard vs clamped 20–80 / 40–60 labels |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_prompts_and_labels` | prompt layouts; soft label 0.3/(0.3+0.1) = 0.75; clamping |
| `test_text_world` | harm judge, style recovery, utility ordering |
| `test_critique_revision_reduces_harm_and_critique_helps` | harm decreases monotonically with revisions; critiques beat direct revision |
| `test_feedback_labels_and_preference_rl` | soft vs CoT labels; the preference model plus RL prefers explaining over complying |

---

## 5. Try it yourself

1. Make the crowd labels less evasion-loving (bias 0.5 instead of 1.5). When does RLHF stop being evasive?
2. Give every principle a bias toward lecturing. Can ensembling still help?
3. Lower the feedback model's accuracy to 0.6. Does chain of thought (with clamping) rescue it?
4. Set β = 0 in `rl_against_pm` and use clamped labels. What happens to the policy's sharpness?
5. Run `experiments.py --only e2` with a small instruction model and check which principle gives the most accurate labels.
