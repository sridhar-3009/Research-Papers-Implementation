# The code, explained simply

How the code in this folder implements model cards (Mitchell et al. 2019) and the disaggregated evaluation behind them.
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `modelcard.py` | the `ModelCard` dataclass and its Markdown rendering, confusion-matrix rates, bootstrap confidence intervals, unitary + intersectional disaggregation, fairness gaps, a threshold sweep, and the two worked examples (smiling, toxicity v1 vs v2) |
| `experiments.py` | CI width vs group size, seed stability, toxicity over seeds, writing cards to disk (E1–E4) |
| `demo.py` | the full smiling card, its findings, the threshold slider, the toxicity per-term table (~1 second) |
| `test_modelcard.py` | 4 quick tests (~0.6 seconds) |

**Run it** (from `14-Production-ML/093-Mitchell-et-al-2019-Model-Cards`):
```
python3 -m pytest -q
python3 demo.py
python3 experiments.py --quick
```

---

## 2. `modelcard.py`

### The card
| Name | What it does |
|---|---|
| `SECTIONS` | the 9 section names of Figure 1 |
| `ModelCard` | a dataclass with one list of bullets per section; `missing()` lists empty sections; `to_markdown()` renders it, writing "(not provided)" for gaps instead of hiding them |

### Disaggregated evaluation
| Name | What it does |
|---|---|
| `confusion_rates(y, yhat)` | FPR, FNR, FDR, FOR and n |
| `bootstrap_ci(y, yhat, reps)` | resamples indices `reps` × n in one vectorised step; returns the 2.5–97.5 percentile range of each rate |
| `disaggregate(y, score, groups, threshold)` | rows for all data, each group of each factor, and each intersection of the first two factors |
| `fairness_gaps(rows, keys)` | the largest FNR difference (equality of opportunity) and the largest FPR-or-FNR difference (equalized odds) |
| `threshold_sweep` | rates per group at thresholds 0.3–0.7 (the "slider") |
| `table_markdown(rows)` | the Quantitative Analyses table with "estimate [low, high]" cells |

### Example 1: smiling
- **`smiling_data`:** 42% male; older share 35% of men and 20% of women; 48% smiling.
  - mouth cue = smile × (1.3 for men, 2.0 for women) + noise;
  - lines cue = 1.2 × smile + 1.6 × old × (1.0 for men, 0.6 for women) + noise.
- **`smiling_card`:** trains logistic regression on 12,000 examples, disaggregates on 8,000, and fills in all 9 sections.

### Example 2: toxicity
| Name | What it does |
|---|---|
| `IDENTITY`, `TARGETED`, `TOXIC`, `NICE`, `NEUTRAL`, `bow` | the vocabulary and a bag-of-words encoder |
| `comment_corpus(n, mitigate)` | 30% toxic; 65% of toxic comments contain an insult; 8% of non-toxic ones do; identity terms appear in 35% of comments, drawn 85% of the time from the targeted list when toxic and the non-targeted list when not. `mitigate` adds n/3 non-toxic sentences using every identity term |
| `template_set()` | 6 innocent and 6 toxic frames for each of 12 terms (144 sentences) |
| `auc(score, y)` | Mann–Whitney AUC with tie-averaged ranks |
| `toxicity_versions()` | trains v1 and v2; per term: subgroup AUC, pinned AUC (averaged over 20 random backgrounds), BPSN AUC, mean innocent-sentence score, and the term's learned weight |

`REPORTED` holds the paper's claims (sections, metrics, the two cards' findings), checked against the PDF.

---

## 3. `experiments.py`

| Function | Studies |
|---|---|
| `e1` | bootstrap CI width of FPR / FNR for groups of 50 to 5,000 |
| `e2` | whether "older men have the highest FDR" holds over 20 seeds |
| `e3` | mean BPSN AUC of the targeted terms, v1 vs v2, over 5 seeds |
| `e4` | writes the smiling card to `cards/smiling.md` |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_confusion_rates_by_hand` | the 8-photo example: 0.25 / 0.5 / 1/3 / 0.4 |
| `test_bootstrap_ci_contains_estimate` | each interval contains its estimate and is narrow for n = 2,000 |
| `test_card_renders_all_sections_and_smiling_findings` | all 9 headings render; an empty card reports all 9 as missing; older men have the highest FDR; men have the higher FNR |
| `test_toxicity_bias_and_mitigation` | targeted terms have BPSN < 0.8 in v1 and > 0.8 in v2; a neutral term stays at 1.0; AUC on a hand example = 0.75 |

---

## 5. Try it yourself

1. Add a third factor (e.g. lighting) and report all pairwise intersections. How small do the groups get?
2. Choose a separate threshold per group to equalise FNR (equality of opportunity). What happens to FPR?
3. Replace data balancing in the toxicity toy with simply dropping identity-term features. Compare BPSN AUC and overall AUC.
4. Add a `diff(card_a, card_b)` that highlights changed numbers between two model versions.
5. Render the card to HTML with a real threshold slider (one table per threshold, shown on demand).
