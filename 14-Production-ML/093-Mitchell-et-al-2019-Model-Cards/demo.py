"""Model cards in ~3 seconds: the paper's two worked examples on synthetic stand-ins -- a smiling classifier with a
full rendered model card and disaggregated (unitary + intersectional) error rates with bootstrap confidence intervals,
and a toxicity scorer whose per-identity-term evaluation exposes a bias that the overall AUC hides, before and after
mitigation."""

import time

import numpy as np

import modelcard as M

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. A model card for a smiling classifier (cf. Figure 2), rendered to Markdown")
card, rows, (y, s, gender, age) = M.smiling_card()
print(card.to_markdown())
print(f"(empty sections: {card.missing() or 'none'})")

section("2. What the disaggregated numbers show")
r = {k: v[0] for k, v in rows.items()}
worst_fdr = max((k for k in r if " " in k), key=lambda k: r[k]["FDR"])
print(f"  overall: FPR {r['all']['FPR']:.3f}  FNR {r['all']['FNR']:.3f}  FDR {r['all']['FDR']:.3f}  FOR {r['all']['FOR']:.3f}")
print(f"  highest false discovery rate among intersections: {worst_fdr} ({r[worst_fdr]['FDR']:.3f}, CI "
      f"{rows[worst_fdr][1]['FDR'][0]:.3f}-{rows[worst_fdr][1]['FDR'][1]:.3f}) vs young women {r['female young']['FDR']:.3f}")
print(f"  men in aggregate: FNR {r['male']['FNR']:.3f} vs women {r['female']['FNR']:.3f}")
print("  -> the same two findings as the paper's smiling card: many older men are wrongly called 'smiling' (their")
print("     face lines look like smile creases), and smiling men are missed more often (subtler mouth cue)")
gaps = M.fairness_gaps(rows, ["male", "female"])
print(f"  fairness criteria (gender): {', '.join(f'{k} {v:.3f}' for k, v in gaps.items())}")
gaps4 = M.fairness_gaps(rows, ["male old", "male young", "female old", "female young"])
print(f"  over the 4 intersections:   {', '.join(f'{k} {v:.3f}' for k, v in gaps4.items())}")
print("  -> neither equality of opportunity (equal FNR) nor equalized odds (equal FPR and FNR) holds; the")
print("     intersectional gaps are larger than the unitary ones")

section("3. The threshold slider: error rates for older vs younger men as the decision threshold moves")
masks = {"male old": (gender == "male") & (age == "old"), "male young": (gender == "male") & (age == "young")}
for t, v in M.threshold_sweep(y, s, masks).items():
    print(f"    threshold {t:.1f}:  older men FPR {v['male old']['FPR']:.3f} FNR {v['male old']['FNR']:.3f}   "
          f"younger men FPR {v['male young']['FPR']:.3f} FNR {v['male young']['FNR']:.3f}")
print("  -> no single threshold equalises both groups: raising it fixes older men's false positives but makes younger")
print("     men's misses worse -- which is why the card must state its threshold")

section("4. A toxicity scorer evaluated per identity term (cf. Figure 3): v1 vs a bias-mitigated v2")
t = M.toxicity_versions()
names = list(t)
print(f"  overall AUC on the identity-phrase templates: v1 {t[names[0]]['overall AUC']:.3f}, v2 {t[names[1]]['overall AUC']:.3f}")
print("    term          v1: subgroup  pinned  BPSN  non-toxic score   |  v2: subgroup  pinned  BPSN  non-toxic score")
for term in M.IDENTITY:
    a, b = t[names[0]]["per term"][term], t[names[1]]["per term"][term]
    flag = "  <- targeted in training data" if term in M.TARGETED else ""
    print(f"    {term:12s}      {a['subgroup AUC']:.2f}    {a['pinned AUC']:.2f}   {a['BPSN AUC']:.2f}   {a['mean score of non-toxic templates']:.2f}"
          f"            |      {b['subgroup AUC']:.2f}    {b['pinned AUC']:.2f}   {b['BPSN AUC']:.2f}   "
          f"{b['mean score of non-toxic templates']:.2f}{flag}")
print("  -> in v1 an innocent sentence like 'i am gay' scores ~0.45 toxic vs ~0.05 for other terms, because those words")
print("     appeared mostly inside abusive comments; BPSN AUC (innocent mentions vs others' toxic sentences) drops to")
print("     0.64 for exactly those terms, as the paper's card found for 'lesbian', 'gay', 'homosexual'. Adding non-toxic")
print("     training sentences with every identity term (v2) lifts it to 0.88-0.97 -- better, not fully fixed")
print("  -> subgroup AUC is 1.00 everywhere: within one term, toxic templates still outrank innocent ones. The bias only")
print("     shows when a term's innocent sentences are compared with OTHER sentences -- choose metrics accordingly")

section("What the paper says (verified against the PDF)")
for k, v in M.REPORTED.items():
    print(f"  {k}: {v}")
print(f"\n(done in {time.time() - T0:.1f} s)")
