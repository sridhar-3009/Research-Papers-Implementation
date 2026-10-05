"""Encoding vs training in ~15 seconds: five ways to build a dictionary x five ways to encode with it, on 8x8 digits
(4x4 patches, normalised and whitened, 48 atoms, quadrant pooling, logistic regression on 20 labelled images per
class) -- the analogue of the paper's Table 1."""

import time

import numpy as np

import encoding as E

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


images, y = E.load_digits_images()
P = E.Pipeline(images)
rng = np.random.default_rng(0)
d = 48

section("1. The pipeline")
print(f"  {len(y)} digit images -> {P.patches.shape[1]} patches of 4x4 each (stride 1) -> per-patch normalisation ->")
print(f"  ZCA whitening -> dictionary of d = {d} unit-norm atoms -> encoder (2d features, polarity split) ->")
print(f"  average pooling over 4 quadrants (4 x 2 x {d} = {8 * d} features) -> logistic regression on 20 labelled images")
print("  per class, tested on the other ~1,600 images (mean of 3 random label draws)")
print(f"  baseline, logistic regression on raw pixels: {E.few_label_accuracy(images.reshape(len(y), -1), y):.3f}")

section("2. Training x encoding (cf. Table 1)")
t = time.time()
D = {name: E.train_dictionary(P.sample[:8000], d, rng=rng, **kw) for name, kw in E.TRAINERS.items()}
print(f"  dictionaries trained in {time.time() - t:.1f} s (sparse coding takes by far the longest)")
cols = ["natural"] + list(E.ENCODERS)
print("    dictionary \\ encoder   " + "  ".join(f"{c:>8s}" for c in cols))
table = {}
for tn, Dn in D.items():
    row = []
    for en in cols:
        m, pa = E.NATURAL[tn] if en == "natural" else E.ENCODERS[en]
        row.append(E.few_label_accuracy(P.features(Dn, m, pa), y))
    table[tn] = dict(zip(cols, row))
    print(f"    {tn:22s}  " + "  ".join(f"{v:8.3f}" for v in row))
print("  natural encoders: R and RP use the soft threshold with alpha = 0 (random projections, as in the paper);")
print("  OMP-k uses OMP-k; SC uses sparse coding with the same lambda")

section("3. What the table says")
sc_col = [table[t]["SC"] for t in table]
t_col = [table[t]["T"] for t in table]
o1_col = [table[t]["OMP-1"] for t in table]
print(f"  with the sparse-coding encoder every dictionary scores {min(sc_col):.3f}-{max(sc_col):.3f}; with the soft "
      f"threshold {min(t_col):.3f}-{max(t_col):.3f}")
print(f"  -> the DICTIONARY barely matters: random patches (RP) or VQ (OMP-1) are as good as sparse coding (SC)")
print(f"  with the hard OMP-1 (vector quantization) encoder: {min(o1_col):.3f}-{max(o1_col):.3f}")
gap = [table[t]["T"] - table[t]["OMP-1"] for t in table]
print(f"  -> the ENCODER matters: VQ-style encoding loses {min(gap) * 100:.0f}-{max(gap) * 100:.0f} points against the soft threshold, "
      f"whatever the dictionary (OMP-5 sits in between)")
print(f"  'natural' pairs: SC + SC {table['SC']['natural']:.3f} vs OMP-1 + OMP-1 {table['OMP-1']['natural']:.3f} -- sparse coding")
print(f"  'wins' over VQ, but give the VQ dictionary the soft threshold and it reaches {table['OMP-1']['T']:.3f}")
print("  Honest difference: on CIFAR random Gaussian dictionaries were clearly worse (73.2 vs ~78-79 with T); on our")
print(f"  16-dimensional patches they are nearly as good ({table['R']['T']:.3f})")

section("What the paper reports (verified against the PDF)")
for k, v in E.REPORTED.items():
    print(f"  {k}: {v}")
print(f"\n(done in {time.time() - T0:.1f} s)")
