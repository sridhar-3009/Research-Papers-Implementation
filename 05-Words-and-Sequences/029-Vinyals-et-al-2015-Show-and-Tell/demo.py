"""A light tour of Show and Tell (Vinyals et al. 2015) on a toy image world. A few seconds.

    python3 demo.py
"""

import random

import torch

import toy
from nic import (NIC, beam_search, bleu, cider_d, human_bleu, nearest_words, normalize_for_annotation, novelty,
                 pad_captions, ranks,
                 recall_report, sample, score_matrix)

torch.manual_seed(0)
rng = random.Random(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. Train NIC: image -> LSTM (fed once) -> words. 4 of the 24 object kinds are held out")
train, held = toy.split()
m = NIC(len(toy.WORDS), toy.FEAT, d=48)
opt = torch.optim.Adam(m.parameters(), 1e-2)
for it in range(300):
    kinds, caps = toy.batch(train, 64, rng)
    loss = -m.log_prob(toy.features(kinds), pad_captions(caps)).sum() / sum(len(c) + 1 for c in caps)
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 100 == 0 or it == 299:
        print(f"  step {it:3d}: loss per word {loss.item():.3f}")
m.eval()
train_caps = {tuple(toy.encode(c)) for k in train for c in toy.captions(*k)}

line("2. Captions for images whose combination was NEVER seen in training (Table 3: N-best lists)")
for kind in held:
    nb = beam_search(m, toy.features([kind], noise=0.0)[0], beam=5)
    print(f"  image = {' '.join(kind)}")
    for s, t in nb[:4]:
        tag = "" if tuple(t) in train_caps else "   <- novel: not a training sentence"
        print(f"     {s:6.2f}  {toy.decode(t)}{tag}")

line("3. Greedy (beam 1) vs beam 5 vs sampling, on 200 noisy test images of all kinds")
gen = torch.Generator().manual_seed(0)
kinds = [rng.choice(toy.ALL) for _ in range(200)]
feats = toy.features(kinds, generator=gen)
refs = [[toy.encode(c) for c in toy.captions(*k)] for k in kinds]
out = {"greedy": [beam_search(m, f, beam=1)[0][1] for f in feats],
       "beam 5": [beam_search(m, f, beam=5)[0][1] for f in feats],
       "sample": sample(m, feats, generator=gen)}
for name, hyps in out.items():
    print(f"  {name:7s}: BLEU-1 {bleu(hyps, refs, 1):5.1f}  BLEU-4 {bleu(hyps, refs):5.1f}  CIDEr-D {cider_d(hyps, refs):6.1f}"
          f"  novel {100 * novelty(hyps, train_caps):4.1f}%")
print(f"  human  : BLEU-1 {human_bleu(refs, 1):5.1f}  BLEU-4 {human_bleu(refs):5.1f}   (each reference vs the other 4)")
print("-> beam search beats greedy on BLEU (the paper: greedy costs ~2 BLEU). The model 'beats' humans only because")
print("   the 5 human captions are deliberately different styles (no 4-gram is shared, so human BLEU-4 is 0); the")
print("   paper warns BLEU does not match human judgement (Sec. 4.3.6).")

line("4. Ranking (Tables 4-5): score log p(caption | image) for every pair, 24 images x 120 captions")
feats = toy.features(toy.ALL, noise=0.0)
caps, gt = [], []
for i, k in enumerate(toy.ALL):
    for c in toy.captions(*k):
        caps.append(toy.encode(c)); gt.append(i)
S = score_matrix(m, feats, pad_captions(caps))
ann, search = ranks(S, gt)
print("  image annotation:", {k: round(v, 1) for k, v in recall_report(ann).items()})
ann_n, _ = ranks(normalize_for_annotation(S), gt)
print("  ... normalized  :", {k: round(v, 1) for k, v in recall_report(ann_n).items()}, " (divide by the caption prior)")
print("  image search    :", {k: round(v, 1) for k, v in recall_report(search).items()})
print("-> a generative model ranks too, by p(S|I). Raw annotation favours short, generic captions; dividing by the")
print("   caption's prior fixes it here. Ambiguous captions ('there is a big circle' fits 4 images) keep")
print("   image search below 100%.")

line("5. Word embeddings W_e (Table 6): nearest neighbours")
for w in ("red", "circle", "big"):
    print(f"  {w:7s} -> {[toy.WORDS[j] for j in nearest_words(m, toy.STOI[w], 3)]}")
