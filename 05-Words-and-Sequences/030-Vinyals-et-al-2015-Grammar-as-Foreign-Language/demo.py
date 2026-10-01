"""A light tour of 'Grammar as a Foreign Language' (Vinyals et al. 2015) on a toy grammar. About 10 seconds.

    python3 demo.py
"""

import random

import torch

import grammar
from parser import (PAD, LSTMA, Vocab, batch_pairs, beam_search, clean_ptb, delinearize, evalb, is_well_formed,
                    linearize, parse_tree, to_string, words)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. Linearizing a tree (Figure 2)")
t = clean_ptb(parse_tree("( (S (NP-SBJ (NNP John)) (VP (VBZ has) (NP (DT a) (NN dog))) (. .)) )"))
print("  tree      :", to_string(t))
print("  with tags :", " ".join(linearize(t, normalize_pos=False)))
print("  normalized:", " ".join(linearize(t)), "   <- what the model outputs (POS tags -> XX)")

line("2. Train LSTM+A and a baseline LSTM (no attention), same budget, on a toy grammar")
train, test = grammar.corpus(3000, 0), grammar.corpus(150, 99)
wv = Vocab([w for t in train for w in words(t)])
lv = Vocab([s for t in train for s in linearize(t)])
enc = lambda t: (wv.encode(words(t)), lv.encode(linearize(t)))
pairs = [enc(t) for t in train]
print(f"  {len(train)} training trees, {len(lv)} output symbols, test sentences of {min(len(words(t)) for t in test)}"
      f"-{max(len(words(t)) for t in test)} words")
models, results = {}, {}
for name, att in (("LSTM+A", True), ("baseline LSTM", False)):
    torch.manual_seed(0)
    rng = random.Random(0)
    m = LSTMA(len(wv), len(lv), d=48, emb=32, layers=1, attention=att)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(300):
        src, tgt = batch_pairs(rng.sample(pairs, 32))
        loss = -m.log_prob(src, tgt).sum() / (tgt != PAD).sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5); opt.step()
    m.eval()
    outs = [beam_search(m, batch_pairs([enc(t)])[0], beam=1) for t in test]
    seqs = [[lv.itos[s] for s in o[1]] for o in outs]
    preds = [delinearize(s, words(t)) for s, t in zip(seqs, test)]
    models[name], results[name] = m, (preds, outs)
    short = [i for i, t in enumerate(test) if len(words(t)) <= 10]
    long_ = [i for i, t in enumerate(test) if len(words(t)) > 10]
    f_all = evalb(test, preds)["F1"]
    f_s = evalb([test[i] for i in short], [preds[i] for i in short])["F1"]
    f_l = evalb([test[i] for i in long_], [preds[i] for i in long_])["F1"]
    bad = sum(not is_well_formed(s) for s in seqs)
    print(f"  {name:13s}: F1 {f_all:5.1f}   (<= 10 words: {f_s:5.1f}, > 10 words: {f_l:5.1f})   malformed {bad}/{len(test)}")
print("-> attention helps, and most on long sentences (Table 1, Figure 3). Malformed trees are rare and are")
print("   fixed by adding brackets at the ends, as in the paper.")

line("3. Where does the attention point when the model emits a word slot 'XX'? (Figure 4)")
hits = mono = n = n_pairs = 0
for t, (_, toks, att) in zip(test, results["LSTM+A"][1]):
    T = len(words(t))
    pos = [T - 1 - torch.tensor(a).argmax().item() for s, a in zip(toks, att) if lv.itos[s] == "XX"]
    hits += sum(abs(p - k) <= 1 for k, p in enumerate(pos)); n += len(pos)
    mono += sum(b >= a for a, b in zip(pos, pos[1:])); n_pairs += max(len(pos) - 1, 0)
print(f"  attention peak moves right (or stays) between consecutive XX: {100 * mono / n_pairs:.0f}% of the time")
print(f"  peak within one word of the word being consumed:              {100 * hits / n:.0f}%")
print("-> the focus sweeps left to right as words are consumed, as in Figure 4, but in this small, briefly trained")
print("   model it is blurrier than the paper's sharp one-word pointer.")
t = test[1]
_, toks, _ = results["LSTM+A"][1][1]
print("\n  sentence:", " ".join(words(t)))
print("  output  :", " ".join(lv.itos[s] for s in toks))
