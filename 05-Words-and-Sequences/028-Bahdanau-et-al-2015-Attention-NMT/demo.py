"""A light tour of attention (Bahdanau, Cho & Bengio 2015). About 10-20 seconds of tiny training.

    python3 demo.py
"""

import torch

from attention import EOS, RNNsearch, paper_init_

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. At the start (paper init: v_a = 0) the attention is perfectly uniform")
m = paper_init_(RNNsearch(10, 10, m=8, n=16, l=8, n_align=8))
src = torch.randint(3, 10, (5, 1))
_, alpha = m(src, torch.ones(2, 1, dtype=torch.long), return_alpha=True)
print("alpha for the first output word:", [round(v, 3) for v in alpha[0, :, 0].tolist()])

line("2. Train both models on a toy 'translation' (reverse the sequence) of length 6 and 16")
results = {}
for L in (6, 16):
    for mode in ("encdec", "search"):
        torch.manual_seed(1)
        model = RNNsearch(12, 12, m=16, n=48, l=16, n_align=24, mode=mode)
        opt = torch.optim.Adam(model.parameters(), 5e-3)
        for _ in range(300):
            s = torch.randint(3, 12, (L, 32))
            loss = -model.log_prob(s, torch.cat([s.flip(0), torch.full((1, 32), EOS)])).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        test = torch.randint(3, 12, (L, 200))
        tgt_in = torch.cat([torch.ones(1, 200, dtype=torch.long), test.flip(0)])
        with torch.no_grad():
            pred = model(test, tgt_in)[:L].argmax(-1)
        acc = (pred == test.flip(0)).float().mean().item()
        results[(mode, L)] = (acc, model)
        print(f"length {L:2d}, {'RNNencdec' if mode == 'encdec' else 'RNNsearch'}: per-word accuracy {100 * acc:5.1f}%")
print("-> squeezing a long sentence into ONE vector (encdec) gets harder as it grows; attention lets")
print("   each output word look up the part of the source it needs.")

line("3. The learned soft alignment (Figure 3), length 6: rows = output words, columns = source words")
model = results[("search", 6)][1]
s = torch.randint(3, 12, (6, 1))
with torch.no_grad():
    _, alpha = model(s, torch.cat([torch.ones(1, 1, dtype=torch.long), s.flip(0)]), return_alpha=True)
shades = " .:-=+*#%@"
print("        source: " + " ".join(f"{w:2d}" for w in s[:, 0].tolist()))
for i, out_w in enumerate(s.flip(0)[:, 0].tolist()):
    row = alpha[i, :, 0]
    print(f"output word {out_w:2d}:  " + " ".join(f" {shades[min(9, int(v * 10))]}" for v in row.tolist()))
print("-> output word k attends to source position L-1-k: the anti-diagonal, learned without any")
print("   alignment supervision (the paper's Figure 3 shows the same for English-French word order).")
