"""A light tour of Sequence to Sequence learning (Sutskever, Vinyals & Le 2014). A few seconds.

    python3 demo.py
"""

import torch
import torch.nn.functional as F

from seq2seq import BOS, EOS, PAD, Seq2Seq, beam_search, count_params, paper_init_, time_lags

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. The paper's model size")
with torch.device("meta"):
    m = Seq2Seq(160_000, 80_000, 1000, 1000, 4)
parts = {"source embeddings": m.src_emb, "target embeddings": m.tgt_emb, "encoder LSTM (4 layers)": m.encoder,
         "decoder LSTM (4 layers)": m.decoder, "softmax over 80k words": m.out}
for name, mod in parts.items():
    print(f"{name:26s} {count_params(mod) / 1e6:6.1f}M")
print(f"{'total':26s} {count_params(m) / 1e6:6.1f}M   (paper: 384M; sentence vector = 4 x (h + c) x 1000 = 8000 numbers)")

line("2. Why reversing the source helps (Section 3.3)")
for n in (5, 20, 50):
    f, r = time_lags(n, False), time_lags(n, True)
    print(f"{n:2d}-word sentences: forward lags min {min(f):2d} mean {sum(f) / n:4.1f} | reversed lags min {min(r):2d} mean {sum(r) / n:4.1f}")
print("-> the AVERAGE distance between a source word and its translation is unchanged, but the first")
print("   words become close neighbours: SGD can 'establish communication' early, then build on it.")

line("3. Reversed vs forward source on a toy translation (copy a 12-symbol sentence)")
for reverse in (False, True):
    torch.manual_seed(0)
    model = paper_init_(Seq2Seq(14, 14, emb=32, hidden=64, layers=1))
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    for step in range(301):
        s = torch.randint(3, 14, (12, 64))
        src = s.flip(0) if reverse else s
        tgt = torch.cat([s, torch.full((1, 64), EOS)])
        tgt_in = torch.cat([torch.full((1, 64), BOS), tgt[:-1]])
        loss = F.cross_entropy(model(src, tgt_in).reshape(-1, 14), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"{'reversed' if reverse else 'forward '} source: loss after 300 steps {loss.item():.3f}")
print("(small toy; the paper's numbers: perplexity 5.8 -> 4.7, BLEU 25.9 -> 30.6 on WMT'14)")

line("4. Beam search keeps the B best prefixes")
src = torch.randint(3, 14, (12, 1))
for beam in (1, 2, 12):
    best = beam_search([model], src.flip(0), beam=beam, max_len=15)[0]
    print(f"beam {beam:2d}: log p = {best[0]:8.3f}, output starts {best[1][:6]}")
print(f"(source was {src[:6, 0].tolist()}... ; beam 1 = greedy decoding)")
print("-> here each wider beam found a MORE PROBABLE output (usual, not guaranteed), but beam 12 found the EMPTY sentence:")
print("   a sum of log-probabilities favours short outputs, and this toy model (loss ~1.8) still gives")
print("   '<EOS>' at step 1 a probability of ~e^-6.5. A well-trained model makes early <EOS> very")
print("   unlikely (the paper's beam of 12 helped); later systems divide by the length ('length")
print("   normalization', Wu et al. 2016) to remove the bias.")
