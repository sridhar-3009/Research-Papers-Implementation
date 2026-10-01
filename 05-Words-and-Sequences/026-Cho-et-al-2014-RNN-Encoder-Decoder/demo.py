"""A light tour of Cho et al. (2014): the GRU and the encoder-decoder. A few seconds.

    python3 demo.py
"""

import torch

from encdec import EOS, EncoderDecoder, GRUCell, bleu

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. The GRU's two gates")
cell = GRUCell(1, 1)
with torch.no_grad():
    cell.W.weight.copy_(torch.tensor([[1.0], [0.0], [0.0]])); cell.U_zr.weight.zero_(); cell.U.weight.fill_(1.0)
for z_bias, r_bias, label in ((5.0, 0.0, "update gate z ~ 1: keep the old state"),
                              (-5.0, 5.0, "z ~ 0, reset r ~ 1: new state from input AND old state"),
                              (-5.0, -5.0, "z ~ 0, reset r ~ 0: new state from the input ONLY ('reset')")):
    with torch.no_grad():
        cell.W.weight[1, 0], cell.W.weight[2, 0] = z_bias, r_bias        # x = 1 acts as a bias here
    h_old = torch.tensor([[0.9]])
    print(f"{label:60s} h: 0.900 -> {cell(torch.tensor([[1.0]]), h_old).item():.3f}")
print("-> z decides how much of the old state to keep (like an LSTM's memory); r decides whether to")
print("   look at the old state when forming the new candidate (dropping information found irrelevant).")

line("2. The encoder-decoder learns to reverse 3-symbol 'phrases' (a toy translation)")
torch.manual_seed(1)
m = EncoderDecoder(8, 8, n=48, emb=16, maxout=24, rank=12)
opt = torch.optim.Adam(m.parameters(), 1e-2)
for step in range(400):
    src = torch.randint(3, 8, (3, 64))
    tgt = torch.cat([src.flip(0), torch.full((1, 64), EOS)])
    loss = -m.log_prob(src, tgt).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step:3d}: -log p(y|x) = {loss.item():.3f}")
test = torch.randint(3, 8, (3, 200))
out = m.generate(test, max_len=4)
hyps = [[w for w in col if w != EOS] for col in out.T.tolist()]
refs = test.flip(0).T.tolist()
print(f"after training: {sum(h == r for h, r in zip(hyps, refs))}/200 test phrases reversed exactly, "
      f"BLEU-3 {bleu(hyps, refs, max_n=3):.1f} (BLEU-4 is {bleu(hyps, refs):.0f}: 3-word phrases have no 4-grams to match)")
print("-> the whole source is squeezed into ONE vector c, and the decoder rebuilds the target from it.")

line("3. Scoring phrase pairs, as the paper does inside the SMT system")
src = torch.tensor([[3], [5], [7]])
for cand in ([7, 5, 3], [3, 5, 7], [7, 7, 7]):
    lp = m.log_prob(src, torch.tensor([[c] for c in cand] + [[EOS]])).item()
    print(f"source 3 5 7 -> candidate {cand}: log p(y|x) = {lp:7.3f}")
print("-> the correct (reversed) phrase gets by far the highest score: that score becomes one more")
print("   feature in the log-linear model of Eq. (9).")
