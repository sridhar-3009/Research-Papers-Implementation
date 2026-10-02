"""A light tour of BERT (Devlin et al. 2019). About 6 seconds.

    python3 demo.py
"""

import collections
import random

import torch
import torch.nn.functional as F

from bert import (CLS, IGNORE, MASK, PAD, SEP, Bert, build_wordpiece_vocab, collate, count_params, mask_tokens,
                  nsp_pairs, optimizer, pack_pair, wordpiece_tokenize)

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. WordPiece: frequent words stay whole, rare ones split into known pieces")
counts = collections.Counter({"play": 30, "playing": 20, "played": 15, "player": 10, "plays": 8, "jump": 12,
                              "jumping": 6, "sing": 9, "singing": 5})
vocab = set(build_wordpiece_vocab(counts, 40))
for w in ("playing", "jumped", "singer", "player"):
    print(f"  {w:8s} -> {wordpiece_tokenize(w, vocab)}")

line("2. Figure 2: one input = token + segment + position, for a sentence PAIR")
names = {CLS: "[CLS]", SEP: "[SEP]", 10: "my", 11: "dog", 12: "is", 13: "cute", 14: "he", 15: "likes", 16: "play", 17: "##ing"}
ids, seg = pack_pair([10, 11, 12, 13], [14, 15, 16, 17])
print("  tokens  :", " ".join(f"{names[i]:>6s}" for i in ids))
print("  segment :", " ".join(f"{'A' if s == 0 else 'B':>6s}" for s in seg))
print("  position:", " ".join(f"{p:>6d}" for p in range(len(ids))))

line("3. Masked LM inputs: 15% chosen; of those 80% [MASK], 10% random, 10% unchanged")
x = torch.randint(5, 1000, (1, 20))
inp, lab = mask_tokens(x, 1000, generator=torch.Generator().manual_seed(3))
print("  original:", x[0].tolist())
print("  input   :", ["[M]" if t == MASK else t for t in inp[0].tolist()])
print("  targets :", [t if t != IGNORE else "." for t in lab[0].tolist()])
big = torch.randint(5, 30000, (200, 512))
inp, lab = mask_tokens(big, 30000, generator=torch.Generator().manual_seed(0))
ch = lab != IGNORE
print(f"  over 102,400 tokens: chosen {ch.float().mean():.3f}; of those [MASK] {(inp == MASK)[ch].float().mean():.3f}, "
      f"unchanged {(inp == big)[ch].float().mean():.3f}, random {((inp != MASK) & (inp != big))[ch].float().mean():.3f}")
print("-> the 10% random / 10% unchanged keep the model from relying on [MASK], which never appears when fine-tuning.")

line("4. Why bidirectional? Each 'marker' token is decided by the token to its RIGHT")
V = 25


def seqs(N, g=None):
    xx = torch.randint(5, 15, (N, 6), generator=g)
    s = torch.stack([xx + 10, xx], 2).reshape(N, 12)
    return torch.cat([torch.full((N, 1), CLS), s, torch.full((N, 1), SEP)], 1)


test = seqs(500, torch.Generator().manual_seed(1))
for causal in (False, True):
    torch.manual_seed(0)
    m = Bert(V, H=48, L=2, A=4, max_len=32, dropout=0.0, causal=causal)
    opt = optimizer(m, 3e-3)
    for _ in range(300):
        s = seqs(64)
        seg = torch.zeros_like(s)
        if causal:                                  # left-to-right LM: predict the next token (like OpenAI GPT)
            T, _ = m(s, seg, s != PAD)
            loss = F.cross_entropy(m.mlm_logits(T)[:, :-1].reshape(-1, V), s[:, 1:].reshape(-1))
        else:                                       # masked LM
            i, l = mask_tokens(s, V)
            T, _ = m(i, seg, s != PAD)
            loss = F.cross_entropy(m.mlm_logits(T).reshape(-1, V), l.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        if causal:
            pred = m.mlm_logits(m(test, torch.zeros_like(test), test != PAD)[0]).argmax(-1)[:, :-1]
            tgt = test[:, 1:]; sel = tgt >= 15
        else:
            inp = test.clone(); pos = torch.arange(1, 13, 2); inp[:, pos] = MASK
            pred = m.mlm_logits(m(inp, torch.zeros_like(test), inp != PAD)[0]).argmax(-1)
            tgt, sel = test, torch.zeros_like(test, dtype=torch.bool); sel[:, pos] = True
    acc = (pred[sel] == tgt[sel]).float().mean().item()
    print(f"  {'left-to-right LM' if causal else 'masked LM (BERT)'}: predicts the markers {100 * acc:5.1f}% right (chance 10%)")
print("-> a left-to-right model never sees the right context; BERT's masked LM uses both sides in every layer.")

line("5. Next sentence prediction on toy 'documents' (each document has its own word range and order)")
rng = random.Random(0)
docs = [[[5 + 4 * d + (s % 4), 100 + s] for s in range(6)] for d in range(20)]
torch.manual_seed(0)
m = Bert(200, H=64, L=2, A=4, max_len=16, dropout=0.0)
opt = optimizer(m, 1e-3)
for _ in range(500):
    batch = nsp_pairs(docs, 32, rng)
    ids, seg, mask = collate([pack_pair(a, b) for a, b, _ in batch])
    loss = F.cross_entropy(m.nsp(m(ids, seg, mask)[1]), torch.tensor([lab for _, _, lab in batch]))
    opt.zero_grad(); loss.backward(); opt.step()
m.eval()
batch = nsp_pairs(docs, 500, random.Random(9))
ids, seg, mask = collate([pack_pair(a, b) for a, b, _ in batch])
with torch.no_grad():
    acc = (m.nsp(m(ids, seg, mask)[1]).argmax(-1) == torch.tensor([lab for _, _, lab in batch])).float().mean()
print(f"  IsNext / NotNext accuracy: {100 * acc:.1f}%   (the real BERT reaches 97-98%)")

line("6. Sizes")
with torch.device("meta"):
    print(f"  BERT-base : {count_params(Bert(30522)) / 1e6:6.1f}M parameters (paper: 110M)")
    print(f"  BERT-large: {count_params(Bert(30522, H=1024, L=24, A=16)) / 1e6:6.1f}M parameters (paper: 340M)")
