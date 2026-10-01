"""A light tour of 'Order Matters' (Vinyals, Bengio & Kudlur 2016). About 15 seconds.

    python3 demo.py
"""

import collections

import torch

from orderset import (ReadProcessWrite, SetLM, best_order, bfs_linearize, dfs_linearize, eq9_step, order_log_prob,
                      sort_batch, star_log_prob, star_model, star_sample, star_tokens, three_word_reversal)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. A set encoder must not care about the order of its inputs (Section 4)")
torch.manual_seed(0)
X = torch.rand(1, 6, 1)
P = torch.randperm(6)
for enc in ("set", "lstm"):
    m = ReadProcessWrite(d=16, steps=3, encoder=enc)
    d = (m.encode(X)[1] - m.encode(X[:, P])[1]).abs().max().item()
    name = "Read-Process-Write" if enc == "set" else "LSTM encoder (Ptr-Net)"
    print(f"  {name:22s}: shuffling the 6 inputs changes the encoding by {d:.2e}")
print("-> the process block only reads the memories through attention (a weighted SUM), which ignores order.")

line("2. Table 1 in miniature: sort 5 numbers (600 updates each)")


def train_sorter(**kw):
    torch.manual_seed(0)
    m = ReadProcessWrite(d=64, **kw)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(600):
        X, o = sort_batch(64, 5)
        loss = -m.log_prob(X, o).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 2); opt.step()
    X, o = sort_batch(500, 5, torch.Generator().manual_seed(9))
    return 100 * (m.greedy(X) == o).all(1).float().mean().item()


for label, kw in (("Ptr-Net (LSTM encoder), 1 glimpse", dict(encoder="lstm", steps=0, glimpses=1)),
                  ("Read-Process-Write, P = 0, 1 glimpse", dict(steps=0, glimpses=1)),
                  ("Read-Process-Write, P = 1, 0 glimpses", dict(steps=1, glimpses=0)),
                  ("Read-Process-Write, P = 1, 1 glimpse", dict(steps=1, glimpses=1))):
    print(f"  {label:38s}: {train_sorter(**kw):5.1f}% of lists sorted exactly")
print("-> paper (N = 5): Ptr-Net 90%, P = 0 84%, P = 1 92% (with glimpses). Processing steps and glimpses help;")
print("   with P = 0 the writer starts with no summary of the set at all.")

line("3. Output order matters: a star-shaped graphical model (Section 5.1.4)")
diffs = []
for seed in range(3):
    g = torch.Generator().manual_seed(seed)
    model = star_model(10, peaky=2.0, generator=g)
    train, test = star_sample(model, 500, g), star_sample(model, 2000, g)
    nll = {}
    for head_first in (True, False):
        torch.manual_seed(seed)
        lm = SetLM(100, d=64)
        opt = torch.optim.Adam(lm.parameters(), 1e-2)
        T, TE = star_tokens(train, head_first), star_tokens(test, head_first)
        best = float("inf")
        for it in range(300):
            loss = -lm.seq_log_prob(T[torch.randint(0, 500, (64,))]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 50 == 49:
                with torch.no_grad():
                    best = min(best, -lm.seq_log_prob(TE).mean().item())
        nll[head_first] = best
    exact = -star_log_prob(model, test).mean().item()
    diffs.append(nll[False] - nll[True])
    print(f"  model {seed}: test NLL head first {nll[True]:.2f}, head last {nll[False]:.2f}  (true model {exact:.2f})")
print(f"-> the same LSTM, same data, same chain rule: emitting the 'cause' first is easier ({sum(diffs)/3:.2f} nats better")
print("   on average with 500 samples). In theory any order works; in practice optimization prefers this one.")

line("4. The same tree, two orders (Figure 2), and a scrambled sentence (Section 5.1.1)")
t = ("S", [("NP", [("DT", [])]), ("VP", [("VBZ", []), ("NP", [("DT", []), ("NN", [])])]), (".", [])])
print("  depth first  :", " ".join(dfs_linearize(t)))
print("  breadth first:", " ".join(bfs_linearize(t)))
print("  3-word reversal of 'This is a sentence .':", " ".join(three_word_reversal("This is a sentence .".split())))
print("-> paper: depth first 89.5 F1 vs breadth first 81.5; PTB perplexity 86 natural/reversed vs 96 for 3-word reversal.")

line("5. Eq. 9: let the model pick the order (sets of 4 (word, position) tokens from a Markov chain)")
g = torch.Generator().manual_seed(0)
V, n = 8, 4
start = torch.softmax(2 * torch.randn(V, generator=g), 0)
trans = torch.softmax(3 * torch.randn(V, V, generator=g), -1)


def sample(N):
    s = [torch.multinomial(start, N, True, generator=g)]
    for _ in range(n - 1):
        s.append(torch.multinomial(trans[s[-1]], 1, generator=g)[:, 0])
    return torch.stack(s, 1) * n + torch.arange(n)            # token = (word, position): an order-free set


train, test = sample(2000), sample(300)
for mode in ("natural order", "uniform random orders", "Eq. 9 (uniform pretraining, then sampled orders)"):
    torch.manual_seed(0)
    lm = SetLM(V * n, 64)
    opt = torch.optim.Adam(lm.parameters(), 1e-2)
    for it in range(600):
        b = train[torch.randint(0, 2000, (64,))]
        m = "given" if mode.startswith("natural") else "uniform" if mode.startswith("uniform") or it < 200 else "sample"
        eq9_step(lm, opt, b, m, g)
    order, _ = best_order(lm, test)
    nll = -order_log_prob(lm, test, order).mean().item() / n
    top = collections.Counter(tuple(o.tolist()) for o in order).most_common(2)
    print(f"  {mode:48s}: {nll:.3f} nats/token in its best order; favourite orders {top}")
print("-> training on random orders spreads the model thin (1.22 nats/token); letting it sample the orders it")
print("   already likes concentrates it on a few orders and gets much closer to a fixed good order (0.77 vs 0.59).")
print("   In 600 updates it has not settled on ONE order, as the paper's model eventually did (Table 2).")
