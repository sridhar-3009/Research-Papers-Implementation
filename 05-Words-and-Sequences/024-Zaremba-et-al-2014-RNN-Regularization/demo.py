"""A light tour of Zaremba, Sutskever & Vinyals (2014). A few seconds; no dataset.

    python3 demo.py
"""

import torch

from reg_lstm import DeepLSTM, count_params, dropouts_on_path

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. The paper's three Penn Treebank models (10,000-word vocabulary, 2 layers)")
for name, n, p in (("non-regularized", 200, 0.0), ("medium", 650, 0.5), ("large", 1500, 0.65)):
    print(f"{name:16s} {n:5d} units per layer, dropout {p:.2f}: {count_params(DeepLSTM(10000, n, 2)) / 1e6:5.1f}M parameters")
print("-> without regularization the paper's best model was the SMALL one: bigger ones overfit.")

line("2. Where dropout goes (Figure 2): count the corruptions on the path of one piece of information")
for k in (1, 10, 100, 1000):
    print(f"information from {k:4d} steps ago, 2 layers: non-recurrent dropout {dropouts_on_path(2, k, 'nonrecurrent')} times, "
          f"naive dropout {dropouts_on_path(2, k, 'naive')} times")
print("-> the paper's recipe corrupts a memory L + 1 = 3 times however long it is kept;")
print("   dropping the recurrent connections too destroys long memories.")

line("3. Check it inside the network: which inputs got zeroed?")
for mode in ("nonrecurrent", "naive"):
    m = DeepLSTM(50, 32, 2, dropout=0.5, mode=mode)
    m.train(); m.record = True
    m(torch.randint(0, 50, (10, 8)), m.init_state(8))
    vert = sum((v == 0).float().mean().item() for _, v, _, _ in m.trace) / len(m.trace)
    rec = sum((r != h).float().mean().item() for _, _, r, h in m.trace[4:]) / len(m.trace[4:])
    print(f"{mode:13s}: vertical inputs zeroed {100 * vert:4.1f}%,  recurrent inputs altered {100 * rec:5.1f}%")
print("('altered' = zeroed, or rescaled by 1/(1-p) as inverted dropout does to the kept units)")
