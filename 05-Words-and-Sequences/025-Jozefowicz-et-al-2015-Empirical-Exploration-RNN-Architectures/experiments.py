"""Reproduce Jozefowicz, Zaremba & Sutskever (2015) at small scale.

  E1  Table 1: next-step-prediction accuracy of Tanh, LSTM, LSTM-f, LSTM-i, LSTM-o, LSTM-b, GRU, MUT1-3
      on Arithmetic, XML and Penn Treebank, each with a few random hyperparameter settings from Section 3.7.
      Paper (Arith / XML / PTB): LSTM .892/.425/.089, LSTM-b .902/.444/.090, GRU .896/.460/.091,
      MUT1 .921/.475/.090, LSTM-f .293/.234/.088, Tanh .295/.321/.088.
  E2  The forget-gate bias on its own: LSTM vs LSTM-b on Arithmetic and XML over several seeds.
  E3  A small architecture search (Section 3.1) with the mutation operators of Section 3.2, starting
      from the LSTM and the GRU graphs; fitness = Eq. (1). (The paper evaluated 10,000 architectures and
      230,000 hyperparameter settings.)

Training (Section 3.5-3.6): minibatch 20, unroll 35, hidden state carried across minibatches; learning
rate lowered by 2 each epoch after 3 epochs without validation improvement, for 4 more epochs.
Accuracy = fraction of correctly predicted next symbols (on the answer part for Arithmetic).

!! HEAVY. Not run on the author's laptop.
       python3 experiments.py --quick          # tiny budgets, ~20-40 minutes
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

from architectures import CELLS, GraphCell, SequenceModel, Search
from tasks import IDX, VOCAB, arithmetic, encode, memorization, xml

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data" / "ptb"
PTB_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.{}.txt"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def char_stream(task, n_examples, rng):
    """Concatenate examples into one stream; mask = 1 where the prediction is scored."""
    ids, mask = [], []
    for _ in range(n_examples):
        s, start = task(rng)
        e = encode(s)
        ids += e
        mask += [0] * start + [1] * (len(e) - start) if task is not xml else [1] * len(e)
    return torch.tensor(ids), torch.tensor(mask, dtype=torch.float32)


def load_ptb():
    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "valid"):
        p = DATA / f"ptb.{split}.txt"
        if not p.exists():
            urllib.request.urlretrieve(PTB_URL.format(split), p)
        out[split] = p.read_text().replace("\n", " <eos> ").split()
    vocab = {w: i for i, w in enumerate(sorted(set(out["train"])))}
    return {k: (torch.tensor([vocab[w] for w in v]), torch.ones(len(v))) for k, v in out.items()}, len(vocab)


def batchify(ids, mask, B=20):
    n = len(ids) // B
    return ids[:n * B].view(B, n).T.contiguous(), mask[:n * B].view(B, n).T.contiguous()


def run(model, data, lr=None, clip=5.0, steps=35):
    """One epoch of truncated BPTT; returns next-step accuracy on the masked positions."""
    ids, mask = data
    train = lr is not None
    model.train(train)
    states = model.init_state(ids.shape[1], DEV)
    right = total = 0.0
    for s in range(0, ids.shape[0] - 1, steps):
        x, y, m = ids[s:s + steps].to(DEV), ids[s + 1:s + 1 + steps].to(DEV), mask[s + 1:s + 1 + steps].to(DEV)
        x = x[:len(y)]
        states = [tuple(t.detach() for t in st) for st in states]
        with torch.set_grad_enabled(train):
            logits, states = model(x, states)
            loss = (F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none") * m.reshape(-1)).sum() / ids.shape[1]
        if train:
            model.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            with torch.no_grad():
                for p in model.parameters():
                    p -= lr * p.grad
        right += ((logits.argmax(-1) == y).float() * m).sum().item()
        total += m.sum().item()
    return right / max(total, 1)


def train_eval(cell_factory, task_data, n_vocab, hp, max_epochs):
    """Section 3.6's search schedule: after 3 epochs without improvement, halve the lr each epoch for 4 epochs."""
    torch.manual_seed(hp["seed"])
    model = SequenceModel(cell_factory, n_vocab, hp["units"], n_vocab, hp["layers"], init_scale=hp["scale"]).to(DEV)
    tr, va = task_data
    lr, best, bad, decaying = hp["lr"], 0.0, 0, None
    for ep in range(max_epochs):
        run(model, tr, lr, hp["clip"])
        acc = run(model, va)
        if acc > best:
            best, bad = acc, 0
        else:
            bad += 1
        if decaying is None and bad >= 3:
            decaying = 4
        if decaying is not None:
            lr /= 2; decaying -= 1
            if decaying == 0:
                break
    return best


def random_hp(rng, units):
    """Section 3.7's ranges (the parameter count is replaced by a fixed unit count here)."""
    return dict(scale=rng.choice([0.3, 0.7, 1, 1.4, 2, 2.8]), lr=rng.choice([0.1, 0.2, 0.3, 0.5, 1, 2, 5]),
                clip=rng.choice([1, 2.5, 5, 10, 20]), layers=rng.choice([1, 2]), units=units, seed=rng.randrange(10 ** 6))


def task_datasets(a):
    rng = random.Random(0)
    out = {}
    for name, fn in (("arithmetic", arithmetic), ("xml", xml)):
        tr = batchify(*char_stream(fn, a.examples, rng))
        va = batchify(*char_stream(fn, a.examples // 5, rng))
        out[name] = ((tr, va), len(VOCAB))
    if not a.skip_ptb:
        ptb, V = load_ptb()
        out["ptb"] = ((batchify(*ptb["train"]), batchify(*ptb["valid"])), V)
    return out


def e1(a):
    data = task_datasets(a)
    rng = random.Random(1)
    out = {}
    for arch in CELLS:
        out[arch] = {}
        for task, (td, V) in data.items():
            out[arch][task] = max(train_eval(CELLS[arch], td, V, random_hp(rng, a.units), a.epochs) for _ in range(a.settings))
        print(f"  E1 {arch}: {out[arch]}", flush=True)
    return out


def e2(a):
    data = task_datasets(a)
    out = {}
    for arch in ("LSTM", "LSTM-b"):
        out[arch] = {task: [train_eval(CELLS[arch], td, V, dict(scale=1.0, lr=1.0, clip=5, layers=1, units=a.units, seed=s), a.epochs)
                            for s in range(a.seeds)] for task, (td, V) in data.items() if task != "ptb"}
        print(f"  E2 {arch}: {out[arch]}", flush=True)
    return out


def e3(a):
    data = {k: v for k, v in task_datasets(a).items() if k != "ptb"}
    rng = random.Random(2)
    gru_acc = {t: train_eval(CELLS["GRU"], td, V, dict(scale=1.0, lr=1.0, clip=5, layers=1, units=a.units, seed=0), a.epochs)
               for t, (td, V) in data.items()}
    memo = batchify(*char_stream(memorization, a.examples, random.Random(3)))
    memo_va = batchify(*char_stream(memorization, a.examples // 5, random.Random(4)))

    def evaluate(graph, task, k):
        td, V = data[task]
        return max(train_eval(lambda n: GraphCell(graph, n), td, V, random_hp(rng, a.units), a.epochs)
                   for _ in range(max(1, k // a.search_divisor)))

    def passes_memo(graph):
        return train_eval(lambda n: GraphCell(graph, n), (memo, memo_va), len(VOCAB),
                          random_hp(rng, a.units), a.epochs) >= 0.95

    search = Search(evaluate, passes_memo, list(data), gru_acc, rng=rng)
    log = [search.step() for _ in range(a.search_steps)]
    best = search.pool[0]
    return {"GRU accuracy": gru_acc, "outcomes": {k: log.count(k) for k in set(log)},
            "best fitness": search.score(best), "best graph": best["graph"].nodes, "best name": best["name"]}


EXPS = {"e1": e1, "e2": e2, "e3": e3}


def report(R, a):
    L = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    if R.get("e1"):
        tasks = list(next(iter(R["e1"].values())))
        L += ["## E1: Table 1 (best next-step accuracy)", "", "| arch | " + " | ".join(tasks) + " |", "|---" * (len(tasks) + 1) + "|"]
        L += [f"| {k} | " + " | ".join(f"{v[t]:.4f}" for t in tasks) + " |" for k, v in R["e1"].items()] + [""]
    if R.get("e2"):
        L += ["## E2: forget-gate bias", ""] + [f"- {k}: {v}" for k, v in R["e2"].items()] + [""]
    if R.get("e3"):
        L += ["## E3: architecture search", "", f"- outcomes: {R['e3']['outcomes']}", f"- best fitness (vs GRU): {R['e3']['best fitness']:.3f}"]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=list(EXPS))
    ap.add_argument("--units", type=int, default=128)
    ap.add_argument("--examples", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--settings", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--search-steps", type=int, default=200)
    ap.add_argument("--search-divisor", type=int, default=10, help="evaluate k/divisor settings instead of the paper's 20")
    ap.add_argument("--skip-ptb", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.units, a.examples, a.epochs, a.settings, a.seeds, a.search_steps, a.skip_ptb = 64, 2000, 5, 1, 2, 10, True
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in EXPS.items():
            if a.only in (None, name):
                print(name, flush=True)
                R[name] = fn(a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
