"""Reproduce Show and Tell (Vinyals et al. 2015) on Flickr8k.

(The paper: a BatchNorm-Inception CNN pretrained on ImageNet and kept fixed, LSTM and embeddings of 512, words
seen >= 5 times, SGD with a fixed learning rate and no momentum, dropout + ensembles, beam 20.)

  E1  Table 2 (Flickr8k column) + Table 1 metrics: BLEU-1 (paper NIC 63, human 70), BLEU-4, CIDEr-D on the
      1000 test images, beam 20 vs greedy (paper: 'a beam size of 1 ... did degrade our results by 2 BLEU').
  E2  Table 4: ranking on Flickr8k test (1000 images x 5000 captions) by log p(S|I): image annotation and image
      search R@1 / R@10 / median rank. Paper NIC: annotation 20 / 61 / 6, search 19 / 64 / 5.
  E3  Sec. 3.1 ablation: image fed only at t = -1 (the paper's choice) vs fed at every step ('inferior results
      ... overfits more easily'). Dev perplexity and test BLEU of both.
  E4  Sec. 4.3.4 diversity: the 15-best list of beam 20. Fraction of best captions found verbatim in the training
      set (paper: 80%), fraction of novel captions in the 15-best (paper: about half), and BLEU agreement among
      the 15 (paper: 58, 'similar to that of humans').
  E5  Table 6: nearest neighbours of 'car', 'boy', 'street', 'horse', 'computer' in the learned W_e.

Image features: torchvision's ImageNet GoogLeNet (1024-d pooled features), the closest pretrained model to the
paper's BN-Inception, computed once and cached. Learning rate and clipping are not given in the paper; we use the
values of the authors' later open-source release (im2txt): SGD lr 2.0, gradient norm clipped at 5.

!! HEAVY (a 1 GB download, CNN features for 8000 images, LSTM training). Not run on the author's laptop.
       python3 experiments.py --quick          # 3 epochs, smaller model, ~20-40 minutes on CPU
       python3 experiments.py --only e2
       python3 experiments.py --report-only
"""

import argparse
import json
import random
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import torch

from nic import (NIC, beam_search, bleu, cider_d, human_bleu, nearest_words, normalize_for_annotation, novelty,
                 pad_captions, ranks, recall_report, score_matrix)

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data" / "flickr8k"
URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/{}"
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------------------------------------------- data
def download():
    DATA.mkdir(parents=True, exist_ok=True)
    for name in ("Flickr8k_text.zip", "Flickr8k_Dataset.zip"):
        z = DATA / name
        if not z.exists():
            print("downloading", name, flush=True)
            urllib.request.urlretrieve(URL.format(name), z)
            zipfile.ZipFile(z).extractall(DATA)


def tokenize(s):
    return "".join(ch if ch.isalnum() or ch == " " else " " for ch in s.lower()).split()


def load_captions():
    caps = defaultdict(list)
    for line in (DATA / "Flickr8k.token.txt").read_text().splitlines():
        img, text = line.split("\t")
        caps[img.split("#")[0]].append(tokenize(text))
    splits = {k: [l.strip() for l in (DATA / f"Flickr_8k.{f}Images.txt").read_text().splitlines() if l.strip()]
              for k, f in (("train", "train"), ("dev", "dev"), ("test", "test"))}
    return caps, splits


@torch.no_grad()
def features(names):
    """Fixed CNN features (the paper does not fine-tune the CNN). Cached to disk."""
    path = DATA / "feats_googlenet.pt"
    cache = torch.load(path) if path.exists() else {}
    todo = [n for n in names if n not in cache]
    if todo:
        import torchvision
        from PIL import Image
        w = torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1
        cnn = torchvision.models.googlenet(weights=w).eval().to(DEV)
        cnn.fc = torch.nn.Identity()
        tf = w.transforms()
        folder = next(p for p in (DATA / "Flicker8k_Dataset", DATA / "Flickr8k_Dataset") if p.exists())
        for i in range(0, len(todo), 64):
            chunk = todo[i:i + 64]
            x = torch.stack([tf(Image.open(folder / n).convert("RGB")) for n in chunk]).to(DEV)
            for n, f in zip(chunk, cnn(x).cpu()):
                cache[n] = f
            print(f"  features {i + len(chunk)}/{len(todo)}", flush=True)
        torch.save(cache, path)
    return torch.stack([cache[n] for n in names])


class Vocab:
    def __init__(self, sents, min_count=5):
        c = Counter(w for s in sents for w in s)
        self.itos = ["<pad>", "<s>", "</s>", "<unk>"] + sorted(w for w, k in c.items() if k >= min_count)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, s):
        return [self.stoi.get(w, 3) for w in s]


def setup(a):
    download()
    caps, splits = load_captions()
    if a.quick:
        splits["train"] = splits["train"][:2000]
        splits["test"] = splits["test"][:200]
    vocab = Vocab([c for n in splits["train"] for c in caps[n]])
    D = {}
    for k, names in splits.items():
        D[k] = {"names": names, "feats": features(names),
                "caps": [[vocab.encode(c) for c in caps[n]] for n in names]}
    return D, vocab


# ---------------------------------------------------------------------------------------------------- training
def perplexity(m, split, a):
    m.eval()
    nll = n = 0
    with torch.no_grad():
        for i in range(0, len(split["names"]), 100):
            f = split["feats"][i:i + 100]
            for j in range(5):
                c = [cs[j] for cs in split["caps"][i:i + 100]]
                nll -= m.log_prob(f.to(DEV), pad_captions(c).to(DEV)).sum().item()
                n += sum(len(x) + 1 for x in c)
    return float(torch.exp(torch.tensor(nll / n)))


def train(D, vocab, a, every=False, tag="nic"):
    torch.manual_seed(0)
    rng = random.Random(0)
    m = NIC(len(vocab.itos), D["train"]["feats"].shape[1], d=a.d, dropout=a.dropout, image_every_step=every).to(DEV)
    opt = torch.optim.SGD(m.parameters(), lr=a.lr)                              # fixed lr, no momentum
    pairs = [(i, j) for i in range(len(D["train"]["names"])) for j in range(len(D["train"]["caps"][i]))]
    best, path = float("inf"), HERE / f"{tag}.pt"
    for ep in range(a.epochs):
        m.train()
        rng.shuffle(pairs)
        for b in range(0, len(pairs), a.batch):
            mb = pairs[b:b + a.batch]
            f = D["train"]["feats"][[i for i, _ in mb]].to(DEV)
            c = pad_captions([D["train"]["caps"][i][j] for i, j in mb]).to(DEV)
            loss = -m.log_prob(f, c).sum() / (c[1:] != 0).sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        ppl = perplexity(m, D["dev"], a)                                         # model selection: dev perplexity
        print(f"    {tag} epoch {ep + 1}: train loss {loss.item():.3f}, dev perplexity {ppl:.2f}", flush=True)
        if ppl < best:
            best = ppl
            torch.save(m.state_dict(), path)
    m.load_state_dict(torch.load(path, map_location=DEV))
    return m.eval().cpu(), best


def generate(m, split, beam):
    return [beam_search(m, f, beam=beam, max_len=20)[0][1] for f in split["feats"]]


# ---------------------------------------------------------------------------------------------------- experiments
def e1(D, vocab, a):
    m, ppl = train(D, vocab, a)
    refs = D["test"]["caps"]
    out = {"dev perplexity": ppl, "human BLEU-1": human_bleu(refs, 1), "human BLEU-4": human_bleu(refs)}
    for beam in (a.beam, 1):
        hyps = generate(m, D["test"], beam)
        out[f"beam {beam}"] = {"BLEU-1": bleu(hyps, refs, 1), "BLEU-4": bleu(hyps, refs), "CIDEr-D": cider_d(hyps, refs),
                               "examples": [" ".join(vocab.itos[w] for w in h) for h in hyps[:10]]}
        print(f"  E1 beam {beam}: {out[f'beam {beam}']['BLEU-1']:.1f} BLEU-1", flush=True)
    return out


def e2(D, vocab, a):
    m = NIC(len(vocab.itos), D["train"]["feats"].shape[1], d=a.d)
    m.load_state_dict(torch.load(HERE / "nic.pt", map_location="cpu"))
    m = m.eval().to(DEV)
    caps, gt = [], []
    for i, cs in enumerate(D["test"]["caps"]):
        caps += cs; gt += [i] * len(cs)
    rows = []
    for b in range(0, len(caps), 1000):                                          # blocks of captions keep memory bounded
        rows.append(score_matrix(m, D["test"]["feats"].to(DEV), pad_captions(caps[b:b + 1000]).to(DEV)).cpu())
    S = torch.cat(rows, 1)
    ann, _ = ranks(normalize_for_annotation(S), gt)
    _, search = ranks(S, gt)
    return {"annotation": recall_report(ann, (1, 5, 10)), "search": recall_report(search, (1, 5, 10))}


def e3(D, vocab, a):
    out = {}
    for every in (False, True):
        m, ppl = train(D, vocab, a, every=every, tag="every" if every else "nic")
        hyps = generate(m, D["test"], a.beam)
        out["every step" if every else "once (t = -1)"] = {"dev perplexity": ppl, "BLEU-4": bleu(hyps, D["test"]["caps"])}
    return out


def e4(D, vocab, a):
    m = NIC(len(vocab.itos), D["train"]["feats"].shape[1], d=a.d)
    m.load_state_dict(torch.load(HERE / "nic.pt", map_location="cpu"))
    m.eval()
    train_set = {tuple(c) for cs in D["train"]["caps"] for c in cs}
    best, top, nbests = [], [], []
    for f in D["test"]["feats"][:a.n_div]:
        nb = [t for _, t in beam_search(m, f, beam=20, max_len=20)[:15]]
        best.append(nb[0]); top += nb; nbests.append(nb)
    k = min(map(len, nbests))
    return {"best caption in training set (%)": 100 * (1 - novelty(best, train_set)),
            "novel among the 15-best (%)": 100 * novelty(top, train_set),
            "BLEU-4 agreement among the n-best": human_bleu([nb[:k] for nb in nbests]),
            "n-best size used": k}


def e5(D, vocab, a):
    m = NIC(len(vocab.itos), D["train"]["feats"].shape[1], d=a.d)
    m.load_state_dict(torch.load(HERE / "nic.pt", map_location="cpu"))
    return {w: [vocab.itos[j] for j in nearest_words(m, vocab.stoi[w], 5)]
            for w in ("car", "boy", "street", "horse", "computer") if w in vocab.stoi}


def report(R, a):
    L = ["# Results", "", f"Flickr8k, d = {a.d}" + (" (QUICK run)" if a.quick else ""), ""]
    if R.get("e1"):
        L += ["## E1: generation (paper Flickr8k: NIC BLEU-1 63, human 70)", "", "```", json.dumps(R["e1"], indent=1), "```", ""]
    if R.get("e2"):
        L += ["## E2: ranking (paper NIC: annotation R@1 20, R@10 61, Med r 6; search 19, 64, 5)", "",
              "```", json.dumps(R["e2"], indent=1), "```", ""]
    if R.get("e3"):
        L += ["## E3: image once vs every step", "", "```", json.dumps(R["e3"], indent=1), "```", ""]
    if R.get("e4"):
        L += ["## E4: diversity (paper: best in training set 80%, ~half of 15-best novel, agreement 58)", "",
              "```", json.dumps(R["e4"], indent=1), "```", ""]
    if R.get("e5"):
        L += ["## E5: embedding neighbours", ""] + [f"- {k}: {', '.join(v)}" for k, v in R["e5"].items()]
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["e1", "e2", "e3", "e4", "e5"])
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--n-div", type=int, default=200)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.d, a.epochs, a.beam, a.n_div = 256, 3, 5, 50
    path = HERE / "results.json"
    R = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        D, vocab = setup(a)
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
            if a.only in (None, name):
                R[name] = fn(D, vocab, a)
                path.write_text(json.dumps(R))
        print(f"done in {time.time() - t0:.0f}s")
    report(R, a)


if __name__ == "__main__":
    main()
