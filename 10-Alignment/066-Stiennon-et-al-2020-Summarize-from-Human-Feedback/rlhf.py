"""Learning to summarize from human feedback (Stiennon, Ouyang, Wu, Ziegler, Lowe, Voss, Radford, Amodei &
Christiano, NeurIPS 2020).

  Data: Reddit TL;DR (filtered, ~123k posts with the posters' own TL;DRs as REFERENCES); 64,832 human comparisons.
  Recipe (Figure 2):
    1. SFT: fine-tune a GPT-style model (1.3B / 6.7B) on the reference summaries.
    2. Collect comparisons: labelers pick the better of two summaries of the same post (policies vs baselines);
       labeler-researcher agreement 77% +- 2% (researcher-researcher 73%).
    3. Reward model r_theta(x, y): SFT model + scalar head; loss = -E[ log sigmoid( r(x, y_i) - r(x, y_{1-i}) ) ];
       outputs normalised so the reference summaries score 0 on average.
    4. RL: PPO on the reward  R(x, y) = r_theta(x, y) - beta * log[ pi_RL(y|x) / pi_SFT(y|x) ]  (KL to the SFT model:
       an entropy bonus that also stops the policy from drifting to outputs the RM has never seen). Each token is a
       time step; the value function is a SEPARATE network initialised from the reward model.
  Results: 1.3B human-feedback model preferred to references 61% vs 43% for a 6.7B supervised model; 6.7B HF ~65%
    after controlling for length. Transfers to CNN/DM without news training. Optimising the RM too hard (small beta)
    eventually makes summaries WORSE than predicted (Figure 5). RM accuracy: +1.1% per doubling of data, +1.8% per
    doubling of size. RMs are length-biased (prefer shortening edits 62.6% vs 76.4% for humans).

This file implements the whole pipeline at toy scale with real networks: a 'summarisation' task with a known true
quality (the labeler), SFT, pairwise comparisons, a Bradley-Terry reward model, and PPO with per-token KL penalty,
clipped ratios, GAE and a separate value network.
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_TOPICS, N_FILLER, POST_LEN, MAX_SUM = 16, 8, 14, 6
SPECIAL = ["PAD", "POST", "SUM", "EOS"]
VOCAB = SPECIAL + [f"T{i}" for i in range(N_TOPICS)] + [f"F{i}" for i in range(N_FILLER)]
IDX = {t: i for i, t in enumerate(VOCAB)}
CTX = POST_LEN + MAX_SUM + 3


# ---------------------------------------------------------------------------------------------------- the task
def make_post(rng):
    """A 'post': three topics with counts 4, 3, 2 plus filler words, shuffled. A good summary names the two most
    frequent topics, in order, and nothing else."""
    a, b, c = rng.sample(range(N_TOPICS), 3)
    words = [f"T{a}"] * 4 + [f"T{b}"] * 3 + [f"T{c}"] * 2
    words += [f"F{rng.randrange(N_FILLER)}" for _ in range(POST_LEN - len(words))]
    rng.shuffle(words)
    return words, (f"T{a}", f"T{b}", f"T{c}")


def reference_summary(top, rng):
    """The posters' own TL;DRs: decent but noisy. Half name only the main topic; others add the right or the wrong
    second topic, or a filler word. Imitating them (greedy SFT) therefore usually stops after one topic."""
    u = rng.random()
    if u < 0.5:
        s = [top[0]]
    elif u < 0.7:
        s = [top[0], top[1]]
    elif u < 0.85:
        s = [top[0], top[2]]
    else:
        s = [top[0], top[1], f"F{rng.randrange(N_FILLER)}"]
    return s


def true_quality(post, top, summary):
    """The simulated labelers' utility: +1 for each of the two main topics, +0.3 for the right order, -1 per token
    not in the post (hallucination), -0.5 per filler word, -0.6 per repeated token, -0.4 per token beyond 3."""
    q = 1.0 * (top[0] in summary) + 1.0 * (top[1] in summary)
    if top[0] in summary and top[1] in summary and summary.index(top[0]) < summary.index(top[1]):
        q += 0.3
    q -= 1.0 * sum(w not in post for w in summary)
    q -= 0.5 * sum(w.startswith("F") for w in summary)
    q -= 0.6 * (len(summary) - len(set(summary)))
    q -= 0.4 * max(0, len(summary) - 3)
    return q


def labeler_prefers(qa, qb, rng, noise=0.5):
    """Noisy human comparison: P(A preferred) = sigmoid((qa - qb) / noise)."""
    return float(rng.random() < 1 / (1 + math.exp(-(qa - qb) / noise)))


def encode(post, summary=None):
    ids = [IDX["POST"]] + [IDX[w] for w in post] + [IDX["SUM"]]
    if summary is not None:
        ids += [IDX[w] for w in summary] + [IDX["EOS"]]
    return ids


# ---------------------------------------------------------------------------------------------------- models
def _llama():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "08-Pretraining-and-Scaling-LLMs" / "056-Touvron-et-al-2023-LLaMA" / "llama.py"
    spec = importlib.util.spec_from_file_location("llama_056", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class LM(nn.Module):
    """A tiny LLaMA-style decoder (paper 056) with an optional scalar head (reward model / value function)."""

    def __init__(self, d=64, layers=2, scalar=False):
        super().__init__()
        L = _llama()
        self.body = L.LLaMA(vocab=len(VOCAB), d=d, layers=layers, heads=4, ctx=CTX, hidden=L.ffn_hidden(d, 16))
        self.head = nn.Linear(d, 1) if scalar else None

    def hidden(self, ids):
        b = self.body
        h = b.tok(ids)
        for blk in b.blocks:
            h = blk(h, b.cos, b.sin)
        return b.norm(h)

    def forward(self, ids):
        h = self.hidden(ids)
        return self.body.out(h) if self.head is None else self.head(h).squeeze(-1)


def pad(seqs):
    L = max(map(len, seqs))
    return torch.tensor([s + [IDX["PAD"]] * (L - len(s)) for s in seqs])


# ---------------------------------------------------------------------------------------------------- 1. SFT
def train_sft(steps=600, seed=0, lr=3e-3, batch=64, d=64):
    rng = random.Random(seed); torch.manual_seed(seed)
    m = LM(d=d)
    opt = torch.optim.AdamW(m.parameters(), lr)
    for _ in range(steps):
        seqs, masks = [], []
        for _ in range(batch):
            post, top = make_post(rng)
            s = encode(post, reference_summary(top, rng))
            seqs.append(s); masks.append([0] * (POST_LEN + 2) + [1] * (len(s) - POST_LEN - 2))
        X = pad(seqs); M = pad(masks).float()[:, 1:]
        nll = F.cross_entropy(m(X)[:, :-1].reshape(-1, len(VOCAB)), X[:, 1:].reshape(-1), reduction="none").view(M.shape)
        loss = (nll * M).sum() / M.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def sample(policy, posts, temperature=1.0, gen=None):
    """Sample summaries for a batch of posts. Returns token lists (without EOS) and the full id sequences."""
    X = torch.tensor([encode(p) for p in posts])
    done = torch.zeros(len(posts), dtype=torch.bool)
    for _ in range(MAX_SUM + 1):
        lg = policy(X)[:, -1] / max(temperature, 1e-6)
        lg[:, :IDX["EOS"]] = -1e9                                               # never PAD/POST/SUM
        nxt = lg.argmax(-1) if temperature == 0 else torch.multinomial(F.softmax(lg, -1), 1, generator=gen).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, IDX["PAD"]), nxt)
        X = torch.cat([X, nxt[:, None]], 1)
        done |= nxt == IDX["EOS"]
        if done.all():
            break
    out = []
    for row in X[:, POST_LEN + 2:].tolist():
        toks = []
        for t in row:
            if t in (IDX["EOS"], IDX["PAD"]):
                break
            toks.append(VOCAB[t])
        out.append(toks[:MAX_SUM])
    return out


def logprobs(model, seqs, start=POST_LEN + 2):
    """Per-token log-probabilities of the summary tokens (incl. EOS) for a list of id sequences."""
    X = pad(seqs)
    lp = F.log_softmax(model(X)[:, :-1], -1).gather(2, X[:, 1:, None]).squeeze(-1)
    mask = torch.zeros_like(lp)
    for i, s in enumerate(seqs):
        mask[i, start - 1:len(s) - 1] = 1
    return lp, mask


# ---------------------------------------------------------------------------------------------------- 2-3. RM
def collect_comparisons(policy, n_pairs, seed=0, temperature=1.0):
    """Pairs of SFT samples for the same post (plus some references), labelled by the simulated labelers."""
    rng = random.Random(seed); gen = torch.Generator().manual_seed(seed)
    posts = [make_post(rng) for _ in range(n_pairs)]
    a = sample(policy, [p for p, _ in posts], temperature, gen)
    b = sample(policy, [p for p, _ in posts], temperature, gen)
    data = []
    for (post, top), sa, sb in zip(posts, a, b):
        if rng.random() < 0.25:
            sb = reference_summary(top, rng)
        y = labeler_prefers(true_quality(post, top, sa), true_quality(post, top, sb), rng)
        data.append((post, sa, sb, y))
    return data


def train_reward_model(sft, data, steps=600, lr=1e-3, batch=64, seed=0):
    """SFT model + scalar head read at the final token; loss -log sigmoid(r_chosen - r_rejected) (soft labels)."""
    torch.manual_seed(seed)
    rm = LM(d=sft.body.tok.embedding_dim, scalar=True)
    rm.body.load_state_dict(sft.body.state_dict())
    opt = torch.optim.AdamW(rm.parameters(), lr)
    rng = random.Random(seed)
    for _ in range(steps):
        b = rng.sample(data, min(batch, len(data)))
        ra = reward(rm, [x[0] for x in b], [x[1] for x in b])
        rb = reward(rm, [x[0] for x in b], [x[2] for x in b])
        y = torch.tensor([x[3] for x in b])
        loss = F.binary_cross_entropy_with_logits(ra - rb, y)
        opt.zero_grad(); loss.backward(); opt.step()
    rm.eval()
    rng2 = random.Random(seed + 7)                                               # normalise: references score 0
    with torch.no_grad():
        refs = [make_post(rng2) for _ in range(256)]
        rm.bias = float(reward(rm, [p for p, _ in refs], [reference_summary(t, rng2) for _, t in refs]).mean())
    return rm


def reward(rm, posts, summaries):
    seqs = [encode(p, s) for p, s in zip(posts, summaries)]
    out = rm(pad(seqs))
    r = out[torch.arange(len(seqs)), torch.tensor([len(s) - 1 for s in seqs])]
    return r - getattr(rm, "bias", 0.0)


def rm_accuracy(rm, data):
    with torch.no_grad():
        ra = reward(rm, [x[0] for x in data], [x[1] for x in data])
        rb = reward(rm, [x[0] for x in data], [x[2] for x in data])
    y = torch.tensor([x[3] for x in data])
    keep = y != 0.5
    return float(((ra > rb).float() == y)[keep].float().mean())


# ---------------------------------------------------------------------------------------------------- 4. PPO
def ppo(sft, rm, beta, iters=150, batch=64, epochs=4, lr=1e-4, clip=0.2, gamma=1.0, lam=0.95, seed=0, log_every=0,
        reward_fn=None):
    """PPO with the paper's reward: at each summary token, -beta * (log pi - log pi_SFT); at the last token,
    + r_RM(x, y). Value function: a separate network initialised from the reward model. Returns the policy and a
    history of (mean RM score, mean true quality, mean KL). reward_fn(posts, tops, summaries), if given, replaces the
    reward model (e.g. a ROUGE-like metric); the value network is still initialised from `rm`."""
    import copy
    torch.manual_seed(seed); rng = random.Random(seed); gen = torch.Generator().manual_seed(seed)
    policy = copy.deepcopy(sft).train()
    value = copy.deepcopy(rm).train()
    opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr)
    history = []
    for it in range(iters):
        posts = [make_post(rng) for _ in range(batch)]
        summaries = sample(policy, [p for p, _ in posts], 1.0, gen)
        seqs = [encode(p, s) for (p, _), s in zip(posts, summaries)]
        with torch.no_grad():
            lp_old, mask = logprobs(policy, seqs)
            lp_ref, _ = logprobs(sft, seqs)
            r = reward_fn([p for p, _ in posts], [t for _, t in posts], summaries) if reward_fn else \
                reward(rm, [p for p, _ in posts], summaries)
            v_old = value(pad(seqs))[:, :-1]
            kl = (lp_old - lp_ref) * mask
            rewards = -beta * kl
            last = mask.cumsum(1).argmax(1)
            rewards[torch.arange(batch), last] += r
            adv = torch.zeros_like(rewards); gae = torch.zeros(batch)
            for t in reversed(range(rewards.shape[1])):                           # GAE over summary tokens
                nv = v_old[:, t + 1] * mask[:, t + 1] if t + 1 < rewards.shape[1] else torch.zeros(batch)
                delta = (rewards[:, t] + gamma * nv - v_old[:, t]) * mask[:, t]
                gae = delta + gamma * lam * gae * mask[:, t]
                adv[:, t] = gae
            ret = adv + v_old
            a_n = (adv - adv[mask > 0].mean()) / (adv[mask > 0].std() + 1e-8)
        for _ in range(epochs):
            lp, _ = logprobs(policy, seqs)
            ratio = torch.exp(lp - lp_old)
            pg = -torch.min(ratio * a_n, ratio.clamp(1 - clip, 1 + clip) * a_n)
            v = value(pad(seqs))[:, :-1]
            loss = ((pg + 0.5 * (v - ret) ** 2) * mask).sum() / mask.sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        q = np.mean([true_quality(p, t, s) for (p, t), s in zip(posts, summaries)])
        history.append((float(r.mean()), float(q), float(kl.sum(1).mean())))
        if log_every and it % log_every == 0:
            print(f"    iter {it}: RM {history[-1][0]:.2f}  true {q:.2f}  KL {history[-1][2]:.2f}", flush=True)
    return policy.eval(), history


# ---------------------------------------------------------------------------------------------------- evaluation
def evaluate_policy(policy, rm, n=400, seed=123, temperature=0.0):
    """Mean true quality, mean RM score, and how often labelers would prefer the policy's summary to the
    reference summary (ties count half)."""
    rng = random.Random(seed); gen = torch.Generator().manual_seed(seed)
    posts = [make_post(rng) for _ in range(n)]
    sums = sample(policy, [p for p, _ in posts], temperature, gen)
    q = [true_quality(p, t, s) for (p, t), s in zip(posts, sums)]
    refs = [reference_summary(t, rng) for _, t in posts]
    qr = [true_quality(p, t, s) for (p, t), s in zip(posts, refs)]
    pref = np.mean([1 / (1 + math.exp(-(a - b) / 0.5)) for a, b in zip(q, qr)])
    with torch.no_grad():
        r = float(reward(rm, [p for p, _ in posts], sums).mean())
    return {"true quality": float(np.mean(q)), "RM score": r, "preferred to reference": float(pref),
            "mean length": float(np.mean([len(s) for s in sums])), "example": (posts[0][1][:2], sums[0])}


REPORTED = {"1.3B human feedback vs references": 0.61, "6.7B supervised vs references": 0.43,
            "6.7B human feedback vs references (length-controlled)": 0.65, "labeler-researcher agreement": 0.77,
            "RM accuracy per doubling of data": "+1.1%", "RM accuracy per doubling of size": "+1.8%",
            "comparisons collected": 64832}
