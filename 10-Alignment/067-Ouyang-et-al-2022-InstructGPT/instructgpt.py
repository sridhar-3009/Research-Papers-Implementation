"""Training language models to follow instructions with human feedback (InstructGPT; Ouyang, Wu, Jiang, Almeida,
Wainwright, Mishkin, Zhang, Agarwal, Slama, Ray, Schulman, Hilton, Kelton, Miller, Simens, Askell, Welinder,
Christiano, Leike & Lowe, NeurIPS 2022).

  Problem: GPT-3 continues text; it does not reliably DO what the prompt asks ('misaligned' with the user's intent).
  Three steps (Figure 2), on prompts from the OpenAI API plus labeler-written ones; ~40 contractors:
    1. SFT on labeler demonstrations (~13k prompts), 16 epochs.
    2. Reward model (6B) on RANKINGS: labelers rank K = 4..9 responses -> C(K, 2) comparisons per prompt (33k prompts).
       All C(K, 2) pairs of one prompt form ONE batch element (shuffling them separately overfits):
         loss = -1/C(K,2) * E[ log sigmoid( r(x, y_w) - r(x, y_l) ) ];  RM normalised so demonstrations score 0.
    3. PPO (31k API prompts) with a per-token KL penalty to SFT and a value function initialised from the RM;
       PPO-ptx also mixes pretraining gradients:
         objective = E[ r(x, y) - beta * log(pi_RL(y|x) / pi_SFT(y|x)) ] + gamma * E_pretrain[ log pi_RL(x) ]
       to reduce the 'alignment tax' (regressions on public NLP benchmarks such as SQuAD, DROP, HellaSwag).
  Results: 1.3B InstructGPT preferred to 175B GPT-3 (100x fewer parameters); 175B InstructGPT preferred to GPT-3
    85 +- 3% and to few-shot GPT-3 71 +- 4%; TruthfulQA truthful+informative ~2x; closed-domain hallucination 21% vs
    41%; ~25% fewer toxic outputs when asked to be respectful; little change in bias; held-out labelers also
    prefer InstructGPT; FLAN / T0 fine-tuning is worse on the API distribution. Agreement: training labelers 72.6%,
    held-out 77.3%.

Toy version: a base LM pre-trained on 'web text' in which instruction-like text is NOT followed by the correct
answer; instruction tasks (sort / first two / last two of a few letters: easy to CHECK, harder to produce); SFT on a few demonstrations; an RM from
K-way rankings; PPO and PPO-ptx; the alignment tax measured on the pre-training distribution.
"""

import copy
import itertools
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LETTERS = list("abcdefgh")
TASKS = ["SORT", "FIRST2", "LAST2"]
VOCAB = ["PAD", "BOS", "SEP", "EOS"] + TASKS + LETTERS
IDX = {t: i for i, t in enumerate(VOCAB)}
CTX = 24


# ---------------------------------------------------------------------------------------------------- the world
def make_instruction(rng):
    task = rng.choice(TASKS)
    args = [rng.choice(LETTERS) for _ in range(rng.randint(3, 5))]
    return task, args


def correct_response(task, args):
    return {"SORT": sorted(args), "FIRST2": args[:2], "LAST2": args[-2:]}[task]


def labeler_utility(task, args, response):
    """+2 x fraction of positions right, +1 for an exact answer, -0.3 per length mismatch."""
    gold = correct_response(task, args)
    pos = sum(a == b for a, b in zip(response, gold)) / len(gold)
    return 2 * pos + 1.0 * (response == gold) - 0.3 * abs(len(response) - len(gold))


def web_document(rng):
    """Pre-training text. Mostly periodic letter patterns ('abcabcab...', the 'public NLP benchmark' is predicting
    them); sometimes an instruction followed by unrelated text, never by the right answer."""
    if rng.random() < 0.8:
        period = [rng.choice(LETTERS) for _ in range(rng.randint(2, 4))]
        return ["BOS"] + [period[i % len(period)] for i in range(CTX - 2)]
    task, args = make_instruction(rng)
    return ["BOS", task] + args + ["SEP"] + [rng.choice(LETTERS) for _ in range(rng.randint(3, 8))] + ["EOS"]


def prompt_tokens(task, args):
    return ["BOS", task] + args + ["SEP"]


def ids(tokens):
    return [IDX[t] for t in tokens]


def pad(seqs, length=None):
    L = length or max(map(len, seqs))
    return torch.tensor([s[:L] + [IDX["PAD"]] * (L - len(s[:L])) for s in seqs])


# ---------------------------------------------------------------------------------------------------- models
def _llama():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "08-Pretraining-and-Scaling-LLMs" / "056-Touvron-et-al-2023-LLaMA" / "llama.py"
    spec = importlib.util.spec_from_file_location("llama_056", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class LM(nn.Module):
    def __init__(self, d=64, layers=2, scalar=False):
        super().__init__()
        L = _llama()
        self.body = L.LLaMA(vocab=len(VOCAB), d=d, layers=layers, heads=4, ctx=CTX, hidden=L.ffn_hidden(d, 16))
        self.head = nn.Linear(d, 1) if scalar else None

    def forward(self, x):
        b = self.body
        h = b.tok(x)
        for blk in b.blocks:
            h = blk(h, b.cos, b.sin)
        h = b.norm(h)
        return b.out(h) if self.head is None else self.head(h).squeeze(-1)


def lm_loss(model, seqs):
    X = pad(seqs)
    mask = (X[:, 1:] != IDX["PAD"]).float()
    nll = F.cross_entropy(model(X)[:, :-1].reshape(-1, len(VOCAB)), X[:, 1:].reshape(-1), reduction="none").view(mask.shape)
    return (nll * mask).sum() / mask.sum()


def pretrain(steps, d=64, seed=0, lr=3e-3, batch=64):
    rng = random.Random(seed); torch.manual_seed(seed)
    m = LM(d=d)
    opt = torch.optim.AdamW(m.parameters(), lr)
    for _ in range(steps):
        loss = lm_loss(m, [ids(web_document(rng)) for _ in range(batch)])
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def benchmark(model, n=200, seed=99):
    """The 'public NLP benchmark': next-token accuracy on held-out periodic documents (after the first period)."""
    rng = random.Random(seed)
    docs = []
    while len(docs) < n:
        d = web_document(rng)
        if "SEP" not in d:
            docs.append(ids(d))
    X = torch.tensor(docs)
    pred = model(X)[:, :-1].argmax(-1)
    return float((pred[:, 6:] == X[:, 7:]).float().mean())


def demonstration(task, args):
    return prompt_tokens(task, args) + correct_response(task, args) + ["EOS"]


def sft(base, steps, n_demos, seed=0, lr=1e-3, batch=32):
    """Fine-tune on a FIXED, small set of labeler demonstrations (loss on the response only)."""
    rng = random.Random(seed + 100)
    demos = [ids(demonstration(*make_instruction(rng))) for _ in range(n_demos)]
    m = copy.deepcopy(base).train()
    opt = torch.optim.AdamW(m.parameters(), lr)
    for s in range(steps):
        b = random.Random(s).sample(demos, min(batch, len(demos)))
        X = pad(b)
        mask = torch.zeros(X.shape[0], X.shape[1] - 1)
        for i, d in enumerate(b):
            sep = d.index(IDX["SEP"])
            mask[i, sep:len(d) - 1] = 1
        nll = F.cross_entropy(m(X)[:, :-1].reshape(-1, len(VOCAB)), X[:, 1:].reshape(-1), reduction="none").view(mask.shape)
        loss = (nll * mask).sum() / mask.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


@torch.no_grad()
def respond(model, prompts, temperature=1.0, gen=None, max_new=7):
    """Batched sampling; prompts are token lists (all the same length is NOT required: we left-pad nothing and
    generate per length group)."""
    out = [None] * len(prompts)
    groups = {}
    for i, p in enumerate(prompts):
        groups.setdefault(len(p), []).append(i)
    for ln, idx in groups.items():
        X = torch.tensor([ids(prompts[i]) for i in idx])
        done = torch.zeros(len(idx), dtype=torch.bool)
        for _ in range(max_new):
            lg = model(X)[:, -1]
            lg[:, :IDX["EOS"]] = -1e9
            lg[:, IDX["EOS"] + 1:IDX["EOS"] + 1 + len(TASKS)] = -1e9                  # no task tokens in answers
            nx = lg.argmax(-1) if temperature == 0 else torch.multinomial(F.softmax(lg / temperature, -1), 1, generator=gen).squeeze(1)
            nx = torch.where(done, torch.full_like(nx, IDX["PAD"]), nx)
            X = torch.cat([X, nx[:, None]], 1); done |= nx == IDX["EOS"]
            if done.all():
                break
        for j, i in enumerate(idx):
            toks = []
            for t in X[j, ln:].tolist():
                if t in (IDX["EOS"], IDX["PAD"]):
                    break
                toks.append(VOCAB[t])
            out[i] = toks
    return out


# ---------------------------------------------------------------------------------------------------- reward model
def collect_rankings(policy, n_prompts, K_range=(4, 9), seed=0, noise=0.3, temperature=1.5):
    """Labelers rank K responses (noisy utility); returns (prompt, responses sorted best-first)."""
    rng = random.Random(seed); gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_prompts):
        task, args = make_instruction(rng)
        K = rng.randint(*K_range)
        resps = respond(policy, [prompt_tokens(task, args)] * K, temperature, gen)
        noisy = [labeler_utility(task, args, r) + rng.gauss(0, noise) for r in resps]
        order = sorted(range(K), key=lambda i: -noisy[i])
        out.append(((task, args), [resps[i] for i in order]))
    return out


def ranking_loss(rm, prompt, ranked):
    """The paper's Eq. 1 for ONE prompt: mean over all C(K, 2) pairs of -log sigmoid(r_w - r_l); one forward pass
    per completion (K passes, not C(K, 2))."""
    seqs = [ids(prompt_tokens(*prompt) + r + ["EOS"]) for r in ranked]
    out = rm(pad(seqs))
    r = out[torch.arange(len(seqs)), torch.tensor([len(s) - 1 for s in seqs])]
    pairs = list(itertools.combinations(range(len(ranked)), 2))                   # (i, j): i ranked above j
    return -torch.stack([F.logsigmoid(r[i] - r[j]) for i, j in pairs]).mean()


def train_reward_model(sft_model, rankings, epochs=3, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    rm = LM(d=sft_model.body.tok.embedding_dim, scalar=True)
    rm.body.load_state_dict(sft_model.body.state_dict())
    opt = torch.optim.AdamW(rm.parameters(), lr)
    for _ in range(epochs):
        for i in range(0, len(rankings), 8):
            loss = torch.stack([ranking_loss(rm, p, r) for p, r in rankings[i:i + 8]]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    rm.eval()
    rng = random.Random(seed + 5)
    with torch.no_grad():                                                         # demonstrations score 0
        demos = [make_instruction(rng) for _ in range(200)]
        rm.bias = 0.0
        rm.bias = float(score(rm, demos, [correct_response(*d) for d in demos]).mean())
    return rm


def score(rm, prompts, responses):
    seqs = [ids(prompt_tokens(*p) + r + ["EOS"]) for p, r in zip(prompts, responses)]
    out = rm(pad(seqs))
    return out[torch.arange(len(seqs)), torch.tensor([len(s) - 1 for s in seqs])] - getattr(rm, "bias", 0.0)


# ---------------------------------------------------------------------------------------------------- PPO / PPO-ptx
def _logprobs(model, seqs, starts):
    X = pad(seqs)
    lp = F.log_softmax(model(X)[:, :-1], -1).gather(2, X[:, 1:, None]).squeeze(-1)
    mask = torch.zeros_like(lp)
    for i, (s, st) in enumerate(zip(seqs, starts)):
        mask[i, st - 1:len(s) - 1] = 1
    return lp, mask


def ppo(sft_model, rm, beta=0.1, gamma_ptx=0.0, iters=100, batch=64, epochs=4, lr=1e-4, clip=0.2, seed=0):
    """Bandit PPO (one prompt -> one response -> one RM reward) with per-token KL to SFT, a value network
    initialised from the RM, and optionally gamma * pretraining log-likelihood (PPO-ptx)."""
    torch.manual_seed(seed); rng = random.Random(seed); gen = torch.Generator().manual_seed(seed)
    policy, value = copy.deepcopy(sft_model).train(), copy.deepcopy(rm).train()
    opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr)
    hist = []
    for _ in range(iters):
        prompts = [make_instruction(rng) for _ in range(batch)]
        resps = respond(policy, [prompt_tokens(*p) for p in prompts], 1.0, gen)
        seqs = [ids(prompt_tokens(*p) + r + ["EOS"]) for p, r in zip(prompts, resps)]
        starts = [len(prompt_tokens(*p)) for p in prompts]
        with torch.no_grad():
            lp_old, mask = _logprobs(policy, seqs, starts)
            lp_ref, _ = _logprobs(sft_model, seqs, starts)
            r = score(rm, prompts, resps)
            kl = (lp_old - lp_ref) * mask
            rew = -beta * kl
            last = mask.cumsum(1).argmax(1)
            rew[torch.arange(batch), last] += r
            ret = rew.flip(1).cumsum(1).flip(1)                                   # undiscounted returns-to-go
            v_old = value(pad(seqs))[:, :-1]
            adv = (ret - v_old) * mask
            adv = (adv - adv[mask > 0].mean()) / (adv[mask > 0].std() + 1e-8)
        for _ in range(epochs):
            lp, _ = _logprobs(policy, seqs, starts)
            ratio = torch.exp(lp - lp_old)
            pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
            v = value(pad(seqs))[:, :-1]
            loss = ((pg + 0.5 * (v - ret) ** 2) * mask).sum() / mask.sum()
            if gamma_ptx:
                loss = loss + gamma_ptx * lm_loss(policy, [ids(web_document(rng)) for _ in range(32)])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        util = np.mean([labeler_utility(*p, r_) for p, r_ in zip(prompts, resps)])
        hist.append((float(r.mean()), float(util), float(kl.sum(1).mean())))
    return policy.eval(), hist


# ---------------------------------------------------------------------------------------------------- evaluation
def evaluate(model, n=300, seed=7, temperature=0.0):
    """Mean labeler utility, exact-match rate (greedy by default; temperature 1 = sampled outputs, as people see
    them in an API), and the pre-training benchmark."""
    rng = random.Random(seed)
    prompts = [make_instruction(rng) for _ in range(n)]
    resps = respond(model, [prompt_tokens(*p) for p in prompts], temperature, torch.Generator().manual_seed(seed))
    u = [labeler_utility(*p, r) for p, r in zip(prompts, resps)]
    exact = [r == correct_response(*p) for p, r in zip(prompts, resps)]
    return {"utility": float(np.mean(u)), "exact": float(np.mean(exact)), "benchmark": benchmark(model)}


def win_rate(model_a, model_b, n=300, seed=11, temp=0.5):
    """How often (simulated) labelers prefer A's response to B's on the same prompts: P = sigmoid((uA - uB)/temp)."""
    rng = random.Random(seed)
    prompts = [make_instruction(rng) for _ in range(n)]
    ra = respond(model_a, [prompt_tokens(*p) for p in prompts], 0.0)
    rb = respond(model_b, [prompt_tokens(*p) for p in prompts], 0.0)
    return float(np.mean([1 / (1 + math.exp(-(labeler_utility(*p, a) - labeler_utility(*p, b)) / temp))
                          for p, a, b in zip(prompts, ra, rb)]))


REPORTED = {"175B InstructGPT vs GPT-3": "85 +- 3%", "175B InstructGPT vs few-shot GPT-3": "71 +- 4%",
            "1.3B InstructGPT vs 175B GPT-3": "preferred (100x fewer parameters)",
            "closed-domain hallucination": "21% vs 41% (GPT-3)", "TruthfulQA truthful+informative": "~2x GPT-3",
            "toxic outputs when prompted to be respectful": "~25% fewer", "labeler agreement": "72.6% (training), 77.3% (held-out)",
            "data": "~13k SFT prompts, ~33k RM prompts, ~31k PPO prompts, ~40 labelers"}
