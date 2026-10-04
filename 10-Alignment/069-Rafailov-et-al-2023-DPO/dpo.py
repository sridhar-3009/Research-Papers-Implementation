"""Direct Preference Optimization (Rafailov et al. 2023): RLHF's KL-constrained objective solved WITHOUT a reward model
or RL, by a single classification loss on preference pairs.

  RLHF objective      max_pi  E[r(x, y)] - beta * KL(pi(.|x) || pi_ref(.|x))
  its optimum         pi*(y|x) = pi_ref(y|x) exp(r(x, y) / beta) / Z(x)                       (Eq. 4)
  so                  r(x, y) = beta log pi*(y|x)/pi_ref(y|x) + beta log Z(x)                   (Eq. 5)
  Bradley-Terry only needs reward DIFFERENCES, so Z(x) cancels:
  DPO loss            -log sigmoid( beta [log pi(yw|x)/pi_ref(yw|x) - log pi(yl|x)/pi_ref(yl|x)] )   (Eq. 7)

This file has:
  * dpo_loss (the paper's Appendix B code), its gradient weight sigma(r_l - r_w), implicit rewards;
  * a BANDIT (one prompt, K answers) where everything is exact: DPO on infinite Bradley-Terry data recovers
    pi_ref exp(r/beta)/Z, and the unweighted 'unlikelihood' update runs away;
  * a CONTROLLED-SENTIMENT toy with a real tiny autoregressive LM (GRU): prompts of 2 words, completions of 6 words,
    a ground-truth sentiment reward (+1 per positive word, -1 per negative word), pairs sampled from the SFT model and
    labelled by Bradley-Terry on the true reward (the paper used a sentiment classifier the same way);
  * methods: DPO, RLHF (Bradley-Terry reward model + policy gradient with the per-sequence KL penalty), Preferred-FT,
    Unlikelihood, Best-of-N; and the EXACT optimal reward-KL frontier (tokens are i.i.d. given the prompt, so the
    optimum factorises per token).
"""

import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------- the loss

def dpo_loss(pi_logp_w, pi_logp_l, ref_logp_w, ref_logp_l, beta):
    """Eq. 7. Inputs are summed log-probs of whole completions, shape (B,). Returns (mean loss, reward_w, reward_l),
    where the implicit rewards are beta * log pi/pi_ref (detached), as in the paper's code."""
    pi_logratios = pi_logp_w - pi_logp_l
    ref_logratios = ref_logp_w - ref_logp_l
    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    return losses.mean(), beta * (pi_logp_w - ref_logp_w).detach(), beta * (pi_logp_l - ref_logp_l).detach()


def dpo_grad_weight(reward_w, reward_l):
    """The gradient of the DPO loss is -beta * sigma(r_l - r_w) * (grad log pi(yw) - grad log pi(yl)): examples the
    implicit reward model gets WRONG get more weight."""
    return torch.sigmoid(reward_l - reward_w)


def unlikelihood_loss(pi_logp_w, pi_logp_l, alpha=1.0):
    """Baseline: raise log p(yw), lower log p(yl) -- DPO's update without the sigma weight and without pi_ref."""
    return -(pi_logp_w - alpha * pi_logp_l).mean()


def optimal_policy(ref_probs, rewards, beta):
    """Eq. 4: pi*(y) = pi_ref(y) exp(r(y)/beta) / Z."""
    with np.errstate(divide="ignore"):
        logits = np.log(ref_probs) + np.asarray(rewards) / beta
    p = np.exp(logits - logits.max())
    return p / p.sum()


def kl(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())

# ----------------------------------------------------------------------------------------------- the bandit

def bandit_expected_dpo(ref_probs, rewards, beta, steps=3000, lr=0.05, method="dpo", alpha=1.0):
    """Optimise a softmax policy on the EXACT expected loss over all pairs (y_i, y_j) ~ pi_ref x pi_ref, labelled by
    Bradley-Terry: P(i preferred to j) = sigma(r_i - r_j). method: 'dpo' | 'unlikelihood' | 'preferred_ft'."""
    ref = torch.tensor(ref_probs, dtype=torch.float64)
    r = torch.tensor(rewards, dtype=torch.float64)
    pair_w = ref[:, None] * ref[None, :]                                       # how often (i, j) is sampled
    p_pref = torch.sigmoid(r[:, None] - r[None, :])                            # P(i beats j)
    logits = torch.log(ref).clone().requires_grad_(True)                       # start at pi_ref
    opt = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        logp = torch.log_softmax(logits, 0)
        if method == "dpo":
            h = beta * (logp - torch.log(ref))                                 # implicit rewards
            loss = -(pair_w * p_pref * F.logsigmoid(h[:, None] - h[None, :])).sum()
        elif method == "unlikelihood":
            loss = -(pair_w * p_pref * (logp[:, None] - alpha * logp[None, :])).sum()
        else:                                                                  # SFT on the winner
            loss = -(pair_w * p_pref * logp[:, None]).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(logits, 0).detach().numpy()

# ----------------------------------------------------------------------------------------------- the sentiment toy

V = 16                                    # 0 = BOS, 1-3 positive, 4-6 negative, 7-15 neutral
POS, NEG = (1, 2, 3), (4, 5, 6)
SENT = np.zeros(V)
SENT[list(POS)], SENT[list(NEG)] = 1.0, -1.0
PROMPT_LEN, GEN_LEN = 2, 6


def prompt_polarity(prompts):
    s = SENT[np.asarray(prompts)].sum(-1)
    return np.sign(s).astype(int)


def generator_probs(q):
    """The 'true' review distribution (what SFT imitates): i.i.d. words, the prompt's tone tilts the sentiment."""
    p = np.zeros(V)
    p[list(POS)] = (0.2 + 0.1 * q) / 3
    p[list(NEG)] = (0.2 - 0.1 * q) / 3
    p[7:] = 0.6 / 9
    return p


def sample_prompts(rng, n):
    return rng.integers(1, V, size=(n, PROMPT_LEN))


def sample_reviews(rng, prompts):
    out = np.zeros((len(prompts), GEN_LEN), int)
    for i, q in enumerate(prompt_polarity(prompts)):
        out[i] = rng.choice(V, size=GEN_LEN, p=generator_probs(q))
    return out


def true_reward(completions):
    """The ground-truth 'sentiment classifier': +1 per positive word, -1 per negative word."""
    return SENT[np.asarray(completions)].sum(-1)


def optimal_frontier(beta):
    """The exact solution of max E[r] - beta KL(pi || generator), averaged over prompts. Because words are i.i.d. and the
    reward is a sum over words, pi* is i.i.d. too: pi*(w|q) = p(w|q) exp(s(w)/beta) / Z. Returns (KL, reward)."""
    qs = prompt_polarity(np.array(list(itertools.product(range(1, V), repeat=PROMPT_LEN))))
    kls, rs = [], []
    for q in qs:
        p = generator_probs(q)
        ps = optimal_policy(p, SENT, beta)
        kls.append(GEN_LEN * kl(ps, p))
        rs.append(GEN_LEN * float(ps @ SENT))
    return float(np.mean(kls)), float(np.mean(rs))


def optimal_reward_at_kl(target_kl):
    """The best achievable mean true reward at a given KL budget (bisection on beta along the exact frontier)."""
    lo, hi = math.log(0.05), math.log(100.0)
    for _ in range(50):
        mid = (lo + hi) / 2
        if optimal_frontier(math.exp(mid))[0] > target_kl:
            lo = mid
        else:
            hi = mid
    return optimal_frontier(math.exp(hi))[1]


def implicit_reward(pi, ref, beta, x, y):
    """Section 5: 'your language model is secretly a reward model': r_hat(x, y) = beta log pi(y|x)/pi_ref(y|x)."""
    with torch.no_grad():
        return beta * (seq_logp(pi, x, y) - seq_logp(ref, x, y))


def pair_accuracy(score_w, score_l):
    return float((score_w > score_l).float().mean() + 0.5 * (score_w == score_l).float().mean())


class LM(nn.Module):
    """A tiny autoregressive LM: embedding -> GRU -> next-word logits."""

    def __init__(self, d=48):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.rnn = nn.GRU(d, d, batch_first=True)
        self.out = nn.Linear(d, V)

    def forward(self, tokens, h=None):
        o, h = self.rnn(self.emb(tokens), h)
        return self.out(o), h


def seq_logp(model, prompts, completions):
    """Summed log pi(completion | prompt), shape (B,)."""
    x = torch.cat([torch.zeros(len(prompts), 1, dtype=torch.long), prompts, completions], 1)
    logits, _ = model(x[:, :-1])
    lp = torch.log_softmax(logits[:, PROMPT_LEN:], -1)                         # predictions for completion words
    return lp.gather(-1, completions[..., None]).squeeze(-1).sum(-1)


def sample(model, prompts, temperature=1.0):
    """Sample completions; returns (completions, summed log-probs under the model at temperature 1)."""
    with torch.no_grad():
        x = torch.cat([torch.zeros(len(prompts), 1, dtype=torch.long), prompts], 1)
        logits, h = model(x)
        last, toks = logits[:, -1], []
        for _ in range(GEN_LEN):
            t = torch.distributions.Categorical(logits=last / temperature).sample() if temperature > 0 else last.argmax(-1)
            toks.append(t)
            logits, h = model(t[:, None], h)
            last = logits[:, -1]
        y = torch.stack(toks, 1)
    return y


def train_sft(rng, steps=600, batch=128, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    m = LM()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        x = sample_prompts(rng, batch)
        y = sample_reviews(rng, x)
        loss = -seq_logp(m, torch.tensor(x), torch.tensor(y)).mean() / GEN_LEN
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()


def make_preferences(rng, ref, n):
    """Sample two completions per prompt from the SFT model, label with Bradley-Terry on the TRUE reward."""
    x = torch.tensor(sample_prompts(rng, n))
    torch.manual_seed(int(rng.integers(1 << 30)))
    y1, y2 = sample(ref, x), sample(ref, x)
    r1, r2 = true_reward(y1.numpy()), true_reward(y2.numpy())
    first = rng.random(n) < 1 / (1 + np.exp(-(r1 - r2)))
    yw = torch.where(torch.tensor(first)[:, None], y1, y2)
    yl = torch.where(torch.tensor(first)[:, None], y2, y1)
    return x, yw, yl


def copy_of(model):
    m = LM()
    m.load_state_dict(model.state_dict())
    return m


def train_offline(ref, data, method="dpo", beta=0.1, alpha=1.0, steps=400, batch=64, lr=1e-3, seed=0):
    """DPO / Unlikelihood / Preferred-FT on a fixed preference dataset (no sampling during training)."""
    torch.manual_seed(seed)
    x, yw, yl = data
    pi = copy_of(ref).train()
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    with torch.no_grad():
        ref_w, ref_l = seq_logp(ref, x, yw), seq_logp(ref, x, yl)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (batch,))
        lw, ll = seq_logp(pi, x[idx], yw[idx]), seq_logp(pi, x[idx], yl[idx])
        if method == "dpo":
            loss = dpo_loss(lw, ll, ref_w[idx], ref_l[idx], beta)[0]
        elif method == "unlikelihood":
            loss = unlikelihood_loss(lw, ll, alpha) / GEN_LEN
        else:
            loss = -lw.mean() / GEN_LEN
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pi.eval()


def train_reward_model(data, steps=800, lr=0.05):
    """Bradley-Terry reward model on bag-of-words counts of the completion (plus the prompt's tone)."""
    x, yw, yl = data

    def feats(xx, yy):
        c = F.one_hot(yy, V).float().sum(1)
        q = torch.tensor(prompt_polarity(xx.numpy()), dtype=torch.float32)[:, None]
        return torch.cat([c, q * c], 1)

    w = torch.zeros(2 * V, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    fw, fl = feats(x, yw), feats(x, yl)
    for _ in range(steps):
        loss = -F.logsigmoid((fw - fl) @ w).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    w = w.detach()
    return lambda xx, yy: feats(xx, yy) @ w


def train_rlhf(ref, reward_fn, beta=0.1, steps=300, batch=64, lr=1e-3, seed=0, rng=None):
    """Policy gradient (REINFORCE with a mean baseline) on r(x, y) - beta (log pi - log pi_ref), the reward PPO sees in
    RLHF. Simpler than PPO (no clipping, no value net) but it optimises the same KL-penalised objective."""
    torch.manual_seed(seed)
    rng = rng or np.random.default_rng(seed)
    pi = copy_of(ref).train()
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    for _ in range(steps):
        x = torch.tensor(sample_prompts(rng, batch))
        y = sample(pi, x)
        lp = seq_logp(pi, x, y)
        with torch.no_grad():
            r = reward_fn(x, y) - beta * (lp - seq_logp(ref, x, y))
            adv = r - r.mean()
        loss = -(adv * lp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pi.eval()


def evaluate(pi, ref, rng, n=2000, temperature=1.0):
    """Mean TRUE reward and sequence-level KL(pi || pi_ref) (sum of per-step KLs, as in the paper's footnote)."""
    x = torch.tensor(sample_prompts(rng, n))
    y = sample(pi, x, temperature)
    with torch.no_grad():
        xx = torch.cat([torch.zeros(n, 1, dtype=torch.long), x, y], 1)[:, :-1]
        lp = torch.log_softmax(pi(xx)[0][:, PROMPT_LEN:], -1)
        lq = torch.log_softmax(ref(xx)[0][:, PROMPT_LEN:], -1)
        k = (lp.exp() * (lp - lq)).sum(-1).sum(-1).mean().item()
    return {"reward": float(true_reward(y.numpy()).mean()), "KL": k}


def best_of_n(ref, reward_fn, rng, n_samples, n=1000):
    """Sample N completions from the SFT model, keep the one the reward model likes best. KL <= log N - (N-1)/N."""
    x = torch.tensor(sample_prompts(rng, n))
    ys = [sample(ref, x) for _ in range(n_samples)]
    scores = torch.stack([reward_fn(x, y) for y in ys], 1)
    best = torch.stack(ys, 1)[torch.arange(n), scores.argmax(1)]
    return {"reward": float(true_reward(best.numpy()).mean()),
            "KL": math.log(n_samples) - (n_samples - 1) / n_samples}


REPORTED = {
    "default hyper-parameters": "beta = 0.1 (0.5 for TL;DR), batch 64, RMSprop lr 1e-6, 150 warm-up steps",
    "IMDb sentiment": "DPO gives the best reward-KL frontier, dominating PPO and even PPO with the true reward (PPO-GT)",
    "TL;DR win rate vs reference (GPT-4)": "DPO ~61% at temperature 0 vs PPO 57% at its best temperature",
    "human eval, TL;DR": "DPO (temp 0.25) preferred to PPO (temp 0) 58% of the time",
    "Anthropic HH": "DPO the only efficient method improving over the chosen responses; ~ Best-of-128",
    "CNN/DailyMail (out of distribution)": "GPT-4 win rate DPO 0.36 / 0.31 vs PPO 0.26 / 0.23 (temp 0 / 0.25)",
    "unweighted update": "the naive version without the sigma weighting makes the LM degenerate (Table 3)",
}
