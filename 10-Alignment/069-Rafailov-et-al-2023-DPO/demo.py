"""DPO in ~15 seconds: the loss and its gradient weight, an exact bandit where DPO recovers the RLHF optimum
pi_ref exp(r/beta)/Z, and a controlled-sentiment task with a real tiny LM where DPO, RLHF (reward model + policy gradient
with a KL penalty), Preferred-FT, Unlikelihood and Best-of-N are compared against the EXACT optimal reward-KL frontier."""

import math
import time

import numpy as np
import torch

from dpo import (REPORTED, best_of_n, bandit_expected_dpo, dpo_grad_weight, dpo_loss, evaluate, implicit_reward, kl,
                 make_preferences, optimal_frontier, optimal_policy, optimal_reward_at_kl, pair_accuracy, seq_logp,
                 train_offline, train_reward_model, train_rlhf, train_sft, true_reward)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. The DPO loss on one pair (Eq. 7)")
lw, ll, rw, rl = torch.tensor([-10.0]), torch.tensor([-12.0]), torch.tensor([-11.0]), torch.tensor([-11.0])
loss, r_w, r_l = dpo_loss(lw, ll, rw, rl, beta=0.5)
print("  log pi(yw)=-10, log pi(yl)=-12, log pi_ref(yw)=log pi_ref(yl)=-11, beta=0.5")
print(f"  implicit rewards: r_w = 0.5*(-10+11) = {r_w.item():.2f}, r_l = 0.5*(-12+11) = {r_l.item():.2f}")
print(f"  loss = -log sigmoid({r_w.item():.2f} - ({r_l.item():.2f})) = {loss.item():.3f}; gradient weight "
      f"sigmoid(r_l - r_w) = {dpo_grad_weight(r_w, r_l).item():.3f}  (small: this pair is already ranked correctly)")

section("2. Exact bandit: DPO on infinite Bradley-Terry data recovers pi* = pi_ref exp(r/beta) / Z (Eq. 4)")
ref_p, r = np.array([0.3, 0.25, 0.2, 0.15, 0.07, 0.03]), np.array([0.0, 1.0, 2.0, -1.0, 3.0, 0.5])
print(f"  pi_ref = {ref_p.tolist()},  r = {r.tolist()}")
for beta in (0.5, 1.0, 2.0):
    d, s = bandit_expected_dpo(ref_p, r, beta), optimal_policy(ref_p, r, beta)
    print(f"    beta={beta}: DPO {np.round(d, 3)}  closed form {np.round(s, 3)}  max |diff| {np.abs(d - s).max():.1e}")
ul = bandit_expected_dpo(ref_p, r, 1.0, method="unlikelihood")
pf = bandit_expected_dpo(ref_p, r, 1.0, method="preferred_ft")
print(f"    unlikelihood (no sigma weight, no pi_ref): {np.round(ul, 3)}  KL to pi_ref {kl(ul, ref_p):.2f}, "
      f"E[r] {ul @ r:.2f}  (best answer r=3 gets only {ul[4]:.0%})")
print(f"    preferred-FT (SFT on winners):            {np.round(pf, 3)}  KL to pi_ref {kl(pf, ref_p):.2f}, E[r] {pf @ r:.2f}")
s1 = optimal_policy(ref_p, r, 1.0)
print(f"    DPO / RLHF optimum at beta=1:              KL {kl(s1, ref_p):.2f}, E[r] {s1 @ r:.2f}")

section("3. Controlled sentiment with a real tiny LM (GRU): prompt of 2 words -> 6-word 'review'")
rng = np.random.default_rng(0)
ref = train_sft(rng)
base = evaluate(ref, ref, rng)
data = make_preferences(rng, ref, 2000)
print(f"  SFT model: mean true reward {base['reward']:+.2f} (+1 per positive word, -1 per negative word)")
print("  2000 preference pairs: two samples from SFT per prompt, labelled by Bradley-Terry on the true reward")
rm = train_reward_model(data)
rows = []
for beta in (0.5, 1.0, 2.0):
    rows.append(("DPO", beta, evaluate(train_offline(ref, data, "dpo", beta=beta), ref, rng)))
    rows.append(("RLHF (RM + policy gradient)", beta, evaluate(train_rlhf(ref, rm, beta=beta, rng=rng), ref, rng)))
rows.append(("Preferred-FT", None, evaluate(train_offline(ref, data, "preferred_ft"), ref, rng)))
for a in (0.1, 1.0):
    rows.append((f"Unlikelihood alpha={a}", None, evaluate(train_offline(ref, data, "unlikelihood", alpha=a), ref, rng)))
for n in (4, 16):
    rows.append((f"Best-of-{n} (by the RM)", None, best_of_n(ref, rm, rng, n)))
print("    method                         beta   KL(pi||ref)  true reward   best possible at that KL   objective R - beta*KL (optimum)")
for name, beta, e in rows:
    best = optimal_reward_at_kl(e["KL"])
    obj = f"{e['reward'] - beta * e['KL']:5.2f}  ({optimal_frontier(beta)[1] - beta * optimal_frontier(beta)[0]:.2f})" if beta else ""
    print(f"    {name:30s} {beta if beta else '':4}   {e['KL']:6.2f}      {e['reward']:+6.2f}        {best:+6.2f}"
          f"  ({e['reward'] / best:4.0%})         {obj}")
dpo1 = [e for n, b, e in rows if n == "DPO" and b == 1.0][0]
rl1 = [e for n, b, e in rows if n.startswith("RLHF") and b == 1.0][0]
ul1 = [e for n, b, e in rows if n == "Unlikelihood alpha=1.0"][0]
print(f"  -> DPO lands close to the exact frontier (at beta=1 it gets {dpo1['reward'] / optimal_reward_at_kl(dpo1['KL']):.0%} of the best reward "
      f"possible at its KL) with no reward model and no")
print(f"     sampling during training. Honest note: here RLHF is closer still ({rl1['reward'] / optimal_reward_at_kl(rl1['KL']):.0%}), "
      "so we do NOT reproduce the paper's")
print("     'DPO strictly dominates PPO' -- our reward model is exactly the right family (bag of words), RL samples fresh")
print("     completions every step, while DPO sees only the 2000 fixed pairs (and its gap grows at large beta, i.e. small KL).")
print(f"  -> Unlikelihood (alpha=1) drifts far from the reference (KL {ul1['KL']:.1f}) for only "
      f"{ul1['reward'] / optimal_reward_at_kl(ul1['KL']):.0%} of the reward possible at that KL; Preferred-FT barely moves.")

section("4. 'Your language model is secretly a reward model' (Section 5)")
held = make_preferences(np.random.default_rng(123), ref, 2000)
x, yw, yl = held
pi = train_offline(ref, data, "dpo", beta=1.0)
true_w, true_l = torch.tensor(true_reward(yw.numpy())), torch.tensor(true_reward(yl.numpy()))
print("  accuracy at ranking 2000 held-out preference pairs (labels are noisy Bradley-Terry samples):")
print(f"    true reward (the ceiling)               {pair_accuracy(true_w, true_l):.1%}")
print(f"    explicit Bradley-Terry reward model     {pair_accuracy(rm(x, yw), rm(x, yl)):.1%}")
print(f"    DPO implicit reward beta*log pi/pi_ref  "
      f"{pair_accuracy(implicit_reward(pi, ref, 1.0, x, yw), implicit_reward(pi, ref, 1.0, x, yl)):.1%}")
print("  -> the DPO policy ranks responses almost as well as a separately trained reward model.")

section("5. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
