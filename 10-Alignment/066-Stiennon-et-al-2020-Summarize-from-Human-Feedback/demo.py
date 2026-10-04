"""Learning to summarize from human feedback in ~30 seconds, with real (tiny) networks: supervised fine-tuning on
noisy reference summaries, human comparisons from simulated labelers, a Bradley-Terry reward model, and PPO with the
paper's KL-penalised reward, with and without the KL penalty."""

import random
import time

import torch

from rlhf import (REPORTED, collect_comparisons, evaluate_policy, make_post, ppo, reference_summary, rm_accuracy,
                  train_reward_model, train_sft, true_quality)

torch.set_num_threads(1)
T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. The toy summarisation task")
rng = random.Random(3)
post, top = make_post(rng)
print(f"  post: {' '.join(post)}")
print(f"  the two most frequent topics are {top[0]} (x4) and {top[1]} (x3); ideal summary '{top[0]} {top[1]}' scores "
      f"{true_quality(post, top, [top[0], top[1]])}")
print("  labelers' true utility: +1 per main topic, +0.3 right order, -1 per hallucinated token, -0.5 per filler,")
print("  -0.6 per repetition, -0.4 per token beyond 3")
print("  reference TL;DRs (noisy, like Reddit's): e.g. " + " | ".join(" ".join(reference_summary(top, rng)) for _ in range(5)))

section("2. Step 1 (SFT) and steps 2-3 (comparisons -> reward model)")
t = time.time()
sft = train_sft(steps=500)
print(f"  SFT on reference summaries: {time.time() - t:.1f}s")
t = time.time()
data = collect_comparisons(sft, 3000)
rm = train_reward_model(sft, data[:2600], steps=400)
print(f"  3000 labelled comparisons of SFT samples (and some references); reward model = SFT model + scalar head,")
print(f"  loss -log sigmoid(r_chosen - r_rejected): held-out accuracy {rm_accuracy(rm, data[2600:]):.1%} "
      f"({time.time() - t:.1f}s)")
print("  (labelers here are noisy, P(prefer A) = sigmoid((qA - qB)/0.5), so even a perfect RM cannot reach 100%)")

section("3. Step 4: PPO against the reward model with R = r(x, y) - beta * log(pi_RL / pi_SFT)")
results = {"SFT (imitation of references)": evaluate_policy(sft, rm)}
for beta in (0.0, 0.2):
    t = time.time()
    policy, hist = ppo(sft, rm, beta, iters=100)
    results[f"PPO, beta = {beta}"] = evaluate_policy(policy, rm)
    print(f"  beta = {beta}: {time.time() - t:.1f}s; during training RM {hist[0][0]:.2f} -> {hist[-1][0]:.2f}, "
          f"true quality {hist[0][1]:.2f} -> {hist[-1][1]:.2f}, KL to SFT {hist[-1][2]:.2f} nats")
print("\n    policy (greedy summaries)          true quality   RM score   preferred to reference   example (ideal, output)")
for name, r in results.items():
    print(f"    {name:32s}     {r['true quality']:5.2f}       {r['RM score']:5.2f}          {r['preferred to reference']:5.1%}"
          f"            {' '.join(r['example'][0])} -> {' '.join(r['example'][1])}")
print("  -> imitating references copies their habit of naming only one topic; optimising human preferences learns")
print("     to name both main topics, and the summaries are preferred to the references (the paper: 61% for 1.3B).")
r0, r2 = results["PPO, beta = 0.0"], results["PPO, beta = 0.2"]
print("  -> without the KL penalty the policy drifts to what the reward model over-rates (here: repeating the main")
print(f"     topic), getting the highest RM score ({r0['RM score']:.2f} vs {r2['RM score']:.2f}) but lower true quality than with")
print(f"     the penalty ({r0['true quality']:.2f} vs {r2['true quality']:.2f}), though still above SFT: the start of Figure 5's over-optimisation.")

section("4. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
