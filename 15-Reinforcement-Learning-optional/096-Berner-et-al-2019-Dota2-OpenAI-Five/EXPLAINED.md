# OpenAI Five (Dota 2 with large-scale deep RL), explained simply

**Paper:** Christopher Berner, Greg Brockman, Brooke Chan, Vicki Cheung, Przemysław Dębiak, Christy Dennison, David Farhi, Quirin Fischer, Shariq Hashme, Chris Hesse, Rafal Józefowicz, Scott Gray, Catherine Olsson, Jakub Pachocki, Michael Petrov, Henrique Pondé de Oliveira Pinto, Jonathan Raiman, Tim Salimans, Jeremy Schlatter, Jonas Schneider, Szymon Sidor, Ilya Sutskever, Jie Tang, Filip Wolski, Susan Zhang (OpenAI), *Dota 2 with Large Scale Deep Reinforcement Learning*, 2019.

**In one sentence:** OpenAI took a fairly standard RL algorithm (PPO), scaled it to batches of millions of game frames on thousands of GPUs for **10 months** of self-play, invented **surgery** to keep training while the game and network kept changing, and beat the **Dota 2 world champions (Team OG) 2–0**.

---

## 1. Why Dota 2 is hard

- **Long horizons:**
  - games run at 30 frames per second for about 45 minutes;
  - OpenAI Five acts every 4th frame, so about **20,000 moves per game** (chess has about 80, Go about 150).
- **Partial observability:** fog of war hides much of the map.
- **Huge action and observation spaces:**
  - **8,000–80,000** valid actions per step, depending on the hero;
  - about 16,000 values observed per step.
- **Five-player teams:** coordination and credit assignment between teammates.
- **Restrictions in the paper's version:** 17 of 117 heroes, and a few mechanics scripted or disabled.

---

## 2. The system

- **Model:**
  - a **single-layer 4096-unit LSTM** (84% of the parameters) with separate heads per action type;
  - each of the five heroes runs its own copy of the same network.
- **Algorithm:** **PPO** (Proximal Policy Optimization) with **GAE** (Generalized Advantage Estimation, λ = 0.95) and Adam.
- **Training:**
  - **self-play:** 80% of games against the current policy, 20% against older versions;
  - **batches** of about **1–3 million timesteps** (in LSTM windows of 16 steps).
- **Infrastructure:**
  - thousands of CPU **rollout workers** play games and send data every 256 steps;
  - **optimizer GPUs** take gradient steps and publish new parameters every 32 steps;
  - everything runs **asynchronously**.
- **Total:** **770 ± 50 PFlops/s·days** over 10 months. Compared with AlphaGo: 50–150× larger batch, 20× larger model, 25× longer training.

### PPO in one paragraph
- **The objective:** for each sample, compute the ratio r = π_new(a|s) / π_old(a|s) between the new and the data-collecting policy. Maximise
  ```
  min( r · A,  clip(r, 1 − ε, 1 + ε) · A )        ε = 0.2
  ```
  where A is the **advantage** (how much better the action was than expected).
- **The clip** stops one batch from moving the policy too far.
- **The gradient** (for samples where the clip isn't active) is A · r · ∇ log π(a|s).

### GAE in one paragraph
- **The building block:** the one-step "surprise"
  ```
  δₜ = rₜ + γ V(sₜ₊₁) − V(sₜ)
  ```
- **The advantage** is a discounted sum of these surprises:
  ```
  Aₜ = δₜ + (γλ) δₜ₊₁ + (γλ)² δₜ₊₂ + …
  ```
  - λ = 0 uses only one step (low variance, more bias from V);
  - λ = 1 uses the whole return (unbiased, high variance).
- **Example:** rewards (1, 1, 1), values all 0.5, γ = 0.9, λ = 0.5, and the episode ends after step 3.
  - δ₃ = 1 − 0.5 = **0.5** (no next state);
  - δ₂ = 1 + 0.9 · 0.5 − 0.5 = **0.95**, and δ₁ = **0.95**;
  - A₃ = 0.5;
  - A₂ = 0.95 + 0.45 · 0.5 = **1.175**;
  - A₁ = 0.95 + 0.45 · 1.175 = **1.479**.
  - (This is our test `test_gae_matches_hand_computation`.)

---

## 3. Surgery: changing the model without restarting

- **The problem:** over 10 months the team kept changing things: the game version, the observations, the actions, even the network size. Restarting a months-long training run after every change was impossible.
- **Surgery** builds a new model π̂ that computes **the same function** as the old one:
  ```
  π̂_θ̂(o) = π_θ(o)  for all observations o        (Eq. 1)
  ```
- **Widening a hidden layer** (Eq. 6), from y = W₁x + b₁ to a bigger ŷ:
  - copy the old weights;
  - give the new units **random incoming weights** (to break symmetry) and **zero outgoing weights**.
  - The next layer ignores the new units at first, so the output is unchanged. Gradients then move the zeros if the new units are useful.
- **Adding observations** (Eq. 9): the new input features get **zero weights**. The same game state, now encoded with extra numbers, still gives the same actions.
- **Widening the LSTM** (2048 → 4096): in a recurrent layer the new units feed back into everything.
  - Zero weights would leave the new units **symmetric forever**; full-size random weights would **change the policy**.
  - OpenAI used **small random weights**, choosing the largest scale that didn't noticeably lower the agent's skill.
- **Scale of use:** over twenty surgeries in 10 months.

**Rerun:**
- **The run:** to check surgery's cost, they trained a fresh agent ("Rerun") on the final environment: **2 months, 150 ± 5 PFlops/s·days** (about 20% of OpenAI Five's compute).
- **Result:** it reached a **> 98% win rate** against the final OpenAI Five.
- **Lessons:**
  - surgery saved enormous time during development (restarting after each of about 20 changes would have taken about **40 months** instead of 10);
  - but from-scratch training on a fixed environment reaches a higher final skill.

---

## 4. What matters at scale: data quality (Figure 5)

- **Batch size:**
  - Rerun's batch (983k timesteps) trained about **2.5× faster** than a 123k batch: 8× the data for 2.5× the speed.
  - The speedup is **sublinear**, but bigger batches still save wall-clock time.
- **Staleness:**
  - if data was generated by parameter version N and the optimizer is at version M, the staleness is M − N;
  - a staleness of about **8 versions** caused significant slowdowns;
  - the final system kept staleness between **0 and 1**.
- **Sample reuse:**
  - this is the rate at which optimizers consume data divided by the rate at which rollouts produce it;
  - reuse of **2–3** could cause a **2× slowdown**, and **8** could prevent learning a competent policy;
  - the target was about **1**.
- **The paper's conclusion:** "high quality data matters even more than compute consumed".

---

## 5. Team spirit and rewards

- **The reward is shaped:**
  - win (the main term);
  - plus smaller rewards for kills, deaths, gold, towers and so on;
  - made **zero-sum** by subtracting the enemy team's average reward.
- **Team spirit τ** mixes each hero's own reward ρᵢ with the team average:
  ```
  rᵢ = (1 − τ) ρᵢ + τ · mean(ρ)
  ```
  - τ = 0: every hero for itself;
  - τ = 1: everyone shares everything equally.
- **Why anneal τ:** the true goal is τ = 1 (team success), but low τ early **reduces gradient variance**, because each hero gets clearer credit for its own actions. OpenAI Five annealed τ from **0.3 to 1.0**.

**Worked example:** five heroes earn raw rewards (2, 0, 0, 0, 0), so the mean is 0.4.
- With τ = 0.3, hero 1 gets 0.7 · 2 + 0.3 · 0.4 = **1.52**, and the others get 0.3 · 0.4 = **0.12** each.
- With τ = 1, all get **0.4**. Hero 1's action now earns it only a fifth of the credit, mixed with four teammates' noise.

---

## 6. Results

- **Against the world champions:** beat **Team OG**, the reigning champions, **2–0** in a best-of-three on April 13, 2019.
- **OpenAI Five Arena** (April 18–21, 2019):
  - **won 7,215 of 7,257 games (99.4%)** against 3,193 human teams;
  - 3,140 of those wins were from abandoned games;
  - 29 teams beat it, for a total of 42 losses.
- **Reaction time** was **217 ms** on average (typical human visual reaction is about 250 ms).
- **Skill is tracked with TrueSkill:** a difference of about 8.3 points is roughly an 80% win rate.

---

## 7. What our code found (laptop scale)

**Surgery** (a small tanh MLP policy, 1,000 random states):

| Operation | Largest change in action probabilities |
|---|---|
| widen 16 → 32, new outgoing weights zero (Eq. 6) | **0.0** |
| add 3 observations with zero weights (Eq. 9) | **0.0** |
| recurrent-style widening, small random outgoing N(0, 0.01) | 0.024 |
| … N(0, 0.1) | 0.123 |
| … N(0, 0.5) | 0.777 |

**An environment under development** (PPO on CartPole; each stage adds 2 observation features and widens the network; 25 PPO versions per stage):

| Stage | Surgery: return at first → last version | Restart: return at first → last version |
|---|---|---|
| 1 | 20 → 241 | 23 → 246 |
| 2 | **260** → 500 | 28 → 280 |
| 3 | **500** → 500 | 18 → 438 |

Surgery keeps what was learned. Each restart starts again at about 20 and is still below 500 after the third stage.

**Data quality** (PPO + GAE on our vectorised CartPole; versions needed to reach a mean training return of 200):

| Setting | Versions | Effect |
|---|---|---|
| batch 8 episodes | 28 | |
| batch 32 episodes | 22 | 4× data → **1.27×** faster (sublinear) |
| fresh data (staleness 0) | 23 | |
| staleness 8 versions | **52** | **2.3× slower** |
| sample reuse 1 | 23 | |
| sample reuse 8 | 35 | 1.5× slower |

- **The same ordering as the paper:** staleness hurts most, reuse hurts, and bigger batches help sublinearly.
- **Honest note:** our reuse penalty (1.5× at 8) is milder than the paper's (2× at 2–3, failure at 8). CartPole is an easy task, and our buffer setup differs from theirs.
- **Over 3 seeds** (during development): batch 8 / 16 / 32 / 64 took 27 / 24 / 22 / 21 versions; staleness 0 / 1 / 2 / 4 / 8 took 24 / 25 / 28 / 35 / 51; reuse 1 / 2 / 4 / 8 took 24 / 24 / 28 / 35.

**Team spirit** (5 agents choosing effort, individual reward plus a team bonus, reward noise; REINFORCE):

| τ | Gradient signal-to-noise at start | Team reward after 50 / 100 / 300 steps |
|---|---|---|
| 0.0 | **0.186** | 0.59 / 0.77 / 0.78 |
| 0.3 | 0.186 | 0.47 / 0.75 / 0.78 |
| 1.0 | **0.095** | 0.15 / 0.46 / 0.75 |

Full team sharing halves each agent's gradient signal-to-noise and slows early learning. All three settings converge to the same team reward, which is why annealing τ upward works.

**`experiments.py`:**
- **E1–E3:** batch, staleness and reuse sweeps over 10 seeds;
- **E4:** surgery vs restart over 5 stages, and the recurrent-widening scale;
- **E5:** τ sweep.
- Only an E5 smoke run was done here (2 seeds: team reward over the first 100 steps 0.53 for τ = 0 vs 0.22 for τ = 1).

---

## 8. Why it matters

- **OpenAI Five showed that "standard RL + enormous scale + good engineering"** can master a complex, long-horizon, team-based, partially observed game.
- **Its lessons carried into large-scale training generally:**
  - surgery-style function-preserving model growth;
  - careful measurement of staleness and sample reuse in asynchronous systems;
  - big-batch scaling;
  - continual training of one long-lived model.
- **It appeared alongside DeepMind's AlphaStar (StarCraft II),** and both are milestones of RL at scale.

---

## 9. Check yourself

1. Why does OpenAI Five face about 20,000 decisions per game?
<details><summary>Answer</summary>Games last about 45 minutes at 30 frames per second, and the agent acts every 4th frame: 45 · 60 · 30 / 4 ≈ 20,000.</details>

2. Write PPO's clipped objective and explain the clip.
<details><summary>Answer</summary>min(r·A, clip(r, 1 − ε, 1 + ε)·A) with r = π_new/π_old. When an update has already made a good action much more likely (r > 1 + ε with A > 0), or a bad one much less likely, the gradient is cut off, so one batch can't move the policy too far.</details>

3. Compute GAE advantages for rewards (1, 1), values (0, 0), γ = 1, λ = 1, with the episode ending after step 2.
<details><summary>Answer</summary>δ₂ = 1 − 0 = 1; δ₁ = 1 + 0 − 0 = 1. A₂ = 1; A₁ = δ₁ + γλ·A₂ = 2 (the full return minus V, since λ = 1).</details>

4. How does surgery widen a hidden layer without changing the policy?
<details><summary>Answer</summary>Keep the old weights, give the new units random incoming weights (to break symmetry) and zero outgoing weights. The next layer ignores them until gradients make those weights non-zero, so the output is initially identical.</details>

5. Why couldn't the LSTM 2048 → 4096 surgery be exact?
<details><summary>Answer</summary>In a recurrent layer the new units' outputs feed back into all units. Zero weights would keep the new units symmetric (they'd never learn different things), while random weights change the function. OpenAI used small random weights, a near-exact compromise.</details>

6. Define staleness and sample reuse.
<details><summary>Answer</summary>Staleness: how many parameter versions old the policy that generated a sample is (M − N). Sample reuse: the optimizer's data consumption rate divided by the rollouts' production rate, i.e. how many times each sample is used on average.</details>

7. What batch-size speedup did the paper find, and is it linear?
<details><summary>Answer</summary>About 2.5× faster for 983k vs 123k timesteps (8× the data) early in training. That is sublinear, but still useful for wall-clock time.</details>

8. Compute team-spirit rewards for raw rewards (3, 1, 1, 0, 0) with τ = 0.5.
<details><summary>Answer</summary>Mean = 1.0. Hero 1: 0.5·3 + 0.5·1 = 2.0; heroes 2 and 3: 0.5·1 + 0.5 = 1.0; heroes 4 and 5: 0 + 0.5 = 0.5.</details>

9. What did Rerun show about surgery?
<details><summary>Answer</summary>Training from scratch on the final environment took 2 months and 20% of the compute and beat the final OpenAI Five more than 98% of the time. Surgery enabled continuous development (restarting each time would have taken about 40 months), but an agent with surgeries plateaued below a clean from-scratch run.</details>

10. In our toy, which hurt more: 8 versions of staleness or 8× sample reuse?
<details><summary>Answer</summary>Staleness: 2.3× more versions to reach the target, vs 1.5× for reuse, the same ordering the paper reports (though our reuse penalty is milder).</details>
