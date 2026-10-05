# Evolution Strategies, explained simply

**Paper:** Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever (OpenAI), *Evolution Strategies as a Scalable Alternative to Reinforcement Learning*, 2017.

**In one sentence:** instead of computing gradients through an agent's actions (as policy-gradient RL does), **randomly jiggle all the policy's weights, run whole episodes, and move the weights toward the jiggles that scored well**. This "evolution strategy" needs no backpropagation and no value function, and is so easy to parallelise (workers exchange one number each) that 1,440 CPU cores solved 3D humanoid walking in 10 minutes.

---

## 1. The idea in plain words

- **Treat the whole RL problem as a black box F(θ):** put in policy weights θ, get back the episode's total reward.
- **Each step:**
  1. make n noisy copies θ + σεᵢ of the weights (εᵢ is Gaussian noise, σ is its size);
  2. run one episode with each copy and get its return Fᵢ;
  3. move θ toward the noise directions with high return, and away from those with low return:
  ```
  θ ← θ + α · (1 / (n σ)) · Σᵢ Fᵢ εᵢ            (Algorithm 1)
  ```
- **Why "evolution":** it's like a population of mutants. The good ones pull the parent toward themselves.

---

## 2. Why that update is a gradient

- **The objective ES really optimises** is the **Gaussian-blurred** return:
  ```
  J(θ) = E_{ε ~ N(0, I)} [ F(θ + σ ε) ]
  ```
- **Its gradient** is (the "score-function" trick, since ∇ log N(θ + σε; θ, σ²) = ε/σ):
  ```
  ∇J(θ) = (1/σ) · E[ F(θ + σ ε) · ε ]
  ```
- **Algorithm 1 is a Monte-Carlo estimate** of exactly this.
- **Blurring makes everything smooth,** even if F itself jumps around: discrete actions, hard thresholds, or a simulator you can't differentiate.

**Worked example (one dimension).** F(θ) = −(θ − 1)², at θ = 0, σ = 0.1.
- Draw ε = +1: F(0.1) = −0.81. Draw ε = −1: F(−0.1) = −1.21.
- **Estimate** = (1/(2 · 0.1)) · [(−0.81)(+1) + (−1.21)(−1)] = 5 · 0.40 = **2.0**.
- The true gradient −2(θ − 1) = **2**. With a mirrored pair, the estimate is exact for a quadratic.

### ES as finite differences
- Since E[F(θ) ε] = 0, the estimate equals E[(F(θ + σε) − F(θ)) ε]/σ. That is a **finite difference along a random direction**.
- So it scales with the problem's **intrinsic** difficulty, not its raw parameter count (section 3.2 of the paper).

---

## 3. The tricks that make it work

1. **Mirrored (antithetic) sampling:** always evaluate pairs +ε and −ε. Anything in F that is *even* in ε (like curvature) cancels exactly.
2. **Fitness shaping:** replace the returns by their **ranks**, scaled to [−0.5, 0.5]. This removes the effect of outliers and makes the update scale-free.
   - **Example:** returns (5, −1, 100) become ranks (1, 0, 2) and then (0, −0.5, +0.5). The 100 no longer dominates.
3. **Weight decay:** keeps the weights from growing large compared with the fixed noise σ.
4. **Fixed σ:** the paper saw no benefit from adapting it.
5. **Virtual batch normalisation (Atari) and discretised actions (some MuJoCo tasks):** these encourage diverse behaviour.

---

## 4. Algorithm 2: parallel ES with shared seeds

- **The communication problem:** with policy gradients, each worker must send its **whole gradient** (millions of floats) every step.
- **ES workers share their random seeds in advance**, so every worker can regenerate every other worker's noise εⱼ. The step then looks like this:
  1. each worker i evaluates F(θ + σεᵢ) and **broadcasts one scalar**, its return;
  2. every worker rebuilds all εⱼ from the seeds and applies **the same update**, so all copies of θ stay identical.
- **Example:** a 1-million-parameter policy on 1,000 workers. A gradient method sends 10⁶ floats per worker per step; ES sends 1. Communication is no longer the bottleneck.

---

## 5. When is ES better than policy gradients? (Section 3.1)

- **Both methods add noise to explore:**
  - policy gradients add it **to every action**: the score is Σₜ ∇ log p(aₜ), a sum of T terms;
  - ES adds it **once to the parameters**.
- **If each single action barely affects the return** (true for hard problems):
  ```
  Var[policy gradient] ≈ Var[R] · Var[Σₜ ∇ log p(aₜ)]   — grows ~linearly with episode length T
  Var[ES gradient]     ≈ Var[R] · Var[∇ log p(θ̃)]       — independent of T
  ```
- **So ES suits long episodes,** delayed rewards and fine time steps. It is **invariant to frame-skip:** the paper's Pong runs with frame-skip 1, 2, 3, 4 learn similarly.
- **Policy-gradient methods fight this variance** with discounting or value functions, both of which add bias.

---

## 6. The paper's results

- **3D Humanoid:**
  - **under 10 minutes** to reach score 6,000 on 1,440 CPU cores (80 machines);
  - about **11 hours on one 18-core machine**;
  - **linear speedup** in the number of cores (Figure 1).
- **Atari:**
  - 51 games, about 1 hour each on 720 CPUs;
  - better than published A3C (1 day of training) on **23 games**, worse on **28**;
  - ES used **3×–10× more data**, partly offset by about **3× less computation per step** (no backpropagation).
- **MuJoCo vs TRPO (Table 1)**, ratio of ES to TRPO timesteps to reach TRPO's 5M-step performance:

| Task | 25% | 50% | 75% | 100% |
|---|---|---|---|---|
| HalfCheetah | 0.15 | 0.49 | 0.42 | 0.58 |
| Hopper | 0.53 | 3.64 | 6.05 | 6.94 |
| InvertedDoublePendulum | 0.46 | 0.48 | 0.49 | 1.23 |
| InvertedPendulum | 0.28 | 0.52 | 0.78 | 0.88 |
| Swimmer | 0.56 | 0.47 | 0.53 | 0.30 |
| Walker2d | 0.41 | 5.69 | 8.02 | 7.88 |

  Within 10× on the hard tasks; up to about 3× **better** than TRPO on simple ones.
- **Different exploration:** on Humanoid, ES found a wide variety of gaits (sideways, backwards), which TRPO never did.

---

## 7. What our code found

**1. The estimator** (F = −|θ − 1|², d = 10, true gradient 2 in every coordinate; 100 evaluations per estimate, 300 estimates):

| σ | Plain sampling: mean / std | Mirrored pairs: mean / std |
|---|---|---|
| 0.1 | 2.05 / **0.654** | 2.06 / 0.925 |
| 1.0 | 2.06 / 0.849 | 2.06 / 0.925 |
| 3.0 | 2.10 / 1.735 | 2.06 / **0.925** |

- **Both are unbiased.**
- **Mirrored pairs remove the curvature term**, so their noise doesn't depend on σ.
- **Honest note:** at small σ the function is nearly linear, and mirroring only halves the number of independent directions, so it is **worse** there. Its benefit grows with σ, or with the curvature of F.

**2. Algorithm 2:** 16 simulated workers with shared seeds.
- After 150 updates all hold **bit-identical** parameters, and the objective improved from −100 to −4.1.
- Each worker sent **150 scalars** instead of 15,000 gradient floats (100× less for a 100-parameter model; the ratio is the parameter count).

**3. CartPole** (our vectorised numpy version with Gym's constants; linear policy, 5 parameters; 100 updates):

| Method | Final greedy return | Environment steps used |
|---|---|---|
| ES (40 perturbations per update) | **500** / 500 | 1,096,703 |
| REINFORCE (40 episodes per update) | **500** / 500 | 681,347 |

- **Both solve it. ES used 1.6× the environment steps,** in line with the paper's "more data, less compute".
- **REINFORCE's stochastic training policy** never averaged ≥ 475 during training, even though its greedy policy scores 500.

**4. Variance vs episode length** (T actions, return = mean of actions + noise of std 0.3, true gradient 0.25):

| T | REINFORCE variance / true² | ES variance / true² |
|---|---|---|
| 10 | 5.6 | 10.1 |
| 30 | 13.1 | 9.4 |
| 100 | 39.5 | 9.8 |
| 300 | 111.5 | 9.6 |
| 1,000 | **367.0** | **9.8** |

The policy-gradient variance grows about linearly with T (×66 for ×100 in T); ES stays flat. This is the paper's section 3.1 argument, measured.

**5. Duplicated features** (regression with x vs (x, x); mean final loss over 20 seeds):

| Setting | Final loss |
|---|---|
| 5 features (σ, lr) | 7.08·10⁻⁵ |
| 10 features, σ/√2, lr/2 | 7.31·10⁻⁵ |
| 10 features, σ/2, lr/2 (the paper's wording) | 7.31·10⁻⁵ |
| 10 features, same σ and lr | 2.73·10⁻⁹ |

- **With the learning rate halved, doubling the parameters doesn't make the problem harder.**
- **Note on σ:** the paper says to divide σ by two. Our derivation says σ/√2 is the exact match, because the effective weight w₁ + w₂ gets noise σ′(ε₁ + ε₂) with variance 2σ′². On this quadratic objective σ cancels completely with mirrored sampling, so both give identical results here.

**`experiments.py`:**
- **E1:** CartPole data efficiency over population size and σ;
- **E2:** frame-skip;
- **E3:** shaping and mirroring ablation with an MLP policy;
- **E4:** time per update vs population;
- **E5:** the variance table to T = 10,000.
- Only an E4 smoke run was done here: 4.9 ms per update for population 20, 9.5 ms for 80, on one process.

---

## 8. Why it matters

- **This paper revived evolution strategies for deep RL** and showed that throwing CPUs at a simple algorithm can match sophisticated ones in wall-clock time.
- **Its ideas reappear in many places:**
  - black-box and zeroth-order optimisation, e.g. for non-differentiable objectives and memory-light fine-tuning of language models with forward passes only;
  - evolution-based meta-learning;
  - "shared random seed" tricks for low-communication distributed training.

---

## 9. Check yourself

1. Write the ES gradient estimate for n samples.
<details><summary>Answer</summary>ĝ = (1/(nσ)) Σᵢ F(θ + σεᵢ) εᵢ with εᵢ ~ N(0, I). It estimates the gradient of the Gaussian-smoothed objective E[F(θ + σε)].</details>

2. Compute the ES estimate for F(θ) = θ² at θ = 1, σ = 0.5, using the mirrored pair ε = ±1.
<details><summary>Answer</summary>F(1.5) = 2.25 and F(0.5) = 0.25. Estimate = (1/(2 · 0.5)) · (2.25 · 1 + 0.25 · (−1)) = 1 · 2.0 = 2.0, the exact derivative 2θ = 2.</details>

3. What do mirrored pairs cancel, and when do they hurt?
<details><summary>Answer</summary>The even part of F around θ (e.g. curvature): F(θ + σε) + F(θ − σε) contributes nothing to the update. When F is nearly linear (small σ), mirroring just halves the number of independent directions, raising variance (0.93 vs 0.65 in our toy).</details>

4. Convert returns (3, 10, −2, 7) to centred ranks.
<details><summary>Answer</summary>Ranks (1, 3, 0, 2); divide by n − 1 = 3 and subtract 0.5: (−0.167, 0.5, −0.5, 0.167).</details>

5. In Algorithm 2, what does each worker send, and why is that enough?
<details><summary>Answer</summary>One scalar, its episode return. All workers know each other's random seeds, so each can regenerate every perturbation εⱼ and compute the identical update locally.</details>

6. Why does the policy-gradient estimator's variance grow with episode length T, but ES's does not?
<details><summary>Answer</summary>The policy-gradient score is a sum of T per-action terms (variance about T times larger), while ES's score is a single parameter perturbation per episode. When individual actions barely affect the return, the policy-gradient noise grows with T; in our toy, 5.6 → 367 vs ES flat at about 10.</details>

7. Why is ES invariant to frame-skip?
<details><summary>Answer</summary>Its gradient estimate doesn't depend on how many decisions the episode is cut into. It only sees the parameter perturbation and the total return.</details>

8. What did ES cost and gain against A3C on Atari?
<details><summary>Answer</summary>3–10× more environment data, but about 3× less computation per step (no backprop) and massive parallelism: 1 hour on 720 CPUs vs 1 day for A3C. It was better on 23 of 51 games and worse on 28.</details>

9. If you duplicate every input feature, how should σ and the learning rate change for ES to behave the same?
<details><summary>Answer</summary>Halve the learning rate and scale σ by 1/√2 (our derivation; the paper says divide both by two). The effective weight w₁ + w₂ then sees the same noise and the same step size.</details>

10. Why did OpenAI's 1,440-core run achieve linear speedup?
<details><summary>Answer</summary>Evaluations are independent full episodes, workers exchange only scalars, and the update is cheap, so adding cores adds evaluations per second without a communication bottleneck.</details>
