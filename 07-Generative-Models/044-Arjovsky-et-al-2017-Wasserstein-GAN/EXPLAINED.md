# Wasserstein GAN (WGAN), explained simply

**Paper:** Martin Arjovsky, Soumith Chintala & Léon Bottou, *Wasserstein Generative Adversarial Networks*, ICML 2017.

**In one sentence:** a normal GAN (paper 042) secretly minimises the Jensen–Shannon divergence, which gives **no useful signal** when the real and fake distributions don't overlap. WGAN minimises the **Earth-Mover (Wasserstein) distance** instead: "how much dirt, moved how far". That distance shrinks smoothly as the fakes get closer, so the generator always gets a useful gradient, and the loss finally means something.

---

## 1. The problem with the original GAN

- With an optimal discriminator, the GAN generator minimises 2·JSD(P_r ‖ P_g) − log 4 (paper 042's Theorem 1).
- That is fine *if the two distributions overlap*. But:
  - real images lie on a thin, **low-dimensional manifold** inside pixel space (most random pixel arrays are not images);
  - a generator that maps 100 noise numbers to 3072 pixels also produces a low-dimensional manifold;
  - two thin manifolds in a huge space **almost never overlap**.
- When they don't overlap:
  - **JSD = log 2**, no matter how close they are;
  - **KL = ∞**;
  - **total variation = 1**.

  The loss is flat, so its gradient is 0 (or garbage). The discriminator becomes perfect and the generator learns nothing. This is the "vanishing gradient" GAN practitioners saw.

---

## 2. Four ways to measure the distance between distributions (Section 2)

| distance | formula | intuition |
|---|---|---|
| Total variation δ | sup_A \|P_r(A) − P_g(A)\| | the biggest disagreement about any event |
| KL | ∫ p_r log(p_r/p_g) | asymmetric, and infinite if p_g = 0 where p_r > 0 |
| JS | ½KL(P_r ‖ M) + ½KL(P_g ‖ M), with M = (P_r + P_g)/2 | symmetric, bounded by log 2 |
| **Earth-Mover W** | inf_γ E_{(x,y)~γ} ‖x − y‖ | the minimum cost of moving the mass of P_r to P_g |

**Earth-Mover in words:**
- Think of P_r as a pile of dirt and P_g as a hole of the same volume.
- A **transport plan** γ(x, y) says how much dirt moves from x to y. Its marginals must be P_r and P_g.
- The cost is the amount × the distance. W is the cheapest plan's cost.

**Worked example (1-D):**
- P_r puts mass ½ at 0 and ½ at 1. P_g puts mass ½ at 2 and ½ at 3.
- **The best plan:** 0 → 2 and 1 → 3, so cost = ½·2 + ½·2 = **2**.
- **Alternative:** 0 → 3 and 1 → 2, so cost = ½·3 + ½·1 = 2 (a tie).
- **In 1-D there's a shortcut:** W = ∫|F_r(x) − F_g(x)| dx, the area between the two CDFs. Here F_r − F_g is ½ on [0, 1), 1 on [1, 2) and ½ on [2, 3). The area is ½ + 1 + ½ = **2** ✓ (one of our tests).

### 2.1 Example 1: learning parallel lines
**The setup:**
- Real data: P₀ = the distribution of (0, Z), with Z ~ Uniform[0, 1]. This is a vertical segment at x = 0.
- Generator: g_θ(z) = (θ, z). This is a vertical segment at x = θ.

**The distances:**

| | value for θ ≠ 0 | value at θ = 0 |
|---|---|---|
| **W(P₀, P_θ)** | **\|θ\|** | 0 |
| JS | log 2 | 0 |
| KL | ∞ | 0 |
| TV | 1 | 0 |

- **Why W = |θ|:** move each point (0, z) straight across to (θ, z). Every unit of mass travels exactly |θ|, and no plan can do better.
- **Why JS = log 2:** the segments never overlap, so the mixture M is half one and half the other. Each KL to M is log 2.

**The punchline:**
- As θ → 0, only W goes smoothly to 0.
- Gradient descent on W moves θ straight to 0.
- On JS, KL or TV, the loss is constant until the exact moment θ = 0. There is no gradient to follow.

### 2.2 Theorems 1–2 (the formal version)
- **Theorem 1:** if g_θ is continuous in θ, then **W(P_r, P_θ) is continuous in θ**. If g_θ is locally Lipschitz (true of neural networks), W is also **differentiable almost everywhere**. JS and KL are not.
- **Theorem 2:** convergence in KL implies convergence in JS ⇔ TV, which implies convergence in W ⇔ convergence in distribution. So W is the **weakest** of the four.
  - "Weakest" is good here: more sequences of distributions converge under W, so it gives a usable signal in more situations.

---

## 3. From an impossible minimum to a trainable maximum (Section 3)

### 3.1 The dual form (Eq. 2)
The infimum over transport plans (Eq. 1) is intractable in high dimensions. The **Kantorovich–Rubinstein duality** rewrites it as:

```
W(P_r, P_g) = sup_{‖f‖_L ≤ 1}  E_{x~P_r}[ f(x) ] − E_{x~P_g}[ f(x) ]
```

- Search over all **1-Lipschitz** functions f: those with |f(x) − f(y)| ≤ ‖x − y‖, so their slope is at most 1 everywhere.
- Find the f that scores the real data as high as possible and the fake data as low as possible, *on average*.

**Why it makes sense (1-D):**
- If f could be arbitrarily steep, the gap could be made infinite.
- Limiting the slope to 1 makes the best achievable gap equal to the cost of moving the mass.
- Example: a point mass at 0 vs a point mass at 3. The best f is f(x) = −x, with gap f(0) − f(3) = 3 = W ✓.

**Our demo** solves both forms as linear programs on a 9-point grid:
- primal (cheapest plan) = **0.206682**;
- dual (best 1-Lipschitz f) = **0.206682**;
- the optimal f's steepest slope is exactly 1.000.

### 3.2 The critic (Eq. 3)
- Replace "all 1-Lipschitz f" with a neural network f_w whose weights w lie in a **compact set** (a bounded box).
- Such a network is **K-Lipschitz** for some K that depends only on the box, so

  ```
  max_w  E_{P_r}[ f_w(x) ] − E_z[ f_w(g_θ(z)) ]   ≈   K · W(P_r, P_θ)
  ```

- f_w is called the **critic**, not the discriminator: it outputs an unbounded score, with **no sigmoid**, rather than a probability.

### 3.3 Theorem 3: the generator's gradient

```
∇_θ W(P_r, P_θ) = −E_z[ ∇_θ f(g_θ(z)) ]        (f = the optimal critic)
```

- So the generator just minimises **−E[f(G(z))]**: push the fakes toward higher critic scores.
- **Example:** P_r = N(0, 1) and P_θ = N(θ, 1) with θ > 0. W = θ, and the optimal f(x) = −x.
  - The generator's loss is −E[−(θ + z)] = θ, and its gradient is **1** = dW/dθ ✓ (one of our tests).

### 3.4 Enforcing Lipschitz: weight clipping
- After each critic update, clamp every weight into [−c, c], with c = 0.01.
- This makes the weight space compact, so f_w is K-Lipschitz. A bound: K ≤ Π (spectral norms of the layers), and each spectral norm is ≤ c·√(m·n).
- **The paper is blunt:** "Weight clipping is a clearly terrible way to enforce a Lipschitz constraint."
  - **Large c:** the critic takes forever to reach optimality.
  - **Small c:** gradients vanish in deep critics.
- **Our demo** (3-layer critic): the Lipschitz bound falls from 221 (c = 1) to 0.91 (c = 0.1) to 0.00098 (c = 0.01). It shrinks roughly like c³.
- The later fix was the **gradient penalty** (WGAN-GP, Gulrajani et al. 2017).

---

## 4. Algorithm 1

```
defaults: α = 0.00005, c = 0.01, m = 64, n_critic = 5
while θ has not converged:
    for t = 1 … n_critic:
        sample real {x_i} and noise {z_i}
        g_w ← ∇_w [ (1/m) Σ f_w(x_i) − (1/m) Σ f_w(g_θ(z_i)) ]
        w ← w + α · RMSProp(w, g_w)          (ascend)
        w ← clip(w, −c, c)
    sample noise {z_i}
    g_θ ← −∇_θ (1/m) Σ f_w(g_θ(z_i))
    θ ← θ − α · RMSProp(θ, g_θ)              (descend)
```

**Key differences from a GAN:**
1. **No sigmoid and no log** in either loss: just the means of the critic scores.
2. **Train the critic more** (n_critic = 5).
   - In a GAN, a too-good discriminator kills G's gradient.
   - In WGAN, a better critic gives a **better** gradient, because W's gradient is meaningful everywhere.
3. **Weight clipping** after each critic step.
4. **RMSProp, not Adam.**
   - Momentum made critic training unstable.
   - When training blew up, the cosine between Adam's step and the gradient turned negative.

---

## 5. Results from the paper

The experiments are on LSUN bedrooms (64×64), against DCGAN as the baseline.

### 5.1 A meaningful loss (Figures 3–4)
- **WGAN:** the critic's estimate of W **decreases steadily** during training, and **lower estimates go with better samples**. This holds for an MLP generator and for a DCGAN generator.
  - When training failed (MLPs with a high learning rate), the curve stayed flat and the samples stayed bad.
- **GAN:** the JS estimate ½L(D, g) + log 2 **stays near log 2 ≈ 0.69 or goes up**, even while samples improve. It doesn't correlate with quality.
- **The paper's own caveat:** the estimate is W only up to an unknown factor K that depends on the critic. So it compares training stages, not different models.
- **Firsts:** this was "the first time in GAN literature" that the loss showed convergence properties.

### 5.2 Stability (Figures 5–7)
| generator | WGAN | standard GAN |
|---|---|---|
| DCGAN | good samples | good samples |
| DCGAN without batch norm, constant filter count | still produces samples | **fails** |
| 4-layer 512-unit MLP | lower quality but sensible | worse, with **mode collapse** |

"In no experiment did we see evidence of mode collapse for the WGAN algorithm."

### 5.3 Why no mode collapse?
- Mode collapse happens when G optimises against a **fixed** discriminator: the best response is to put all mass on the discriminator's favourite point (paper 042's Helvetica scenario).
- WGAN can train its critic **to optimality** without hurting G's gradients.
- An optimal critic accounts for *all* of P_g's mass, so dumping everything on one point is costly.

---

## 6. Why it works (the intuition)

1. **W measures "how far", not just "how different".**
   - JS asks: do the distributions overlap?
   - W asks: how much work would it take to turn one into the other?
   - The second keeps giving a gradient when the answer to the first is "not at all".
2. **The Lipschitz constraint keeps the critic honest.**
   - A GAN discriminator can become a step function: 0 on fakes, 1 on reals, with flat regions in between and therefore zero gradient.
   - A Lipschitz critic can't jump. It must slope gently between the distributions, so it gives a usable gradient everywhere (Figure 2's "very clean gradients").
3. **A better critic means a better generator.** There is no balancing act: train the critic hard.

---

## 7. What our code found

All numbers come from `demo.py` (about 4 s) and `test_wgan.py` (7 tests, about 1.7 s).

1. **Example 1:**
   - W = |θ| while JS = log 2, KL = ∞ and TV = 1 for every θ ≠ 0;
   - a sample check of the straight-across coupling gives exactly |θ|.
2. **Kantorovich–Rubinstein:**
   - primal LP = dual LP = 0.206682;
   - both match the 1-D CDF formula;
   - the optimal f is exactly 1-Lipschitz;
   - the transport plan has the right marginals.
3. **Figures 1–2**, with each judge trained 300 steps at fixed θ:

   | θ | JS estimate | \|∂G-loss/∂θ\| (GAN) | W estimate | \|∂G-loss/∂θ\| (WGAN) |
   |---|---|---|---|---|
   | 2.0 | 0.688 | 9.8·10⁻³ | 0.478 | 0.084 |
   | 1.0 | 0.686 | 0.039 | 0.289 | 0.288 |
   | 0.5 | 0.681 | 0.173 | 0.151 | 0.304 |
   | 0.2 | 0.662 | 1.01 | 0.054 | 0.288 |
   | 0.05 | 0.425 | **11.0** | 0.009 | 0.256 |

   - **JS:** stuck near log 2, and its gradient swings by about 1000×.
   - **W:** the estimate is about 0.27·θ (linear), and the gradient is roughly steady.
   - **Honest note:** at θ = 2 the clipped critic's gradient is smaller (0.084). The critic hadn't fully grown its linear slope in 300 steps.
4. **WGAN learns Example 1:** θ goes 1.0 → 0.47 → 0.19 → −0.05 → −0.001 (steps 0, 25, 50, 100, 300), even though the supports never overlap on the way.
5. **A meaningful loss?** Partly.
   - The critic's estimate tracked the big early drop in true W1 (1.12 → 0.36), with an overall correlation of **0.89**.
   - But after step 200 it hovered around 0 (even negative) while the true W stayed around 0.3–0.4, and the correlation was −0.29.
   - With clipping at c = 0.1 and short critic training, the estimate reports coarse progress, not fine differences. (The paper median-filters its curves, too.)
6. **Clipping:** the spectral-norm Lipschitz bound is 221 / 0.91 / 0.00098 for c = 1 / 0.1 / 0.01; the observed maximum gradients are 13.6 / 0.050 / 0.000068.

**Not run (too heavy):** with CIFAR-10 in place of LSUN,
- E1: Figure 3 curves and samples;
- E2: Figure 4's GAN JS curves;
- E3: Figures 5–7 robustness for three generators;
- E4: Adam vs RMSProp, with the cosine diagnostic;
- E5: the clipping constant.

---

## 8. Check yourself

1. P_r = point mass at 0, P_g = point mass at 5. What are W, JS and KL?
   <details><summary>Answer</summary>W = 5, JS = log 2, KL = ∞. Only W knows the masses are 5 apart, rather than just "different".</details>
2. Why must the critic be Lipschitz?
   <details><summary>Answer</summary>Without a slope limit, the gap E_r[f] − E_g[f] can be made arbitrarily large, so the supremum isn't W (it's ∞). The 1-Lipschitz limit makes the supremum equal to the transport cost.</details>
3. Why is there no sigmoid at the end of the critic?
   <details><summary>Answer</summary>The critic estimates a difference of expectations of an unbounded Lipschitz function, not a probability. A sigmoid would saturate, bringing back flat regions and vanishing gradients.</details>
4. In a standard GAN, why is a too-good discriminator a problem, while in WGAN a better critic is better?
   <details><summary>Answer</summary>The GAN's optimal discriminator gives a JS-based loss that is flat (log 2) for disjoint supports, so G's gradient vanishes. The optimal WGAN critic gives W's gradient, which is informative almost everywhere (Theorem 1), so a better critic means a more accurate gradient.</details>
5. 1-D: P_r is uniform on [0, 1], P_g is uniform on [2, 3]. What is W?
   <details><summary>Answer</summary>Every point shifts by 2, so W = 2. Equivalently, the area between the CDFs: 1 on [1, 2], plus triangles of area ½ on [0, 1] and [2, 3], gives 2.</details>
6. Name one weakness of weight clipping the paper admits.
   <details><summary>Answer</summary>Large c makes the critic slow to train to optimality. Small c causes vanishing gradients in deep networks (the Lipschitz scale shrinks like c^depth).</details>
