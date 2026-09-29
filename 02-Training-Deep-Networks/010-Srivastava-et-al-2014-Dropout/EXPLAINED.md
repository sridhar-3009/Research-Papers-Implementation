# Srivastava et al. (2014), explained simply

**Paper:** *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*
**Authors:** Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov
**Published in:** Journal of Machine Learning Research 15 (2014), pp. 1929–1958
**Page numbers** below are the journal's (1929–1958).

This is the **full version** of Paper 009. Read 009 first for the core idea; this explains the extras.

---

## The big idea in one line

> **Train a large network while randomly dropping units, so it becomes an ensemble of exponentially many thinned networks sharing weights. At test time, scale the weights to average them all. It's one of the most effective and general regularizers ever found.**

---

## 1. Motivation (pages 1929–1931)

- Big networks overfit. The best cure, **averaging many different models**, is too expensive for neural nets.
- **Dropout** approximately averages **2ⁿ** thinned networks, at the cost of training just one.
- **The sex analogy:** genes that must work with *any* random partner genes become robust individually. Hidden units under dropout do the same.

---

## 2. The model (Section 4, pages 1933–1934)

Standard layer:
```
z = W·y + b,     y_next = f(z)
```
Dropout layer:
```
r ~ Bernoulli(p)          one random 0/1 per unit, independently
ỹ = r * y                 thinned outputs
z = W·ỹ + b,  y_next = f(z)
```
- **p = the probability of KEEPING a unit.** The paper uses p = 0.5 for hidden units and 0.8 for inputs.
- **Test time:** use W_test = p·W, so each unit's expected output matches training.

---

## 3. Training (Section 5, pages 1934–1935)

- Normal SGD. Each training case gets its own thinned network, and a unit that was dropped contributes a **zero gradient** for that case.
- **Max-norm regularization:** keep each hidden unit's incoming weights inside a ball, ‖w‖ ≤ c, and project back whenever an update leaves it.
- **Together:** max-norm + **large decaying learning rate** + **high momentum**. The ball stops the weights blowing up, the noise lets training explore, and the decaying rate lets it settle.
- **Pretrained nets:** scale the pretrained weights **up by 1/p** before dropout fine-tuning, and use a smaller learning rate.

---

## 4. Results across domains (Section 6, pages 1935–1943)

| Data set | Domain | Best without dropout | With dropout |
|---|---|---|---|
| **MNIST** | digits | 1.60% | **0.95%** (8192-unit layers, max-norm) |
| SVHN | house numbers | 3.95% | **2.55%** |
| CIFAR-10 | photos | 14.98% | **12.61%** |
| CIFAR-100 | photos | 43.48% | **37.20%** |
| ImageNet (ILSVRC-2012) | photos | 26.2% (top-5) | **16.4%** |
| TIMIT | speech | 23.4% | **21.8%** |
| Reuters | text | 31.05% | 29.62% |

**MNIST details (Table 2):**

| Network | Error |
|---|---|
| standard net | 1.60% |
| dropout, logistic | 1.35% |
| dropout, ReLU | 1.25% |
| dropout + max-norm, ReLU | **1.06%** |

A net with 65 million weights trained on 60,000 images doesn't overfit with dropout, and **doesn't even need early stopping**.

- **Robust to architecture (Figure 4):** the same hyperparameters work across many network shapes.
- **Bayesian neural nets** do slightly better on a tiny genetics data set, but they're much slower.

### Table 9: dropout vs other regularizers (MNIST, 784-1024-1024-2048-10, ReLU)
| Method | Error |
|---|---|
| L2 | 1.62% |
| L2 + L1 | 1.60% |
| L2 + KL-sparsity | 1.55% |
| max-norm | 1.35% |
| dropout + L2 | 1.25% |
| **dropout + max-norm** | **1.05%** |

---

## 5. Why it works: the "salient features" (Section 7, pages 1943–1947)

### 7.1 Features
In a 256-unit ReLU autoencoder on MNIST:
- **without** dropout, the features are co-adapted mush;
- **with** dropout, they are clean **edges, strokes and spots** (Figure 7).

### 7.2 Sparsity
Dropout makes the hidden activations **sparse** without any sparsity penalty. Mean activation is about **2.0** without dropout and about **0.7** with it, and most activations sit near 0 (Figure 8).

### 7.3 The dropout rate p (Figure 9)
- **Fixed n** (same width, 784-2048×3-10): very small p **underfits**. The error is flat for 0.4 ≤ p ≤ 0.8 and rises again near 1 (no dropout).
- **Fixed p·n** (widen the layers as p drops, so about the same number of units stays active): small p hurts much less; for p = 0.1 the error falls from 2.7% to 1.7%. p ≈ 0.6 is best, and 0.5 is close.

### 7.4 Data set size (Figure 10)
- With **100 or 500** examples, dropout doesn't help: the net memorizes anyway.
- With **medium** amounts, it helps **a lot**.
- With **very large** data, the gain shrinks, because overfitting was less of a problem anyway.
- So there is a "sweet spot".

### 7.5 Monte-Carlo averaging vs weight scaling (Figure 11)
- Averaging the predictions of k sampled thinned nets matches the one-pass weight-scaling method at about **k = 50**. Beyond that it's only slightly better.
- **Weight scaling is a good approximation**, and it's k times cheaper.

---

## 6. Dropout RBMs (Section 8, pages 1947–1949)

Dropout also works for Restricted Boltzmann Machines: drop hidden units while training with CD-1. The features become coarser, there are fewer dead units, and the representations are sparser.

(Not implemented here: RBMs are a separate topic.)

---

## 7. Marginalizing dropout (Section 9, pages 1949–1950)

### 9.1 Linear regression: dropout = ridge regression
Drop inputs with keep probability p. The **expected** loss is:
```
E ‖y − (R*X) w‖²  =  ‖y − pXw‖²  +  p(1 − p) ‖Γw‖²,      Γ = diag(XᵀX)^½
```
- That's **ridge regression**, where each weight's penalty is scaled by how much its input varies.
- With w̃ = pw, the penalty constant is **(1 − p)/p**. **Less keeping** (smaller p) means **stronger shrinkage**.

### 9.2 Logistic regression and deep nets
There's no exact closed form. Approximate Gaussian marginalization (Wang & Manning, 2013) works for logistic regression, but it doesn't carry over well to deep nets.

---

## 8. Gaussian dropout (Section 10, pages 1950–1951)

- Multiply each unit by **r ~ N(1, σ²)** instead of a 0/1 mask.
- With σ² = (1 − p)/p it has the **same mean and variance** as Bernoulli dropout in its "scale by 1/p during training" form (inverted dropout).
- The mean is 1, so there's **no scaling at test time**.
- It works as well or **slightly better** (MNIST: Bernoulli 1.08% vs Gaussian **0.95%**; CIFAR-10: 12.6% vs 12.5%).

---

## 9. Conclusion and practical tips (Section 11 and Appendix A, pages 1951–1954)

- **Downside:** training takes **2–3× longer**, because the gradients are noisy.
- **Network size:** if n units is right without dropout, use at least **n/p** with it.
- **Learning rate:** 10–100× larger than usual, with momentum **0.95–0.99**.
- **Max-norm:** c is typically **3–4**.
- **p:** 0.5 for hidden layers; 0.8 for real-valued inputs.

---

## 10. What our code found

**Scale note:** at your request, the MNIST experiments were **not run** on this laptop. `experiments.py` reproduces Table 9 and Figures 7–11 plus Table 10 (about an hour on a GPU). Compare against the paper's numbers above.

**Checked (small computations):**
- **Dropout in linear regression is ridge regression (Section 9.1).**
  - Training with **actual random dropout masks** by SGD lands on the same weights as the closed-form ridge solution: [0.84, −2.01, 0.01, 1.01, −0.02] vs [0.88, −2.01, 0.02, 1.01, −0.06].
  - On the dropout objective, both beat ordinary least squares (1,610 vs 1,624).
- The tests also check:
  - Bernoulli test-time scaling = the expected training input;
  - Gaussian dropout has mean 1 and variance (1 − p)/p, and needs no test-time scaling;
  - max-norm projection;
  - the ridge solution has zero gradient;
  - smaller p shrinks the weights more.

---

## 11. Check yourself

1. Write the dropout layer equations. What is p, and what happens at test time?
2. Why do max-norm and a large learning rate work well together?
3. Explain Figure 9: why is "p·n fixed" better than "n fixed" for small p?
4. Why doesn't dropout help with 100 training examples?
5. Derive the ridge-regression form for dropout in linear regression. What does Γ do?
6. Why doesn't Gaussian dropout need test-time scaling?
