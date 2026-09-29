# Hinton et al. (2012), explained simply

**Paper:** *Improving neural networks by preventing co-adaptation of feature detectors*
**Authors:** Geoffrey Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov
**Published:** arXiv:1207.0580, July 2012 (18 pages: 6 pages of main text plus appendices)

This is the **first dropout paper**. Paper 010 is the full, detailed journal version.

---

## The big idea in one line

> **While training, randomly switch off half of the hidden units for every example. No unit can then rely on particular other units being there, so each learns something useful on its own. At test time, use all the units with their outgoing weights halved. That's the average of an enormous number of networks, computed in one pass.**

---

## 1. The problem: co-adaptation and overfitting (page 1)

- A big network trained on little data finds **many** weight settings that fit the training set almost perfectly. Most of them do badly on new data.
- **Why:** feature detectors become **co-adapted**. A unit is only useful **together with** several specific other units. Those combinations fit the training data but don't generalize.

---

## 2. Dropout (pages 1–2)

- **For every training case:** drop each hidden unit with probability **0.5**, meaning its output is set to 0 for that case.
- **Inputs:** often drop **20%** of them too.
- A unit can't count on any particular partner being present, so it must learn a feature that's **useful in many contexts**.

**Another view: a huge ensemble.**
- With N hidden units there are **2ᴺ possible "thinned" networks**.
- Every training case trains a different one, but they all **share the same weights**.
- Training many separate networks and averaging them is a classic way to improve accuracy. Dropout does it almost for free.

### Max-norm instead of weight decay (page 2)
- Rather than penalizing large weights (L2), **cap** the length of each hidden unit's incoming weight vector: squared length ≤ l (l = 15). If an update breaks the cap, scale the vector back down.
- **Why:** weights can't blow up however big the step, so you can start with a **very large learning rate** that decays. That lets training explore much more of the weight space.

### At test time: the "mean network" (page 2)
- Use **all** units, but **halve their outgoing weights**. Twice as many units are active as during training, so halving keeps the next layer's input the same size on average.
- **An exact fact:** with **one hidden layer and a softmax output**, the mean network gives **exactly the normalized geometric mean** of the predictions of all 2ᴺ thinned networks.
- **And:** the mean network gives the correct answer a **higher log-probability** than the average log-probability of the individual thinned networks.

---

## 3. Results (pages 2–5)

| Task | Without dropout | With dropout |
|---|---|---|
| **MNIST** (plain nets, no tricks) | ~160 errors (best published) | ~130 (50% hidden), **~110** (+20% input) |
| MNIST, pretrained Deep Belief Net | 118 errors | 92 |
| MNIST, pretrained Deep Boltzmann Machine | ~94 | **79** (record) |
| TIMIT speech (frame recognition) | 22.7% | 19.7% |
| CIFAR-10 (conv net) | 16.6% | 15.6% |
| ImageNet 2010 | 48.6% | **42.4%** (record) |
| Reuters text | 31.05% | 29.62% |

**MNIST setup (Appendix A.1):**
- architectures: 784-800-800-10 and larger;
- 50% dropout on hidden units, 20% on inputs;
- minibatches of 100, cross-entropy;
- learning rate starting at **10**, × 0.998 each epoch;
- momentum rising 0.5 → 0.99 over 500 epochs, with the gradient multiplied by (1 − momentum);
- max-norm with squared length ≤ 15;
- initial weights N(0, 0.01²);
- **3,000 epochs**.

---

## 4. Why it works: more views (pages 5–6)

- **Dropout probabilities:** 0.5 for hidden units works well almost everywhere. For inputs, keep more than 50%.
- **vs Bayesian averaging:** Bayesian methods weight each model by how good it is, which is expensive. Dropout gives all models equal weight, which is cheap.
- **vs bagging:** it's like bagging (training models on different data subsets), taken to the extreme. Each model sees **one** example, and the weight sharing is a strong regularizer.
- **Naive Bayes** is an extreme form of dropout: every feature is trained alone.
- **Sex in evolution:** mixing genes breaks up co-adapted gene sets, so functions get done in several robust ways, just as dropout breaks up co-adapted units.

---

## 5. What the features look like (Appendix A.3, Figure 5)

First-layer features of a 784-500-500 net:
- **Standard backprop:** noisy and hard to interpret.
- **Dropout:** clean, **stroke-like** detectors.

This supports the idea that dropout makes each unit learn a meaningful feature on its own.

---

## 6. What our code found

**Scale note:** at your request, the MNIST experiments were **not fully run** on this laptop. `experiments.py` runs them (about 25 minutes on a GPU; the paper used 3,000 epochs, the script 100).

**Checked exactly:**
- **The mean network = the normalized geometric mean of all 2ᴺ thinned networks.** Enumerating all **1,024** sub-networks of a 10-hidden-unit net, the difference is **1×10⁻¹⁶**.
- **The mean network gives the correct class a higher log-probability** than the average thinned network: **−1.80 vs −1.99** in one test, **−1.88 vs −2.17** in another.

**Partial MNIST results before stopping** (784-800-800-10, 100 epochs, the paper's time-compressed schedule):

| method | test errors (of 10,000) |
|---|---|
| standard backprop, best learning rate (3.0) | **171** (paper: ~160 best published) |
| max-norm only, no dropout | 225 |
| dropout runs | not finished |

**A demo run** (20,000 training images, 15 epochs): standard backprop got **620** test errors and dropout **500**. The gap between test and training error rates was 1.9 points for standard backprop and 1.1 for dropout.

**A detail the paper leaves implicit:** without max-norm, its learning rate of 10 makes a plain network blow up to chance level (about 9,000 errors). Max-norm is what makes those huge rates safe, so the no-dropout baseline needs its own, smaller learning rate.

---

## 7. Check yourself

1. What is "co-adaptation", and why does it hurt generalization?
2. During training, what exactly happens to a dropped unit?
3. Why are the outgoing weights halved at test time?
4. Why is the mean network exactly the geometric mean when there's one hidden layer and a softmax?
5. Why does max-norm let you use a huge learning rate?
6. How is dropout like bagging, and how is it different?
