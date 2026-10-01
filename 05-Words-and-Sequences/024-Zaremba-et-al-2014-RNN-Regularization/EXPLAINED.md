# Zaremba, Sutskever & Vinyals (2014), explained from scratch

**Paper:** *Recurrent Neural Network Regularization*
**Authors:** Wojciech Zaremba, Ilya Sutskever, Oriol Vinyals (NYU / Google Brain)
**Published:** arXiv:1409.2329 (2014)

Read Paper 010 (dropout) and Paper 021 (LSTM) first. This guide:
- counts exactly how often dropout hits a memory under each scheme;
- derives the model sizes;
- explains perplexity, truncated BPTT, gradient clipping and ensembling with small numbers.

---

## 0. The whole idea in one line

> **Dropout didn't work for RNNs because people applied it to the recurrent connections, where the noise piles up over time and destroys memory. Apply it only to the vertical, layer-to-layer connections, and dropout works for LSTMs, which makes large LSTMs usable without overfitting.**

---

## 1. The problem (Sections 1–2)

- **Large RNNs overfit** language data, so people used small ones (the best unregularized PTB model had only 200 units).
- **Dropout** (Paper 010) is the best regularizer for feed-forward nets, but "does not work well with RNNs". Bayer et al. (2013) argued that the **recurrence amplifies the noise**.

---

## 2. The LSTM used (Section 3.1)

This is the modern form (Graves 2013), **with a forget gate** (added by Gers et al. 2000 to Paper 021's cell). One affine map T reads the input from below and the layer's own previous output:
```
[i; f; o; g] = [σ; σ; σ; tanh]( T_{2n,4n} [h^{l−1}_t ; h^l_{t−1}] )
c^l_t = f ⊙ c^l_{t−1} + i ⊙ g              memory: keep a fraction f, write i·g
h^l_t = o ⊙ tanh(c^l_t)                    output: a gated view of the memory
```
- **Inputs and outputs:** h⁰_t is the word embedding, and h^L_t feeds the softmax that predicts the next word.
- **The forget gate's effect on the error:** ∂c_t/∂c_{t−1} = f_t. With f ≈ 1 the memory (and its gradient) flows through almost unchanged (Paper 021's CEC). The forget gate lets the network *choose* to clear it.

### 2.1 Model sizes, derived (vocabulary V = 10,000, 2 layers of n units)
```
embedding        V·n
each LSTM layer  4·(2n)·n + 4n          (T maps 2n inputs → 4n gate pre-activations, plus biases)
softmax          n·V + V
```
| Model | n | embedding | 2 LSTM layers | softmax | **Total** |
|---|---|---|---|---|---|
| small (non-regularized) | 200 | 2.0M | 0.64M | 2.01M | **4.7M** |
| **medium** | 650 | 6.5M | 6.77M | 6.51M | **19.8M** |
| **large** | 1500 | 15.0M | 36.0M | 15.0M | **66.0M** |

These are the classic "20M / 66M" PTB models (our code counts the same).

---

## 3. The recipe: dropout only on non-recurrent connections (Section 3.2, Figure 2)

```
[i; f; o; g] = [σ; σ; σ; tanh]( T [ D(h^{l−1}_t) ; h^l_{t−1} ] )
```
- **Where D goes:** the dropout operator D(·) is applied to the input **from the layer below**, including the embedding and the softmax input.
- **Where it never goes:** **never** on h^l_{t−1}, the recurrent connection.

### 3.1 Why it works: count the corruptions (Figure 3)
- **Follow one piece of information** from the word at time t − k to the prediction at time t. It moves:
  - **up** L + 1 times: embedding → layer 1 → … → layer L → softmax;
  - **sideways** k times, through the recurrent connections.
- **Under the paper's recipe,** only the upward moves pass through D. So the information is corrupted exactly **L + 1 times**, whatever k is.
- **Under naive dropout** (on the recurrent connections too), it's hit **L + 1 + (k − 1)** times.

| how long ago | paper's recipe (L = 2) | naive |
|---|---|---|
| 1 step | 3 | 4 |
| 10 steps | 3 | 13 |
| 100 steps | 3 | 103 |
| 1000 steps | 3 | **1003** |

### 3.2 Survival
- **The arithmetic:** with keep-probability p = 0.5, one particular unit's signal survives k independent dropouts with probability 0.5ᵏ.
  - 10 steps gives 0.5¹⁰ ≈ **0.001**;
  - under the paper's recipe the survival stays at 0.5^{L+1} = 0.125, **however long ago**.
- **The real effect is softer** (memories are spread over many units), but the trend is the point: **naive dropout erases long memories step by step**, while the paper's recipe gives memory a noise-free path through time.

---

## 4. Training details

### 4.1 Truncated BPTT with state carry-over
- **The setup:** PTB is one long text, cut into 20 parallel streams (batch 20), and each update processes **35** consecutive words per stream.
- **The gradient** flows back at most 35 steps ("truncated BPTT").
- **The state carries over:** the final hidden state of one minibatch is the **initial** state of the next, so the *forward* memory can span the whole text even though gradients don't.

### 4.2 Gradient-norm clipping
```
if ‖g‖ > c:   g ← c · g / ‖g‖            (c = 5 for medium, 10 for large)
```
- **What it does:** it keeps the direction and caps the length.
- **Why:** RNN gradients occasionally explode (Paper 021, section 2), and one huge step can wreck training. Clipping makes those steps harmless.

### 4.3 Learning-rate schedules
| model | init | learning rate | epochs |
|---|---|---|---|
| **medium** | U[−0.05, 0.05] | 1, ÷1.2 per epoch after epoch 6 (ending at 1/1.2³³ ≈ 0.0024) | 39 (~½ day on a K20) |
| **large** | U[−0.04, 0.04] | ÷1.15 after epoch 14 | 55 (~1 day) |
| **non-regularized** | | halved per epoch after epoch 4, unroll 20 | 13 |

Medium also uses clip 5; large uses clip 10.

---

## 5. Experiments (Section 4)

### 5.1 Perplexity: how language models are scored
```
perplexity = exp( −(1/N) Σ_t log p(w_t | w_<t) )          (the exponential of the average loss per word)
```
- **The meaning:** the model is, on average, as uncertain as a **uniform choice among that many words**.
  - A model that always gives the correct word probability 1/83 has perplexity exactly **83**.
  - A uniform model over V words has perplexity V (our test: an untrained all-zero model gets exactly 10,000).
- **Relation to Paper 022:** the same idea as bits per character, but per word and with base e instead of 2.

### 5.2 Penn Treebank (Table 1)
**Data:** 929k / 73k / 82k words, 10k vocabulary. Models are 2-layer LSTMs.

| Model | Units | Dropout | Valid | Test perplexity |
|---|---|---|---|---|
| non-regularized (best size found) | 200 | – | 120.7 | 114.5 |
| **medium** regularized | 650 | 50% | 86.2 | **82.7** |
| **large** regularized | 1500 | 65% | 82.2 | **78.4** |
| 10 non-regularized, averaged | 200 | – | 83.5 | 80.0 |
| 10 medium regularized | 650 | 50% | 75.2 | 72.0 |
| 38 large regularized | 1500 | 65% | 71.9 | **68.7** |

- **Without regularization, bigger nets overfit,** which "effectively constrains" you to 200 units.
- **With dropout, the 66M model wins:** 114.5 → 78.4.

**Why averaging models helps perplexity:**
- **The method:** an ensemble averages the *probabilities* of K models, p̄ = (1/K)Σ p_k.
- **The guarantee:** −log is convex, so −log p̄ ≤ (1/K)Σ(−log p_k) (Jensen's inequality). The ensemble's loss is never worse than the average model's.
- **In practice it's much better,** because the models make different mistakes.

### 5.3 Speech (Table 2): Icelandic, 93k utterances, frame accuracy
| | Training | Validation |
|---|---|---|
| non-regularized | 71.6% | 68.9% |
| regularized | 69.4% | **70.5%** |

Training accuracy **drops** (noise during training) while validation accuracy **rises**: the classic dropout signature.

### 5.4 Machine translation (Table 3): WMT'14 English → French
The model is 4 layers of 1000 units; dropout 0.2 was best.

| | Test perplexity | BLEU |
|---|---|---|
| non-regularized | 5.8 | 25.9 |
| regularized | **5.0** | **29.03** |
| LIUM (phrase-based) | – | 33.30 |

### 5.5 Image captioning (Table 4): MSCOCO, Show and Tell (Paper 029)
| | Test perplexity | BLEU |
|---|---|---|
| non-regularized | 8.47 | 23.5 |
| regularized | 7.99 | 24.3 |
| 10 non-regularized (ensemble) | 7.5 | 24.4 |

**One regularized model ≈ an ensemble of 10.**

---

## 6. Why it matters

- **The standard LSTM regularizer for years:** the PTB "medium" and "large" models were the benchmark baselines.
- **Later work:**
  - **variational dropout** (Gal & Ghahramani 2016) reuses **one mask for all time steps**, so it *can* be applied to recurrent connections without the step-by-step erasure;
  - **AWD-LSTM** (Merity et al. 2017) adds DropConnect on the recurrent weights.

---

## 7. What our code found

**Scale note:**
- At your request, nothing was trained on this laptop.
- `experiments.py` uses PTB from the paper's own GitHub repository and reproduces:
  - E1: Table 1's three single models with the exact recipes;
  - E2: the "where to put dropout" ablation (none / non-recurrent / naive);
  - E3: model averaging;
  - E4: samples after "the meaning of life is".

**Checked (tests and demo, about a second):**
- **Our hand-written LSTM cell = `torch.nn.LSTMCell`,** once the gate order is mapped (ours i, f, o, g; PyTorch i, f, g, o).
- **Sizes:** small 4.7M, **medium 19.8M, large 66.0M** (derived in section 2.1).
- **The recipe really is non-recurrent:**
  - "nonrecurrent" mode zeroes ~50% of each vertical input and alters **0%** of the recurrent inputs;
  - "naive" mode alters the recurrent inputs too (zeroing them, or rescaling the kept ones by 1/(1 − p)).
- **Figure 3's count:** 3 corruptions for any lag under the recipe; 4 / 13 / 103 / 1003 under naive dropout.
- **No dropout at test time:** eval outputs are deterministic.
- **Truncated BPTT works:**
  - an all-zero model has perplexity exactly V;
  - a repeating stream trains below 1.5;
  - an "ensemble" of one model equals the model.

---

## 8. Check yourself

1. Why did people think dropout doesn't work for RNNs?
2. Write the LSTM equations with the paper's dropout. Which input gets D(·)?
3. With 2 layers, how often is information from 1000 steps ago corrupted under the recipe? Under naive dropout?
4. With keep-probability 0.5, what is 0.5¹⁰? What does that say about naive dropout?
5. Derive the medium model's 19.8M parameters.
6. Define perplexity. What perplexity does a model get if it always gives the right word probability 1/50?
7. Why does the final hidden state carry over between minibatches, while gradients stop after 35 steps?
8. Why can an ensemble's perplexity never be worse than its average member's?
