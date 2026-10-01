# Sutskever, Vinyals & Le (2014), explained from scratch

**Paper:** *Sequence to Sequence Learning with Neural Networks*
**Authors:** Ilya Sutskever, Oriol Vinyals, Quoc V. Le (Google)
**Published at:** NIPS 2014 (arXiv:1409.3215)

Read Papers 021 (LSTM), 024 (LSTM training) and 026 (encoder–decoder) first. This guide:
- derives the model's 384M parameters;
- proves why reversing the source shortens the *minimum* time lag;
- works beam search on a tiny example where greedy decoding fails.

---

## 0. The whole idea in one line

> **A plain, big, deep LSTM can translate whole sentences by itself: one LSTM reads the source sentence into a fixed vector, another LSTM writes the translation from it. With the source sentence reversed, this "Seq2Seq" model beat a phrase-based statistical MT system on a large task for the first time.**

---

## 1. Why sequences are hard for neural nets (Section 1)

- **The mismatch:** classic deep nets need inputs and outputs of **fixed size**, but sentences have **different, unknown lengths**, and the alignment between source and translation words is complicated and non-monotonic.
- **The idea (Figure 1):**
  - an LSTM reads the input one token at a time into a **fixed-dimensional vector v**;
  - a second LSTM, a language model **conditioned on v**, writes the output.

---

## 2. The model (Section 2)

```
p(y_1 … y_T′ | x_1 … x_T) = Π_t p(y_t | v, y_1 … y_{t−1})          (1)
```
- **v** is the encoder's final state. Each factor is a softmax over the target vocabulary.
- **Every sentence ends with <EOS>.** Producing <EOS> is how the model **chooses the length**. Without it, (1) couldn't define a probability over outputs of different lengths.

**Three choices that mattered:**
1. **Two different LSTMs** for input and output: more parameters for almost no extra computation per step.
2. **Deep LSTMs** (4 layers): "each additional layer reduced perplexity by nearly 10%".
3. **Reverse the source:** map c, b, a → α, β, γ instead of a, b, c → α, β, γ (section 3).

### 2.1 The size, derived (V_src = 160,000, V_tgt = 80,000, 4 layers × 1000 cells)
```
one LSTM layer:  4 gates × (1000 input + 1000 recurrent) × 1000 + 4×1000 biases = 8,004,000
4 layers:        32.0M per LSTM  → 64M for the encoder + decoder
embeddings:      160,000 × 1000 = 160M (source) + 80,000 × 1000 = 80M (target)
softmax:         1000 × 80,000 + 80,000 = 80.08M
total:           ≈ 384.1M          (the paper: 384M)
```
- **Where the parameters are:** **~84% are in the word tables** (embeddings and softmax), not the LSTMs.
- **The "sentence vector"** is the full final state: 4 layers × (h + c) × 1000 = **8,000 numbers**.

---

## 3. Reversing the source (Section 3.3)

**The result:** test perplexity 5.8 → **4.7**, BLEU 25.9 → **30.6**.

### 3.1 The lag arithmetic
- **Think of a sentence** where source word i translates to target word i, with T words each.
- **Forward reading:** source word i is read at step i; target word i is produced at step T + i. The lag is **(T + i) − i = T for every word**.
- **Reversed reading:** source word i is read at step T − i + 1. The lag is (T + i) − (T − i + 1) = **2i − 1**.
  - word 1 has lag **1**, word 2 has 3, …, word T has 2T − 1;
  - the mean lag is (1/T)·Σ(2i − 1) = **T**, the same as before.

| sentence length | forward lags (min / mean) | reversed lags (min / mean) |
|---|---|---|
| 5 | 5 / 5.0 | **1** / 5.0 |
| 20 | 20 / 20.0 | **1** / 20.0 |
| 50 | 50 / 50.0 | **1** / 50.0 |

### 3.2 Why that helps
- **The early words become easy:** a lag of 1 is trivially learnable (Paper 021: short lags are easy and long ones hard). SGD can "establish communication" between source and target early, and the network then builds on it.
- **The authors found it also helped later in the sentence** and guess it improves memory use overall.
- **Our toy** (copy a 12-symbol sentence, 300 steps): loss **1.76 reversed vs 2.21 forward**.

---

## 4. Training (Section 3.4)

- **The objective:** maximize (1/|S|) Σ log p(T | S) over the training pairs, with teacher forcing (Paper 026, section 1.4).
- **Data:** WMT'14 English → French, a 12M-sentence "selected" subset (348M French and 304M English words). Out-of-vocabulary words → UNK.
- **Init:** uniform in [−0.08, 0.08].
- **Optimizer:** **plain SGD without momentum**, lr 0.7. After 5 epochs, halve the lr every half epoch; 7.5 epochs in total.
- **Batches:** 128 sentences (the gradient divided by 128).
- **Clipping:** if s = ‖g‖ > 5, then g ← 5g/s. LSTMs can still have *exploding* gradients (Paper 024, section 4.2).
- **Length bucketing:** batch sentences of similar length together, so short sentences don't wait for padding. That gives a **2× speed-up**.
- **Hardware:** 8 GPUs (one per LSTM layer, 4 for the softmax), 6,300 words/s, about **10 days**.

---

## 5. Decoding: beam search (Section 3.2)

**The goal:** find the most likely translation, argmax_y p(y | x).
- **Why you can't just try everything:** with an 80,000-word vocabulary there are 80,000ᵀ candidates of length T. That's 10⁴⁹ for T = 10.
- **Greedy decoding** (pick the best word each step) is fast, but it can be fooled. An early word that looks best may lead only to poor continuations.

### 5.1 Beam search
- **The algorithm:**
  - keep the **B most probable partial translations** (the "beam");
  - at each step, extend each one with every possible next word, and keep the best B by total log-probability;
  - a hypothesis that emits **<EOS>** moves to the "finished" set;
  - stop when no live hypothesis can beat the best finished one.
- **B = 1 is greedy.**

**A tiny example where greedy fails:**
```
step 1:  P(A) = 0.6   P(B) = 0.4
step 2:  after A:  P(x) = 0.5, P(y) = 0.5         after B:  P(z) = 0.9, ...
greedy:     A then x  →  0.6 × 0.5 = 0.30
beam (B=2): keeps A and B; finds B then z → 0.4 × 0.9 = 0.36  ← better
```
**Our tests:** on a constructed example where greedy is fooled, a wide beam finds the **exact best** sequence (confirmed by exhaustive search).

**The paper's finding:** beam 1 already works well, and **beam 2 gets most of the benefit**.

### 5.2 Ensembles and rescoring
- **Ensemble decoding:** **average the probabilities** of several models at every step.
- **Cost:** 5 models with beam 2 costs less than 1 model with beam 12, and does better (Table 1).
- **Rescoring:** take a baseline SMT system's **1000-best** lists and re-rank them by "an even average" of the SMT score and the LSTM's log-probability.

### 5.3 The short-output bias (our demo's lesson)
- **The bias:** beam search maximizes a **sum** of log-probabilities. Each extra word adds a negative number, so shorter outputs have an advantage.
- **Our undertrained toy model, at beam 12,** returned the **empty** sentence: log p = −6.5, against about −18 for a full output.
- **Why the paper didn't suffer:** a well-trained model makes an early <EOS> very improbable.
- **The fix in later systems** (Wu et al. 2016) is **length normalization**: rank by log p / |y|^α instead of log p.

---

## 6. Results (Section 3.6)

### Table 1: direct translation (test BLEU, ntst14)
| Method | BLEU |
|---|---|
| Bahdanau et al. (attention, Paper 028, early version) | 28.45 |
| baseline phrase-based SMT | 33.30 |
| single **forward** LSTM, beam 12 | 26.17 |
| single **reversed** LSTM, beam 12 | 30.59 |
| 5 reversed, beam 1 | 33.00 |
| 5 reversed, beam 2 | 34.50 |
| **5 reversed, beam 12** | **34.81** |

**The first pure neural system to beat a phrase-based SMT system on a large MT task,** despite being unable to translate words outside its 80k vocabulary.

### Table 2: rescoring the SMT 1000-best lists
| Method | BLEU |
|---|---|
| baseline | 33.30 |
| Cho et al. (Paper 026) | 34.54 |
| **rescored by 5 reversed LSTMs** | **36.5** |
| best WMT'14 system | 37.0 |
| oracle (best pick from the 1000-best) | ~45 |

### Long sentences (Section 3.7, Figure 3)
- **No degradation** up to ~35 words, contrary to others' experience. It drops only for very rare words.

### Sentence representations (Section 3.8, Figure 2)
A 2-D PCA of the encoder's state shows it is:
- **sensitive to word order:** "John admires Mary" ≠ "Mary admires John";
- **fairly invariant to voice:** "I was given a card by her" ≈ "She gave me a card".

---

## 7. Why it matters

- **"Seq2Seq" became the standard framework** for translation, summarization, dialogue, speech and parsing (Paper 030).
- **The one-vector bottleneck** was removed by **attention** (Paper 028), and later by the **Transformer**.
- **The recipe** (deep LSTMs, clipping, length bucketing, beam search, ensembles) became standard practice.

---

## 8. What our code found

**Scale note:**
- At your request, nothing was trained on this laptop.
- `experiments.py` reproduces, on Multi30k English → German:
  - E1: reversal (perplexity and BLEU);
  - E2: beam 1/2/12 × single model vs ensemble;
  - E3: BLEU by sentence length;
  - E4: PCA of sentence vectors for word-order and voice variants;
  - E5: n-best rescoring with a second model.

**Checked (tests and demo, a few seconds):**
- **The size:** **384.1M** (160M + 80M embeddings, 32M per LSTM, an 80.1M softmax); the sentence vector is exactly **8,000** numbers.
- **The reversal lags:** min 20 → 1 for a 20-word sentence, with the mean unchanged. On the toy copy task the loss is 1.76 reversed vs 2.21 forward.
- **Beam search:**
  - a wide beam finds the exact best sequence where greedy is fooled;
  - an ensemble of a model with itself equals the model;
  - beam scores equal teacher-forced log-probabilities.
- **The details:** the lr schedule (0.7, then halving every half epoch after epoch 5), clipping to exactly norm 5, length-bucketed batches, "even average" rescoring.
- **The short-output bias:** see section 5.3.

---

## 9. Check yourself

1. Write Eq. (1). What is v? Why must every sentence end with <EOS>?
2. Derive the 32M parameters per 4-layer LSTM, and the 384M total.
3. Show that reversing gives lags 2i − 1. What are the minimum and mean lags for T = 10?
4. Why does a lag-1 pair make learning easier (Paper 021)?
5. Run beam search with B = 2 on section 5.1's example. What does greedy return?
6. Why can 5 models with beam 2 be cheaper and better than 1 model with beam 12?
7. Why does a sum of log-probabilities favour short translations? Write the length-normalized score.
8. What did Figure 2 show about the sentence vectors?
