# Cho et al. (2014), explained from scratch

**Paper:** *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*
**Authors:** Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
**Published at:** EMNLP 2014 (arXiv:1406.1078)

Read Paper 021 (LSTM) first. This paper introduces two things everyone uses: the **encoder–decoder** and the **GRU**. This guide:
- runs a GRU step by hand;
- computes BLEU on a real sentence;
- explains the paper's training pieces (maxout, orthogonal init, AdaDelta, low-rank embeddings) with numbers.

---

## 0. The whole idea in one line

> **One RNN (the encoder) reads a variable-length sentence and squeezes it into a fixed-length vector c. A second RNN (the decoder) generates the translation from c. Both are trained together to maximize log p(target | source). The paper also introduces a simpler gated unit, the GRU.**

---

## 1. The RNN Encoder–Decoder (Section 2.2, Figure 1)

### 1.1 The probability being modelled
- **The chain rule** (Paper 031, Eq. 1) splits the probability of a whole target sentence y = (y₁ … y_{T′}) into next-word predictions:
  ```
  p(y | x) = Π_t P(y_t | y_{t−1}, …, y₁, x)
  ```
- **The decoder** only has to answer "what's the next word, given the source and the words so far?".

### 1.2 Encoder
Read x₁ … x_T. The final hidden state summarizes the whole sentence:
```
h_t = f(h_{t−1}, x_t)            c = tanh(V h_T)          (Appendix A.1)
```

### 1.3 Decoder
Another RNN whose state and outputs **see c at every step**:
```
h′₀ = tanh(V′ c)
h′_t = f(h′_{t−1}, y_{t−1}, c)
P(y_t | y_{<t}, c) = g(h′_t, y_{t−1}, c)
```

### 1.4 Training (Eq. 4) and the two uses
- **The objective:** max_θ (1/N) Σ_n log p_θ(y_n | x_n), i.e. cross-entropy summed over target words.
- **Teacher forcing:** during training the decoder is fed the **true** previous word y_{t−1}, not its own guess.
- **Two uses:**
  1. **generate** a translation;
  2. **score** a given pair, p(y | x). **This paper uses (2).**

---

## 2. The GRU: "a hidden unit that adaptively remembers and forgets" (Section 2.3)

For hidden unit j:
```
r_j = σ([W_r x]_j + [U_r h_{t−1}]_j)                (5)  reset gate
z_j = σ([W_z x]_j + [U_z h_{t−1}]_j)                (6)  update gate
h̃_j = tanh([W x]_j + [U (r ⊙ h_{t−1})]_j)           (8)  candidate state
h_j = z_j h_{t−1,j} + (1 − z_j) h̃_j                  (7)  new state: mix old and candidate
```

### 2.1 One step by hand (one unit)
**Setup:** h_{t−1} = 0.9, [Wx] = 0.5, U = 1, z = 0.2.
- **With r = 1:** h̃ = tanh(0.5 + 0.9) = tanh(1.4) = 0.885, so h = 0.2·0.9 + 0.8·0.885 = **0.888**.
- **With r = 0 (reset):** h̃ = tanh(0.5) = 0.462, so h = 0.18 + 0.8·0.462 = **0.550**. The candidate ignored the past.
- **With z = 1 (update off):** h = 1·0.9 + 0 = **0.9**. The state is copied unchanged.

Our demo shows the same three behaviours: 0.900 → 0.900, 0.955 and 0.765.

### 2.2 What each gate does
- **r ≈ 0 (reset):** the candidate ignores the previous state, so the unit can **drop** information that turned out irrelevant (e.g. at a phrase boundary).
- **z ≈ 1 (update):** the old state is **copied through**: an adaptive "leaky integrator", like an LSTM memory.
- **Units specialize:** short-term units have frequently active **reset** gates; long-term units have mostly active **update** gates.

### 2.3 Why it keeps gradients alive
- **The direct path:** differentiating Eq. 7 with respect to the old state gives **diag(z)**, plus terms that go through h̃.
- **With z ≈ 1 the gradient passes almost unchanged.** That's the same additive-path trick as the LSTM's CEC (Papers 021, 025).
- **Without gates** it fails: "We were not able to get meaningful result with an oft-used tanh unit without any gating."

### 2.4 GRU vs LSTM
- 2 gates instead of 3;
- no separate memory cell (h is the memory);
- no output gate;
- **about 25% fewer parameters** for the same width (Paper 025's comparison).

---

## 3. Inside a phrase-based SMT system (Section 3)

### 3.1 The log-linear model (Eq. 9)
- **The model:** statistical MT scores a candidate translation f of a sentence e as
  ```
  log p(f | e) = Σ_n w_n f_n(f, e) + log Z(e)
  ```
- **The features f_n:**
  - phrase-translation probabilities;
  - a language-model score;
  - a length penalty, and so on.
- **The weights w_n** are tuned to maximize BLEU on a development set.
- **A tiny example:** two candidates with features (LM score, phrase score) = (−3, −2) and (−2, −4), with weights (1, 1). They total −5 and −6, so the first wins. Adding a third feature (the RNN's log p) with weight 0.5 can flip the ranking if the RNN strongly prefers the second.

### 3.2 The paper's use
1. **Training data:** train the encoder–decoder on the **phrase table**, deliberately **ignoring how often** each pair occurs. The table already captures frequency, so the RNN can focus on **linguistic regularity**: what plausible translations look like.
2. **The new feature:** add its score log p(f | e) as **one more feature** in Eq. 9.

---

## 4. The training pieces, explained

| Piece | What it is | Why |
|---|---|---|
| **low-rank embeddings** | a word's vector = a 15,000 × 100 matrix followed by a 100 × 1000 matrix | 15,000·1000 = 15M parameters vs 15,000·100 + 100·1000 = **1.6M**: ~10× fewer |
| **maxout** (500 units, each pooling 2) | each unit outputs max(a, b) of two linear projections | a learned piecewise-linear activation; max(0.3, −1.2) = 0.3 |
| **orthogonal recurrent init** | U = the left singular vectors of a random Gaussian matrix | all singular values = 1, so at init the linear part neither grows nor shrinks the state over time (Papers 007, 008) |
| **AdaDelta** (ε = 10⁻⁶, ρ = 0.95) | per-weight steps from running averages of the squared gradient and the squared past steps: Δθ = −(√(E[Δθ²] + ε)/√(E[g²] + ε))·g | no global learning rate to tune; units-consistent steps |
| **init N(0, 0.01²)** | small random weights | breaks symmetry |
| batch | 64 phrase pairs per update, ~3 days | |

---

## 5. Experiments (Section 4)

**Setup:**
- **Data:** WMT'14 English → French. 348M words selected to train the RNN, 418M for the language model.
- **Baseline:** Moses (phrase-based).
- **Vocabularies:** 15,000 most frequent words per side (93% coverage); the rest → [UNK].
- **RNN:** 1000 GRU units in the encoder and the decoder.

### 5.1 BLEU: how translations are scored
```
BLEU-N = BP · exp( (1/N) Σ_{n=1..N} log p_n ),     p_n = clipped n-gram precision,
BP = 1 if the candidate is longer than the reference, else e^{1 − r/c}   (r, c = reference, candidate lengths)
```
- **Clipped:** a candidate n-gram counts at most as many times as it appears in the reference.

**A worked example:** candidate "the cat sat on the mat" vs reference "the cat is on the mat".

| n | matches | p_n |
|---|---|---|
| 1 | the ×2, cat, on, mat | 5/6 |
| 2 | "the cat", "on the", "the mat" | 3/5 |
| 3 | "on the mat" | 1/4 |
| 4 | none | 0/3 |

- **BLEU-2** = √(5/6 · 3/5) = **0.707**; the lengths are equal, so BP = 1.
- **BLEU-4 = 0,** because one zero precision zeroes the geometric mean. That's why BLEU-4 is 0 on our 3-word toy phrases (they have no 4-grams), while BLEU-3 is 100.

### 5.2 Table 1: BLEU
| System | dev | test |
|---|---|---|
| baseline (Moses) | 30.64 | 33.30 |
| + RNN Encoder–Decoder | 31.20 | 33.87 |
| + CSLM (neural LM) + RNN | **31.48** | **34.64** |
| + CSLM + RNN + word penalty | 31.50 | 34.54 |

**The CSLM and the RNN help in different ways:** their gains add up.

### 5.3 Qualitative analysis (Section 4.3, Tables 2–3)
- **Long source phrases:** the RNN prefers translations that are **closer to literal**, and often **shorter**. The frequency-based table favours odd phrases because of corpus statistics.
- **Sampled from the RNN alone,** it produces well-formed phrases, some of them absent from the table.

### 5.4 Representations (Section 4.4, Figures 4–5)
- **Word embeddings** cluster semantically similar words.
- **The phrase vectors c** capture semantics and syntax: time phrases ("for the first time", "in the past few months") cluster together, and so do syntactically similar phrases.

---

## 6. Why it matters

- **The encoder–decoder template** runs through Seq2Seq (027), attention (028), captioning (029) and the Transformer (an encoder + a decoder).
- **The GRU** is, with the LSTM, one of the two standard gated cells (compared in Paper 025).
- **The weakness:** the **fixed-length bottleneck c**. A long sentence must squeeze into the same 1000 numbers as a short one. Paper 028 (Bahdanau, same group) fixed this with **attention**.

---

## 7. What our code found

**Scale note:**
- At your request, nothing was trained on this laptop.
- `experiments.py` uses **Multi30k** (29k English–German pairs) instead of WMT'14 + Moses:
  - E1: rank the true translation among 10 candidates by log p(y | x), GRU vs the plain tanh unit;
  - E2: greedy-translation BLEU;
  - E3: nearest neighbours of phrase vectors c;
  - E4: a tiny log-linear rescorer (Eq. 9) with its weight tuned on dev.

**Checked (tests and demo, about a second):**
- **The GRU implements Eqs. 5–8 exactly:** with z ≈ 1 the state is copied (0.900 → 0.900); r ≈ 0 makes the candidate ignore the past (→ 0.765); r ≈ 1 uses both (→ 0.955).
- **The encoder ignores padding.** log p(y | x) sums only real target tokens, and each step's distribution sums to 1.
- **A toy translation** (reverse 3-symbol phrases):
  - 200/200 test phrases exactly right after 300 steps;
  - the true translation gets log p = −0.003 vs −14.3 for the unreversed phrase and −57.8 for "7 7 7";
  - so for short inputs the whole source **does** fit in one vector c.
- **BLEU** is implemented per Papineni et al. (clipped counts, brevity penalty), checked on hand cases like the one above.

**The paper is inconsistent about two sizes:**
- **The embedding size:** the main text says embeddings come from **rank-100** matrices, while the appendix says **500-d** embeddings. We default to 100.
- **The output factors:** the appendix gives the output factors as G_l ∈ R^{K×500}, G_r ∈ R^{500×1000}, which doesn't match the 500 maxout units. We use G = G_l·G_r with G_r: 500 → 100 and G_l: 100 → K, following "rank-100".

We follow the appendix's decoder exactly: there the reset gate multiplies (U′h + Cc), after the matrix product, while the encoder applies it before U.

---

## 8. Check yourself

1. Write p(y | x) as a product of next-word probabilities. What is teacher forcing?
2. Where does c enter the decoder? What is h′₀?
3. Run the GRU by hand: h = 0.5, [Wx] = 0, U = 1, z = 0.5, r = 1. (h̃ = tanh(0.5) = 0.462, so h = 0.25 + 0.231 = 0.481.)
4. What does ∂h_t/∂h_{t−1} = diag(z) + … say about gradients when z ≈ 1?
5. Compute BLEU-1 and BLEU-2 for candidate "a b c d" vs reference "a b d c". (p₁ = 1; p₂ = 1/3 ("a b"); BLEU-2 = √(1/3) = 0.577.)
6. How does a low-rank embedding save parameters? Compute it for V = 15,000, d = 1000, rank 100.
7. Why did the authors train on unique phrase pairs, ignoring frequency?
8. Why is the fixed-length c a problem for long sentences?
