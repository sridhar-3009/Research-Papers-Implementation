# Bahdanau, Cho & Bengio (2015), explained from scratch

**Paper:** *Neural Machine Translation by Jointly Learning to Align and Translate*
**Authors:** Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
**Published at:** ICLR 2015 (arXiv:1409.0473)

Read Papers 026 (encoder–decoder, GRU) and 027 (Seq2Seq) first. This paper introduces **attention**, the key idea behind Transformers. This guide:
- computes one attention step with real numbers;
- explains every part of the architecture;
- explains why attention removes the long-sentence problem.

---

## 0. The whole idea in one line

> **Don't squeeze the whole source sentence into one vector. Keep one vector per source word (an "annotation"). For every word the decoder writes, let it compute a soft weighting over those annotations (the "attention" or "soft alignment") and use their weighted average. Long sentences stop being a problem, and the model learns word alignments by itself.**

---

## 1. The bottleneck (Sections 1–2)

- **The bottleneck:** in the encoder–decoder (Papers 026–027), the **whole** sentence must fit into **one fixed-length vector**: 1000 numbers for 5 words or for 50.
- **The hypothesis:** this is a bottleneck. Cho et al. (2014b) had found that performance **drops quickly as sentences get longer**.
- **The fix:**
  - the encoder keeps a **sequence** of vectors, one per source word;
  - the decoder **searches** for the relevant ones at each step;
  - alignment and translation are learned **jointly**.

---

## 2. The decoder with attention (Section 3.1)

**Each target word i gets its own context vector c_i:**
```
p(y_i | y_1 … y_{i−1}, x) = g(y_{i−1}, s_i, c_i)              (4)
s_i = f(s_{i−1}, y_{i−1}, c_i)                                 decoder state update

e_ij = a(s_{i−1}, h_j)          a score: how well does source position j match what the decoder needs now?
α_ij = exp(e_ij) / Σ_k exp(e_ik)                               (6)  softmax over source positions
c_i  = Σ_j α_ij h_j                                             (5)  weighted average of the annotations
```

**The alignment model** a is a small feed-forward network:
```
a(s, h) = v_aᵀ · tanh(W_a s + U_a h)                ("additive attention")
```
- **It's differentiable,** so the alignment is trained **jointly** with everything else by ordinary backprop. No alignment labels are needed.
- **Precomputing:** U_a h_j doesn't depend on i, so it is computed **once** per sentence.

### 2.1 One attention step with numbers
**Setup:** 3 source annotations (2-d, for simplicity), and scores from the alignment net:
```
h₁ = (1, 0)   h₂ = (0, 1)   h₃ = (1, 1)          e = (2.0, 0.5, 1.0)
```
**Step 1, softmax:**
```
e^e = (7.39, 1.65, 2.72), sum = 11.76
α = (0.629, 0.140, 0.231)                      (positive, sums to 1)
```
**Step 2, the context vector:**
```
c = 0.629·(1,0) + 0.140·(0,1) + 0.231·(1,1) = (0.860, 0.371)
```
- **Read it:** the decoder "looks mostly at word 1" right now.
- **Training sharpens this:** in our toy reversal task, each output word ends up putting nearly all its weight on its mirror source word.

### 2.2 The interpretation
- **α_ij** = the probability that target word i is **aligned to** (translated from) source word j.
- **c_i** = the **expected annotation** under that distribution.
- **This "relieves the encoder from the burden of having to encode all information in the source sentence into a fixed-length vector."**
- **The cost:** every output step scores every input, so O(T_x · T_y) per sentence.

---

## 3. The encoder: a bidirectional RNN (Section 3.2)

- **Two RNNs:** a **forward** RNN reads x₁ → x_T, giving →h_j; a **backward** RNN reads x_T → x₁, giving ←h_j.
- **The annotation** of word j is **h_j = [→h_j ; ←h_j]**. It summarizes the words **before and after** j, with a focus on the words around j (an RNN's state is dominated by recent inputs).
- **Why both directions:** a word's translation can depend on what comes after it (French adjectives follow nouns; English adjectives precede them).

---

## 4. The exact architecture (Appendix A)

- **GRUs** (Paper 026), written as **s = (1 − z) ⊙ s_prev + z ⊙ s̃**. That's the mirror image of Cho et al.'s convention, so here z is the share of the *new* state.
- **The decoder GRU reads c_i in every gate:**
  ```
  s̃_i = tanh(W E y_{i−1} + U [r_i ⊙ s_{i−1}] + C c_i),   z_i = σ(… + C_z c_i),   r_i = σ(… + C_r c_i)
  ```
- **The first decoder state:** s₀ = tanh(W_s ←h₁). The backward RNN's state at word 1 has read the whole sentence.
- **The output layer is a maxout network:**
  ```
  t̃_i = U_o s_{i−1} + V_o E y_{i−1} + C_o c_i,    t_i = max over pairs of t̃_i,    p(y_i) ∝ exp(y_iᵀ W_o t_i)
  ```
  **Maxout:** each output unit takes the larger of two linear projections, a learned piecewise-linear activation (Paper 026, section 4).
- **Sizes:**
  - n = 1000 hidden units;
  - m = 620-d embeddings;
  - l = 500 maxout units;
  - n′ = 1000 in the alignment net.
- **Init (Appendix B.1):**
  - recurrent matrices random **orthogonal**;
  - W_a, U_a ~ N(0, 0.001²);
  - **v_a = 0** and biases 0;
  - everything else ~ N(0, 0.01²).

  **Why v_a = 0 matters:** every score e_ij = v_aᵀ(…) = 0, so α_ij = 1/T_x for all j. **Attention starts perfectly uniform** and learns to focus. Our demo shows 0.2 each for 5 words.
- **Training (Appendix B.2):**
  - AdaDelta (ε = 10⁻⁶, ρ = 0.95), gradient norm clipped at **1**, minibatches of 80;
  - every 20 updates: take 1600 pairs, **sort by length**, and split into 20 minibatches (padding waste stays small);
  - ~5 days per model.

---

## 5. Experiments (Sections 4–5)

**Setup:**
- **Data:** WMT'14 English → French, 348M words selected. 30,000-word vocabularies; others → [UNK].
- **Models:** RNNencdec (Cho et al.) vs **RNNsearch**, each trained on sentences up to **30** or up to **50** words. Decoded with beam search.

### 5.1 Table 1: test BLEU
| Model | All sentences | Sentences with no UNK |
|---|---|---|
| RNNencdec-30 | 13.93 | 24.19 |
| RNNsearch-30 | 21.50 | 31.44 |
| RNNencdec-50 | 17.82 | 26.71 |
| RNNsearch-50 | 26.75 | 34.16 |
| RNNsearch-50* (trained longer) | **28.45** | **36.15** |
| Moses (phrase-based SMT) | 33.30 | 35.63 |

- **RNNsearch beats RNNencdec everywhere.** RNNsearch-**30** even beats RNNencdec-**50**.
- **On sentences with known words, RNNsearch-50\* (36.15) beats Moses (35.63).** Moses also uses an extra 418M-word monolingual corpus.

### 5.2 Figure 2: long sentences
- **RNNencdec's BLEU collapses** as sentences grow: one vector can't hold them.
- **RNNsearch-50 shows no deterioration** even at 50+ words: each output word looks up what it needs.

### 5.3 Figure 3: the learned alignments
- **Mostly monotonic,** like real word alignments, but they **handle reordering**. "European Economic Area" → "zone économique européenne" shows the reversed order on the diagonal.
- **Soft:** "the man" → "l'homme" attends to both words to choose "l'". A hard one-to-one alignment couldn't express that.

---

## 6. Why it matters

- **Attention is the core of the Transformer** ("Attention Is All You Need", Stage 06), where it replaces the RNNs entirely.
  - Bahdanau's score v_aᵀ tanh(W_a s + U_a h) is "**additive attention**".
  - Transformers use "**scaled dot-product attention**" sᵀh/√d, which is cheaper and parallel.
- **Soft, differentiable "look-ups" over a memory** reappear in:
  - image captioning (Show, Attend and Tell);
  - Pointer Networks (Paper 031), where the attention weights *are* the output;
  - memory networks and Neural Turing Machines.

---

## 7. What our code found

**Scale note:**
- At your request, nothing was trained on this laptop beyond seconds-long toy demos.
- `experiments.py` reproduces, on Multi30k English → German:
  - E1: Table 1, the 4 models with short/long cutoffs, with and without UNK;
  - E2: Figure 2, BLEU by length;
  - E3: Figure 3, alignment heatmaps.

**Checked (tests and demo, about 10 seconds):**
- **Truly bidirectional annotations:** changing the **last** source word leaves the forward half of h₁ unchanged but changes its backward half.
- **Attention is a proper distribution over real words:**
  - the weights sum to 1;
  - **zero** weight lands on padding;
  - c_i is exactly the weighted average (Eq. 5).
- **Precomputing U_a h_j** gives exactly the same attention.
- **v_a = 0 gives perfectly uniform attention at the start** (0.2 × 5).
- **The paper's central result, in miniature** (toy "reverse the sequence", same budget):

  | Length | RNNencdec | **RNNsearch** |
  |---|---|---|
  | 6 | 97.7% | 100% |
  | 16 | **59.1%** | **100%** |

  The one-vector model breaks as length grows; attention doesn't.
- **The learned alignment is the anti-diagonal** (each output word attends to its mirror position), **learned with no alignment supervision**. Beam search then outputs the exact reversed sequence.

**An observation about the init:**
- With the paper's tiny init, a short toy run barely learns: loss 7.3 after 400 Adam steps, attention still almost uniform.
- With PyTorch's default init, the same run reaches 0.002.
- The paper's init was meant for days of AdaDelta, so our learning test uses the default init.

---

## 8. Check yourself

1. Why is a fixed-length c a bottleneck? What did RNNencdec's BLEU do on long sentences?
2. Write Eqs. 5–6. Compute α and c for e = (0, 0, ln 2) with h = (1,0), (0,1), (1,1). (α = (¼, ¼, ½), c = (0.75, 0.75).)
3. Write the alignment model a(s, h). Why can U_a h_j be precomputed?
4. Why does v_a = 0 make attention uniform at the start?
5. Why use a bidirectional encoder? What is s₀?
6. What does "soft" alignment mean? Why does it make training easy (no alignment labels)?
7. What is the cost of attention per sentence, in terms of T_x and T_y?
8. How is additive attention different from the Transformer's dot-product attention?
