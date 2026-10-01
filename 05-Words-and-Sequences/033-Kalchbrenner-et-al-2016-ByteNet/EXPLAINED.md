# Kalchbrenner et al. (2016), explained from scratch

**Paper:** *Neural Machine Translation in Linear Time* ("ByteNet")
**Authors:** Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aäron van den Oord, Alex Graves, Koray Kavukcuoglu (Google DeepMind)
**Published:** arXiv:1610.10099, 2016 (revised 2017)

Read Papers 013 (convolutions), 016 (residual blocks), 027 (Seq2Seq) and 028 (attention) first. This paper is **optional** in the reading order: it is the convolutional bridge between RNN translators and the Transformer. This guide:
- derives the receptive-field formula behind dilation;
- explains masked convolutions, residual blocks, layer norm and multiplicative units;
- explains "dynamic unfolding";
- explains *why* this runs in linear time.

---

## 0. The whole idea in one line

> **Translate characters with two stacks of 1-D convolutions instead of RNNs. Dilated convolutions see far with few layers. Masked (causal) convolutions let the decoder be a language model. The decoder sits directly on top of the encoder's output, position by position, so nothing is squeezed into one vector, and the cost grows only linearly with sentence length.**

---

## 1. What was wrong with RNN translators (Sections 1–2)

### 1.1 The three desiderata (Section 2.1)
1. **Linear running time** in the source and target lengths, and **parallel** computation along the sequence. Characters make sentences ~5× longer than words, so this matters.
2. **A resolution-preserving source representation:** its size should grow with the source length. A fixed vector (Papers 026–027) forces a "memorization step".
3. **Short paths** between any input token and any output token, so signals and gradients travel few steps (Paper 021: long paths make learning hard).

### 1.2 How earlier models score (Table 1)
| Model | Time | Resolution preserving | Path source → target | Path target → target |
|---|---|---|---|---|
| RNN Enc-Dec (027) | \|S\| + \|T\| | **no** | \|S\| + \|T\| | \|T\| |
| RNN Enc-Dec + attention (028) | **\|S\|·\|T\|** | yes | 1 | \|T\| |
| Grid LSTM | \|S\|·\|T\| | yes | \|S\| + \|T\| | \|S\| + \|T\| |
| Extended Neural GPU | \|S\|² + \|S\|·\|T\| | yes | \|S\| | \|T\| |
| **ByteNet** | **c\|S\| + c\|T\|** | **yes** | **c** | **c** |

- **Attention** gives short source paths but costs **quadratic** time.
- **The plain encoder–decoder** is linear but squeezes everything into one vector.
- **ByteNet** is linear, keeps the full resolution, and has short paths (c ≈ the number of layers, about log of the dependency distance).

---

## 2. The building blocks

### 2.1 One-dimensional convolution, masked and dilated
- **A 1-D convolution** with kernel size k and dilation r computes, at position i:
  ```
  out[i] = Σ_{j=0}^{k−1} W_j · in[i − j·r]          (masked / causal: only the present and the past)
  ```
- **Masked (Section 3.4):** the decoder must never see future target characters, or predicting them would be cheating. So its kernel only reaches **backwards**.
  - **Our implementation:** pad (k − 1)·r zeros on the left.
  - **The paper's equivalent:** zero half of a wider kernel of size 2k − 1.
- **The encoder** may look both ways: centred kernels.
- **Our test:** with k = 3 and r = 2, a masked output at position 10 depends on exactly positions **10, 8 and 6** (i, i − r, i − 2r). The centred version uses **8, 10 and 12**.

### 2.2 Dilation: exponential reach (Section 3.5)
- **The idea:** dilation r means the kernel's taps are r positions apart.
- **The schedule:** the rates **double every layer**, 1, 2, 4, 8, 16, then start again at 1.
- **The receptive field** (how many input positions one output can see) of a stack of layers is:
  ```
  RF = 1 + Σ_layers (k − 1) · r_layer
  ```
  **Why:** each layer extends the window by (k − 1)·r positions, and the extensions add up.

| layers (k = 3) | rates | receptive field | without dilation |
|---|---|---|---|
| 3 | 1, 2, 4 | 1 + 2·7 = **15** | 7 |
| 5 | 1, 2, 4, 8, 16 | 1 + 2·31 = **63** | 11 |
| 10 | (1–16) × 2 | 125 | 21 |
| 30 (the paper's LM) | (1–16) × 6 | **373** | 61 |

- **Exponential vs linear:** with dilation, each block of 5 layers multiplies its window's reach by about 32, while undilated layers add only 2 positions each.
- **The paper says the 30-block decoder sees 315 characters.** Our formula gives 373 (and 311 for 25 blocks). We couldn't reproduce 315; the gradient check below confirms our formula for our implementation.

### 2.3 Residual blocks with layer normalization (Section 3.6, Figure 3)
**The ReLU block** (used for translation). The input has 2d channels:
```
LN → ReLU → 1×1 conv (2d → d) → LN → ReLU → masked 1×k dilated conv (d → d) → LN → ReLU → 1×1 conv (d → 2d) → + input
```
- **The 1×1 convs** narrow the stream to d channels for the expensive k-wide conv, then widen it back: a bottleneck, as in ResNet-50 (Paper 016).
- **The "+ input"** is ResNet's shortcut, so gradients pass through 60 layers.
- **Layer norm, not batch norm:**
  - Batch norm computes statistics over the batch **and over time**, so a position's normalization would depend on **future** positions. In a causal decoder that leaks the answer.
  - **Layer norm** normalizes over the **channels of each position separately**, so nothing mixes across time (Ba et al. 2016).

**The multiplicative-unit (MU) block** (used for language modelling) replaces the middle with Multiplicative Units (from Video Pixel Networks):
```
g₁, g₂, g₃ = σ(convs of h),   u = tanh(conv of h)
MU(h) = g₁ ⊙ tanh(g₂ ⊙ h + g₃ ⊙ u)
```
It's like an LSTM cell (Paper 021) without the time recurrence: gates decide how much of h to keep, how much new content u to add, and how much to output.

### 2.4 The output head
After the blocks:
```
1×1 conv → ReLU → 1×1 conv → softmax over characters
```
This happens at every position in parallel.

---

## 3. Encoder–decoder stacking and dynamic unfolding (Sections 3.1–3.2)

### 3.1 Stacking (resolution preserving)
- **The decoder sits on top of the encoder,** position by position (Figure 1). At target position i, the decoder sees the encoder's output at **position i** plus the embedding of the previous target character.
- **No squeezing:** nothing is compressed into one vector (unlike Paper 027), and there's no attention pooling (unlike Paper 028).
- **Generalization of RNN Enc-Dec:**
  - an RNN Enc-Dec is a "recurrent ByteNet" with every source→target connection cut except the first;
  - so ByteNet generalizes it (Section 4.1).

### 3.2 Dynamic unfolding (Eq. 2, Figure 2)
- **The problem:** the source and target lengths differ, so position i of the target needs *some* encoder output. Choose a tight upper bound on the target length:
  ```
  |t̂| = a·|s| + b              (English → German: a = 1.2, b = 0)
  ```
- **The fix:** pad the source to length |t̂| so the encoder produces |t̂| outputs.
- **Decoding:**
  - the decoder unfolds step by step over them until it emits **EOS**;
  - beyond |t̂| it simply sees no representation (zeros).
- **Why a linear rule works (Figure 5):** source and target **character** lengths are extremely correlated, ρ = 0.968, and German is a bit longer than English.

### 3.3 Training is parallel; decoding is sequential
- **Training:** every target character is known, so **all positions are computed at once** (teacher forcing). There's no recurrence to wait for.
- **Generation:** predicting character i needs characters < i, so decoding is step by step.

---

## 4. Why "linear time"

- **The cost:** each convolution layer touches each position a constant number of times (k·d² operations per position). With L layers and n positions, the cost is **O(n·L·d²)**: linear in n.
- **How deep L must be:** to connect tokens D apart, L must be about log₂ D (dilation), hence the paper's "c ≈ log d".
- **Attention instead** scores every (source, target) pair: |S|·|T| scores, which is **quadratic**.
  - **Our demo:** at 4000 characters, attention would compute **16 million** scores.
  - Our decoder's forward pass went 2.0 → 3.2 → 4.9 → 8.3 ms for 500 → 1000 → 2000 → 4000 characters: roughly linear, with fixed overhead at small sizes.

---

## 5. Results

### 5.1 Character-level language modelling (Section 5, Table 3)
- **The data:** Hutter Prize Wikipedia (enwik8): 90M / 5M / 5M bytes, 205 symbols.
- **The model:**
  - **30 MU blocks** (6 sets of rates 1–16), kernel 3, d = 512;
  - Adam lr 3e-4, weight decay 1e-4, dropout 0.1 before the softmax, no learning-rate decay;
  - 500-character sequences, predicting the last 400.

| Model | Test bits/byte |
|---|---|
| stacked LSTM (Graves 2013) | 1.67 |
| Grid-LSTM | 1.47 |
| HM-LSTM | 1.40 |
| large layer-norm HyperLSTM | 1.34 |
| Recurrent Highway Networks | 1.32 |
| **ByteNet decoder** | **1.31** |

**The first non-recurrent model to beat all the LSTM variants** on this benchmark.

### 5.2 Character-to-character translation (Section 6, Tables 2 and 4)
- **The setup:** WMT English → German, characters in and characters out (323 German / 296 English symbols).
- **The model:**
  - 30 + 30 ReLU blocks, d = 800, kernel 3, Adam 3e-4;
  - sentences padded to the next multiple of 50, plus 20% for unfolding;
  - beam 12 over total log-likelihood, **no length normalization**.

| Model | Inputs → outputs | WMT '14 | WMT '15 |
|---|---|---|---|
| phrase-based MT | phrases | 20.7 | 24.0 |
| RNN Enc-Dec + attention (Chung et al.) | BPE → char | 21.33 | 23.45 |
| GNMT (Wu et al.) | char → char | 22.62 | — |
| GNMT | word-pieces | 24.61 | — |
| **ByteNet** | **char → char** | **23.75** | **26.26** |

- **The best character-level result,** second only to word-piece GNMT on 2014, and the best published on 2015.
- **0.521 / 0.532 bits per character.**

### 5.3 What it learned (Table 5, Figure 6)
- **Translations:** correct **reordering** and **transliteration** (e.g. "Jungle Book" kept as is).
- **Gradient saliency** (the magnitude of ∂output/∂input) shows a clear **alignment** between source and target words, without any attention mechanism. It also shows dependence on earlier target words.

---

## 6. Why it matters

- **It showed that recurrence is not needed** for state-of-the-art sequence modelling. Convolutions with dilation, residual connections and layer norm suffice.
- **Lineage:**
  - **WaveNet** (dilated causal convolutions for audio), same group, same year;
  - **ConvS2S** (Gehring et al. 2017);
  - the **Transformer** (2017), which keeps the parallel, resolution-preserving, short-path design but connects every pair of positions directly with attention. That gives path length 1, at quadratic cost.
- **Its pieces are now standard:** layer norm in residual stacks, and causal masking of a decoder so it can be trained in parallel.

---

## 7. What our code found

**Scale note:**
- At your request, nothing heavy was run on this laptop.
- `experiments.py` reproduces:
  - E1: Table 3 on enwik8;
  - E2: char-level translation on Multi30k, since WMT is too large;
  - E3: Figure 5's length correlation;
  - E4: Figure 6's saliency maps;
  - E5: time vs length against Paper 028's attention model.

**Checked (tests and demo, ~11 seconds):**
- **Masked vs centred convolutions** depend on exactly the expected positions (i, i − r, i − 2r vs i − r, i, i + r).
- **The receptive-field formula is exact:** for 5 dilated blocks, the gradient of one output reaches **exactly 63** positions, all in the past.
- **The paper's 30-block LM sees 373 characters by the formula,** not the stated 315.
- **The full ByteNet is causal on the target side:** changing target character 6 changes no prediction before position 6. The encoder sees the whole source: the first position reacts to the last character.
- **Dynamic unfolding:** the encoder output has length round(1.2·|s|), and decoding runs fine past it.
- **The MU formula, padding masking and saliency** all check out; saliency's target map is strictly lower-triangular.
- **Dilation, demonstrated** (copy the character 10 steps back, 3 blocks, 250 updates):

  | stack | sees | loss |
  |---|---|---|
  | dilated (1, 2, 4) | 15 positions | **0.001** (learned) |
  | undilated (1, 1, 1) | 7 positions | **1.617** (the guessing level is 1.609) |

  Same depth and parameters: only dilation makes the dependency reachable.
- **Translation with dynamic unfolding** (a substitution cipher): **100/100** strings exactly right, including stopping with EOS. Saliency recovers the alignment: output i depends most on source i.
- **Linear time:** 2.0 → 8.3 ms forward for 500 → 4000 characters.

**Honest notes:**
- **Long dependencies inside the window can still be hard to learn.** A lag of 20 was **not** learned in 800 updates by a 5-block stack (receptive field 63), while a lag of 10 was learned in 200. Reaching exactly 19 positions back needs one specific combination of taps (16 + 2 + 1); a lag of 10 (8 + 2) has an easier route. Being inside the receptive field is necessary, not sufficient.
- **How the decoder combines its inputs:** we **add** the encoder output to the target embedding at each position. The paper's figure shows stacking but doesn't spell out add vs concatenate.

---

## 8. Check yourself

1. Write a masked 1-D convolution with kernel 3 and dilation 4. Which input positions does output 20 use? (20, 16, 12.)
2. Derive RF = 1 + Σ(k − 1)r. Compute it for rates 1, 2, 4, 8 with k = 3. (31.)
3. How many undilated k = 3 layers would you need to see 63 positions? (31.)
4. Why does the decoder use layer norm instead of batch norm?
5. What is dynamic unfolding? Why is a linear rule a|s| + b good enough for characters?
6. Why is ByteNet's training parallel but its decoding sequential?
7. Compare the costs of ByteNet and attention for |S| = |T| = 1000 characters.
8. Why is "inside the receptive field" necessary but not sufficient for learning a dependency?
