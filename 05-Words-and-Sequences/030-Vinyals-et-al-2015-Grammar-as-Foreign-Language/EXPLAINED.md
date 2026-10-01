# Vinyals, Kaiser, Koo, Petrov, Sutskever & Hinton (2015), explained from scratch

**Paper:** *Grammar as a Foreign Language*
**Authors:** Oriol Vinyals, Łukasz Kaiser, Terry Koo, Slav Petrov, Ilya Sutskever, Geoffrey Hinton (Google)
**Published at:** NIPS 2015 (arXiv:1412.7449)

Read Papers 027 (Seq2Seq) and 028 (attention) first.

---

## The big idea in one line

> **Write a parse tree as a string of brackets, then "translate" the sentence into that string with a general-purpose seq2seq model with attention. With no parsing-specific machinery, it matches or beats decades of hand-engineered parsers.**

---

## 1. The problem

- **Constituency parsing** turns a sentence into a tree of phrases (S, NP, VP, PP, …).
- **The best parsers were specialized:**
  - chart parsers are cubic in sentence length;
  - fast shift-reduce parsers were less accurate.

  The idea of a tree is built into every part of them.
- **This paper's question:** can a domain-agnostic sequence model do it?

---

## 2. Trees as sequences (Section 2.2, Figure 2)

Linearize the tree **depth-first**:
```
John has a dog .   →   (S (NP NNP )NP (VP VBZ (NP DT NN )NP )VP . )S
```

**Three tricks (Section 2.3):**
- **POS-tag normalization:** replace every part-of-speech tag with **XX**: `(S (NP XX )NP (VP XX (NP XX XX )NP )VP XX )S`.
  - POS tags are not scored by the F1 metric, and this **improved F1 by about 1 point**.
  - That is surprising: classic parsers *need* POS tags.
- **Reverse the input sentence,** but not the tree, as in Paper 027. This is worth about 0.2 F1.
- **Nothing else:** no binarization, no special handling of unary rules. Unknown words become a single UNK.

The output vocabulary is 128 symbols. Each XX "consumes" the next word, so the tree can be rebuilt exactly from the output and the words.

### 2.1 How long is the output?
- **Each word gives one XX,** and each constituent gives an opening "(L" and a closing ")L".
- **For n words and k constituents,** the output has **n + 2k** symbols.
  - "John has a dog ." has 5 words and 4 constituents (S, NP, VP, NP), giving 5 + 8 = **13** symbols.
- **Typical length:** a WSJ sentence of 25 words has roughly 20 constituents, so about **65** output symbols. That's long for a 2014-era decoder, and part of why attention matters.

### 2.2 Rebuilding the tree (our `delinearize`)
1. **Walk the symbols left to right, with a stack of open nodes:**
   - "(L" opens a node under the current one and pushes it;
   - "XX" attaches the **next word** to the current node;
   - ")L" pops (closes the innermost node).
2. **Repair a malformed output (Section 3.2):** add brackets at the start or end until it balances:
   - an unclosed "(L" gets ")L" appended at the end;
   - a stray ")L" gets "(L" prepended at the start.

### 2.3 How parses are scored: labelled bracketing F1 (EVALB)
- **Every constituent is a triple (label, first word, last word).**
- **Precision** P = the fraction of predicted constituents that are in the gold tree.
- **Recall** R = the fraction of gold constituents that were predicted.
- **F1 = 2PR/(P + R)**, their harmonic mean.
- **POS tags are not scored,** which is why replacing them with XX costs nothing. Punctuation is ignored.

**Worked example** ("John has a dog .", punctuation ignored, so word positions 1–4):
```
gold:      S[1,4]  NP[1,1]  VP[2,4]  NP[3,4]
predicted: S[1,4]  NP[1,1]  VP[2,2]  NP[3,4]      (the object NP attached outside the VP)
matches: S, NP[1,1], NP[3,4] = 3      P = 3/4, R = 3/4, F1 = 75%
```
Our test reproduces exactly this: P = R = 75.

---

## 3. The model: LSTM+A (Section 2)

**The LSTM, as written in the paper:**
```
i = σ(W1 x + W2 h),  i' = tanh(W3 x + W4 h),  f = σ(W5 x + W6 h),  o = σ(W7 x + W8 h)
m_t = m_{t−1} ⊙ f + i ⊙ i'
h_t = m_t ⊙ o                    (no tanh on the memory, as in Show and Tell's Eq. 8)
```

**Seq2Seq:**
- A deep LSTM reads (A₁ … A_TA, B₁ … B_TB).
- **The encoder and the decoder use separate parameters.**
- P(B | A) = Π softmax(W_o h_{TA+t})_{B_t}, and every output ends with an end-of-sequence token.

**Attention (Section 2.1), adapted from Paper 028:**
```
u_ti = vᵀ tanh(W1' h_i + W2' d_t)     h_i: top-layer encoder states, d_t: the CURRENT decoder state
a_t  = softmax(u_t)
d'_t = Σ_i a_ti h_i
```
- **[d_t ; d'_t] is "the new hidden state from which we make predictions, and which is fed to the next time step".**
- **Different from Bahdanau:** the score uses d_t, not d_{t−1}, and there is no bidirectional encoder.

**Sizes and options:**
- 3 layers × 256 units, a 90K input vocabulary, and word2vec embeddings of size 512 as the starting point (fine-tuned during training).
- **For the small WSJ set:** dropout between layers 1–2 and 2–3 (**LSTM+A+D**).
- **Beam search** with beam 10.

---

## 4. Data (Section 3.1)

- **WSJ only:** the standard Penn Treebank training set, **40K sentences**. That is "very small by neural network standards".
- **BerkeleyParser corpus:** about 7M web-news sentences parsed by one parser, plus about 90K gold sentences.
- **High-confidence corpus:** about 11M sentences on which **two different parsers agree** ("tri-training"), resampled to match WSJ's length distribution, plus the gold sentences.

---

## 5. Results (Section 3.2)

### Table 1: F1 (EVALB)
| Parser | Training set | WSJ 22 | WSJ 23 |
|---|---|---|---|
| baseline LSTM+D | WSJ only | < 70 | < 70 |
| **LSTM+A+D** | WSJ only | 88.7 | 88.3 |
| LSTM+A+D ensemble | WSJ only | 90.7 | **90.5** |
| baseline LSTM | BerkeleyParser corpus | 91.0 | 90.5 |
| **LSTM+A** | high-confidence | 93.3 | **92.5** |
| LSTM+A ensemble | high-confidence | 93.5 | **92.8** |
| Petrov et al. 2006 (BerkeleyParser) | WSJ only | 91.1 | 90.4 |
| Huang & Harper 2010 ensemble | semi-supervised | 92.8 | 92.4 |

**On small data, attention is what makes it work:**
- the plain LSTM gets "no reasonable score";
- one LSTM+A+D gets 88.3;
- 5 of them match the BerkeleyParser (90.5 vs 90.4).

**On big data, a single LSTM+A (92.5) beats every previous single model and every previous ensemble.**

**Other findings:**
- **Malformed trees:** 1.5% (WSJ-trained) and 0.8% (big data). They are fixed by adding brackets at the start or the end. All the unbalanced cases were sentences without final punctuation.
- **Length (Figure 3):** from length ≤ 30 to ≤ 70, F1 drops 1.3 for the BerkeleyParser, 1.7 for the baseline LSTM and only **0.7 for LSTM+A**.
- **Beam size:** "almost irrelevant". Beam 2 costs 0.2 F1, beam 1 costs 0.5, and above 10 there's no gain.
- **Dropout** on WSJ is worth over 2 points (86.5 without it, 88.7 with it, on dev).
- **word2vec pretraining** adds a small 0.3–0.4 F1.
- **Other domains:** 95.7 on questions (QTB) and 84.6 on web text (WEB, better than the previous best of 83.5).
- **Speed:** over 120 sentences per second on CPU with an unoptimized decoder.

### Figure 4: what attention learned
- The focus is **sharp, on one word**, and **moves monotonically left to right**.
- It **steps right every time a word (XX) is consumed**: a learned stack-like procedure.
- It sometimes skips words. Because the input is reversed, the state at position i knows about all the words after i.

---

## 6. Why it matters

- **Structured prediction can be "just" sequence prediction.** This idea runs through later work:
  - parsing, code generation, and math as text;
  - T5's "everything is text-to-text";
  - today's LLMs producing JSON, trees and programs.
- **It showed that attention makes seq2seq data-efficient,** and that **synthetic labels** (data labelled by existing models, filtered by agreement) can train a model better than the labellers. This is an early form of self-training and distillation.

---

## 7. What our code found

**Scale note:**
- At your request, nothing heavy was run.
- The PTB and the 11M-sentence corpus aren't free, so `experiments.py` uses the **10% WSJ sample in NLTK** (3,914 sentences, about 10× less than "WSJ only") for:
  - E1: Table 1's small-data rows, with malformed rate and speed;
  - E2: F1 by length;
  - E3: beam size;
  - E4: XX vs real tags;
  - E5: input reversal;
  - E6: a Figure 4 attention matrix.

**Checked (tests and demo, about 10 seconds):**
- **Linearization reproduces Figure 2 character for character.** Rebuilding the tree from the output and the words is exact (F1 100 on 50 random trees).
- **Repair works:**
  - unbalanced outputs are fixed at the ends;
  - with too many or too few XX symbols, every word still ends up in the tree.
- **Our EVALB re-implementation** (labeled spans, punctuation deleted, ADVP = PRT) gives P = R = 75% on a hand-checked wrong attachment.
- **The toy grammar** (NP/VP/PP with PP-attachment recursion, 4–24 words), with one layer and the same short budget for both models:

  | | all | ≤ 10 words | > 10 words | malformed |
  |---|---|---|---|---|
  | **LSTM+A** | **89.8** | 97.7 | **82.7** | 1/150 |
  | baseline LSTM | 80.5 | 92.8 | 69.6 | 11/150 |

  Attention helps most on long sentences, and the baseline produces 10× more malformed trees.
- **Attention (Figure 4):** while emitting XX, the attention peak moves right or stays put **79%** of the time; it does sweep left to right. But it's within one word of the word being consumed only **25%** of the time. Our small, briefly trained model is **much blurrier than the paper's sharp pointer.** We say so rather than claiming we reproduced Figure 4.

**Our notes on the text:**
- The LSTM is printed as **h = m ⊙ o (no tanh)**, and we implement it that way. `cell_tanh=True` gives the standard form.
- **"[d_t; d′_t] … is fed to the next time step"**: we read this as *input feeding*, meaning the next decoder input is [embedding of the previous symbol ; d_{t−1} ; d′_{t−1}].
- **The paper gives no learning rate or batch size.** Our experiments use SGD lr 0.5, batch 64 and clipping at 5.
- **When the number of XX doesn't match the number of words**, the paper doesn't say what it does. We drop extra XX symbols and attach left-over words to the root.

---

## 8. Check yourself

1. Linearize (S (NP (DT the) (NN cat)) (VP (VBD slept)) (. .)) with POS normalization.
2. Why might replacing POS tags with XX *help* this model, even though it hurts classic parsers?
3. How is this paper's attention different from Bahdanau's (Paper 028)?
4. What made a WSJ-only model work at all? What closed the remaining gap with big data?
5. How is a malformed output repaired? Which sentences produced them?
6. Describe what Figure 4's attention does while the tree is generated.
