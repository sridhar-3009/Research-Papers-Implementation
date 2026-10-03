# GPT-2: Language Models are Unsupervised Multitask Learners, explained simply

**Paper:** Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei & Ilya Sutskever, *Language Models are Unsupervised Multitask Learners*, OpenAI, 2019.

**In one sentence:** train a bigger GPT (up to 1.5B parameters) on a large, diverse, quality-filtered web dataset, and it starts doing tasks **with no fine-tuning at all**: reading comprehension, translation, summarisation, question answering. It does them just because predicting internet text well requires it, and the task can be *described in the text itself*.

---

## 1. The big idea

### 1.1 From fine-tuning to zero-shot
- **GPT (paper 048):** pre-train, then **fine-tune** a copy for each task with labelled data.
- **GPT-2 asks:** what if we skip fine-tuning completely?

**The reasoning:**
- A task is a conditional distribution: p(output | input).
- A general system must also know *which* task: p(output | input, **task**).
- **Language can specify the task.** "translate to french: cheese =" or "Article … TL;DR:" are just text.
- So a language model trained on enough text that *naturally contains* such demonstrations might learn to perform them in order to predict the text better.

**Real examples the authors found in WebText (Table 1):**
> "I'm not the cleverest man in the world, but like they say in French: Je ne suis pas un imbecile [I'm not a fool]."

The internet is full of accidental translation, question–answer and summary examples.

### 1.2 Why this should work in principle
- For a supervised task written as a sequence, the supervised objective is the language-modelling objective evaluated on just the output tokens.
- So the **global minimum** of the language-modelling objective is also a minimum of the supervised one.
- The question is only whether we can optimise the LM well enough. That needs big models and lots of diverse data.

---

## 2. The data: WebText (Section 2.1)

- **Common Crawl** is huge but full of junk ("mostly unintelligible" documents).
- **WebText** uses humans as a quality filter: scrape every **outbound link from Reddit with at least 3 karma**. If people upvoted the link, the page was probably interesting or informative.

**Size and cleaning:**
- 45 million links;
- after extracting text, de-duplicating and cleaning: **8 million documents, 40 GB** of text;
- **Wikipedia removed**, because many test sets come from it and overlap would complicate evaluation.

---

## 3. The input representation: byte-level BPE (Section 2.2)

**The goal:** a model that can assign probability to **any** string, with no lowercasing, no tokenizer-specific preprocessing and no ⟨UNK⟩.

| option | problem |
|---|---|
| words | out-of-vocabulary words; preprocessing loses information |
| Unicode characters as BPE base | the base alphabet alone would be 130,000+ symbols |
| raw bytes | only 256 symbols, but byte-level LMs were not competitive |
| **byte-level BPE** | **base vocabulary of 256 bytes, plus merges learned on byte sequences** |

**One fix:**
- Greedy BPE on bytes learns wasteful tokens like "dog.", "dog!" and "dog?".
- So BPE is **not allowed to merge across character categories** (letters, digits, punctuation).
- One exception: a **space** may attach to the following word ("␣dog"). This improves compression a lot.
- **Final vocabulary: 50,257 tokens.**

**Pre-tokenization example:**

```
"I said: dog, dog!" → ["I", " said", ":", " dog", ",", " dog", "!"]
```

BPE merges happen only *inside* each piece.

**The payoff:**
- Tokenization is **invertible**: decode(encode(s)) = s, for emoji, accents and any script.
- Every dataset can be evaluated as-is, whatever its own tokenization.

---

## 4. The model (Section 2.3, Table 2)

It is the same decoder-only Transformer as GPT, with these changes:

| change | why |
|---|---|
| **pre-LN:** LayerNorm at the *input* of each sub-block (x + f(LN(x))) | like pre-activation ResNets; the residual path stays an identity, which helps deep models train |
| **an extra LayerNorm after the last block** | normalises the residual stream before the output layer |
| **residual-layer weights scaled by 1/√N at init** (N = number of residual layers) | the residual stream is a *sum* of N branch outputs; this keeps its variance from growing with depth |
| vocabulary 50,257; context 1024 (was 512); batch 512 | more data per step, longer context |

**Why 1/√N?**
- If each residual branch adds an independent contribution with variance v, then after N branches the stream's variance grows by N·v.
- Scaling each branch's output weights by 1/√N scales its variance by 1/N. The total added variance stays about v, independent of depth.
- **Our demo** (48 layers, so 96 residual branches): the stream's std after all blocks is **0.212 unscaled vs 0.034 scaled**; the input std was 0.029.

**Table 2:**

| label | layers | d_model | exact count of this architecture |
|---|---|---|---|
| 117M | 12 | 768 | **124.4M** |
| 345M | 24 | 1024 | **354.8M** |
| 762M | 36 | 1280 | **774.0M** |
| 1542M (GPT-2) | 48 | 1600 | **1557.6M** |

- The exact counts use tied embeddings: about 12·d² + 13·d per block, plus 50,257·d + 1024·d for embeddings, plus the final LN.
- **The paper's labels are slight undercounts.** The "117M" model really has 124M parameters. OpenAI later used the corrected sizes.
- The smallest is GPT-sized; the second matches BERT-Large.

---

## 5. Zero-shot results (Section 3)

### 5.1 Language modelling across domains (Table 3)
Every number is **zero-shot**: no training on these datasets.

| dataset | previous SOTA | 117M | 345M | 762M | 1542M |
|---|---|---|---|---|---|
| LAMBADA (PPL) | 99.8 | 35.13 | 15.60 | 10.87 | **8.63** |
| LAMBADA (ACC) | 59.23 | 45.99 | 55.48 | 60.12 | **63.24** |
| CBT-CN (ACC) | 85.7 | 87.65 | 92.35 | 93.45 | **93.30** |
| CBT-NE (ACC) | 82.3 | 83.4 | 87.1 | 88.0 | **89.05** |
| WikiText2 (PPL) | 39.14 | 29.41 | 22.76 | 19.93 | **18.34** |
| PTB (PPL) | 46.54 | 65.85 | 47.33 | 40.31 | **35.76** |
| enwik8 (BPB) | 0.99 | 1.16 | 1.01 | 0.97 | **0.93** |
| text8 (BPC) | 1.08 | 1.17 | 1.06 | 1.02 | **0.98** |
| WikiText103 (PPL) | 18.3 | 37.50 | 26.37 | 22.05 | **17.48** |
| 1BW (PPL) | **21.8** | 75.20 | 55.72 | 44.58 | 42.16 |

- **7 of 8 state-of-the-art results**, zero-shot.
- **The exception:** 1 Billion Word. It shuffles sentences, destroying the long-range structure GPT-2 relies on.

**Fair comparison across tokenizers:**
- Results are reported per **canonical unit** (word, character or byte): the total log-probability divided by the number of units.
- Example: 100 words with a total NLL of 100·ln 8 nats give a per-word perplexity of exactly 8, i.e. 3 bits per word.

### 5.2 Long-range dependencies: LAMBADA (Section 3.3)
- **The task:** predict the last word of a passage. Humans need ≥ 50 tokens of context to do it.
- **Results:**
  - perplexity **99.8 → 8.6**;
  - accuracy **19% → 52.66%**.
- **Most errors were valid continuations, but not valid *final words*.** Adding a **stop-word filter** (don't predict "the", "it", …) raised accuracy to **63.24%**.

### 5.3 Other tasks, all zero-shot, all steered by how the prompt is written

| task | prompt trick | result |
|---|---|---|
| **Children's Book Test** | score each of 10 candidate words *in the full sentence* | 93.3% common nouns, 89.1% named entities (SOTA) |
| **Winograd** | the more probable completion resolves the pronoun | 70.70% (+7 over SOTA) |
| **CoQA reading comprehension** | document + dialogue + "A:" | **55 F1**, matching 3 of 4 baselines that used 127k+ training pairs |
| **summarisation** | article + "**TL;DR:**", top-k sampling with k = 2, first 3 sentences | ROUGE only slightly above random-3-sentences; **drops 6.4 points without "TL;DR:"** (the hint matters) |
| **translation** | "english sentence = french sentence" examples, then "sentence =" | 5 BLEU EN→FR, **11.5 BLEU FR→EN** (WebText had only ~10 MB of French) |
| **Natural Questions** | example Q/A pairs, then "Q: … A:" | **4.1%** exact match; but **63.1%** on its most-confident 1% |

**Summarisation scores (ROUGE F1, Table 4):**

| | R-1 | R-2 | R-L | R-AVG |
|---|---|---|---|---|
| Bottom-Up Sum (SOTA) | 41.22 | 18.68 | 38.34 | 32.75 |
| Lede-3 | 40.38 | 17.66 | 36.62 | 31.55 |
| Seq2Seq + Attn | 31.33 | 11.81 | 28.83 | 23.99 |
| GPT-2 TL;DR: | 29.34 | 8.27 | 26.58 | 21.40 |
| Random-3 | 28.78 | 8.63 | 25.52 | 20.98 |
| GPT-2 no hint | 21.58 | 4.03 | 19.47 | 15.03 |

**The common thread:** the **bigger** the model, the better the zero-shot score. Performance grows **log-linearly** with capacity on every task (Figure 1), and none had flattened at 1.5B parameters.

---

## 6. Memorization or generalization? (Section 4)

**The worry:** test answers might simply be in the training data.

**The tool:**
- A **Bloom filter** holds every 8-gram of WebText, normalised to lower-case alphanumeric words with single spaces.
- Its false-positive rate was bounded below 10⁻⁸.
- Then measure what fraction of each test set's 8-grams appear in it.

**How a Bloom filter works:**
- An m-bit array and k hash functions. Adding an item sets k bits; checking an item tests those k bits.
- **No false negatives.** The false-positive rate is about (1 − e^{−kn/m})^k.
- **Example:** n = 15,900 items in m = 2²⁰ bits with k = 5 gives (1 − e^{−0.0758})⁵ = (0.0730)⁵ ≈ **2.1·10⁻⁶**. This is our demo's filter.

**The findings:**
- Common test sets overlap WebText by **1–6%** (average 3.2%).
- That is often *less* than their overlap with **their own training splits** (average 5.9%; 1BW overlaps its own training set by 13.2%).
- Removing overlapping LAMBADA examples changes the results only slightly (8.6 → 8.7 PPL; 63.2 → 62.9% accuracy).
- **Train and held-out WebText perplexities are similar and fall together with size:** GPT-2 still **underfits** WebText.

---

## 7. Why it works (the intuition)

1. **The web is a giant pile of task demonstrations.** Translation, Q&A, summaries and lists all appear naturally. A model that predicts web text well must implicitly learn them.
2. **A prompt selects the task.** "TL;DR:" or "A:" makes the desired continuation the most probable one.
3. **Capacity is the limit.** The bigger model learns more of these implicit tasks. The log-linear trends showed no saturation, which set up GPT-3 (paper 050) and scaling laws (paper 051).
4. **Byte-level BPE removes preprocessing.** One model can score any text, so evaluation is fair across very different benchmarks.

---

## 8. What our code found

All numbers come from `demo.py` (about 4 s) and `test_gpt2.py` (9 tests, about 0.8 s).

1. **Byte-level BPE:**
   - "naïve café 🐶 漢字" and arbitrary whitespace round-trip exactly;
   - no learned token mixes letters with . ! ?;
   - "dog. dog! dog?" becomes [d, og, ., ␣dog, !, ␣dog, ?].
2. **Sizes:** our exact counts are **124.4 / 354.8 / 774.0 / 1557.6M** for the four Table 2 shapes, slightly above the paper's labels.
3. **Pre-LN + 1/√N init:**
   - the residual layers' init std is 0.02/√(2L) and the other layers stay at 0.02;
   - the model is causal;
   - after 48 blocks the stream's std is **0.034** with scaling vs **0.212** without (input 0.029).
4. **Zero-shot task transfer** in a toy web text:
   - Q/A demonstrations appeared for countries 1–20 only;
   - capitals of countries 21–40 appeared **only as prose**;
   - a 2-layer word-level LM answered "q : what is the capital of X ? a :" correctly **100%** of the time for **both** groups.

   So the Q/A format transferred to facts the model had only read as prose, with no fine-tuning.

   **Honest note:** the toy is easy enough that a tiny model succeeds, so it can't show the paper's "bigger is better". In one development run, a 4-layer, 128-wide model scored only 10% on the prose-only countries. We did not investigate why.
5. **The 8-gram Bloom filter:**
   - 0.0% overlap for fresh text;
   - 21.5% for a test text with two copied training documents;
   - no false negatives (tested).

**Not run (too heavy):**
- E1: tokenizer comparison;
- E2: 4 shape-matched model sizes on OpenWebText or Gutenberg, with zero-shot bits/byte on other datasets;
- E3: LAMBADA with and without the stop-word filter;
- E4: TL;DR summarisation ROUGE;
- E5: Bloom-filter overlap;
- E6: pre-LN vs post-LN at 48 layers.

---

## 9. Check yourself

1. Why can a pure language model translate at all, with no translation training?
   <details><summary>Answer</summary>Web text contains natural translation demonstrations, so to predict that text the model must learn to translate. A prompt like "english = french" examples then invokes that behaviour.</details>
2. What does byte-level BPE have that word-level vocabularies lack?
   <details><summary>Answer</summary>Any string can be encoded (the 256-byte base covers everything), and encoding is invertible. There is no ⟨UNK⟩ and no lossy preprocessing.</details>
3. A 48-layer model has how many residual layers, and what is the scaled init std of their output weights?
   <details><summary>Answer</summary>N = 96 (attention + MLP per block), so 0.02/√96 ≈ 0.00204.</details>
4. A test set with 1,000 words has a total NLL of 2,000 nats. What is the per-word perplexity?
   <details><summary>Answer</summary>exp(2000/1000) = e² ≈ 7.39.</details>
5. Why does adding a stop-word filter help on LAMBADA?
   <details><summary>Answer</summary>Many of the model's wrong guesses were plausible continuations like "the" that can't be the final word of the sentence. Filtering them enforces that constraint.</details>
6. What did GPT-2 score on Natural Questions overall, and on its most confident 1%?
   <details><summary>Answer</summary>4.1% exact match overall; 63.1% on the most confident 1%.</details>
