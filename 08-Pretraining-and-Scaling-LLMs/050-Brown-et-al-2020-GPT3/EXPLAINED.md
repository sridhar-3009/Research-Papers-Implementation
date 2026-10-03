# GPT-3: Language Models are Few-Shot Learners, explained simply

**Paper:** Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, and many others (OpenAI), *Language Models are Few-Shot Learners*, NeurIPS 2020.

**In one sentence:** scale GPT-2 up 100× to **175 billion parameters**, and it can learn a new task from **a few examples written in its prompt**, with no training and no weight changes. This is **in-context learning**. Often it is competitive with models fine-tuned on thousands of labelled examples.

---

## 1. The big idea

### 1.1 Fine-tuning has three problems
1. **Every task needs a big labelled dataset**, and many useful tasks have none.
2. **A huge model fine-tuned on a narrow dataset can exploit spurious patterns**, so it generalises badly out of distribution.
3. **Humans don't need it:** "Here are two examples, now do this one" is enough for a person.

### 1.2 In-context learning (Figure 2.1)
Give the model the task **in its context window** and let it continue the text:

| setting | what goes in the prompt | example |
|---|---|---|
| **zero-shot** | a task description only | "Translate English to French:\ncheese =>" |
| **one-shot** | the description plus 1 demonstration | "…\nsea otter => loutre de mer\ncheese =>" |
| **few-shot** | the description plus **K** demonstrations (K ≈ 10–100, as many as fit in 2048 tokens) | "…\nsea otter => loutre de mer\npeppermint => menthe poivrée\n…\ncheese =>" |
| fine-tuning | (not used) gradient updates on a labelled dataset | |

**No weights change in zero-, one- or few-shot.** The "learning" happens inside one forward pass.

**Meta-learning view:**
- **The outer loop** is pre-training on huge, diverse text. It teaches the model *how to pick up patterns from context*.
- **The inner loop** is reading the examples in the prompt and adapting on the fly.

---

## 2. The models (Section 2.1, Table 2.1)

GPT-2's architecture (pre-LN, scaled init, byte-level BPE), plus **alternating dense and locally banded sparse attention** (as in paper 036).

| model | params | layers | d_model | heads | d_head | batch (tokens) | learning rate |
|---|---|---|---|---|---|---|---|
| Small | 125M | 12 | 768 | 12 | 64 | 0.5M | 6.0·10⁻⁴ |
| Medium | 350M | 24 | 1024 | 16 | 64 | 0.5M | 3.0·10⁻⁴ |
| Large | 760M | 24 | 1536 | 16 | 96 | 0.5M | 2.5·10⁻⁴ |
| XL | 1.3B | 24 | 2048 | 24 | 128 | 1M | 2.0·10⁻⁴ |
| 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 1M | 1.6·10⁻⁴ |
| 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2M | 1.2·10⁻⁴ |
| 13B | 13.0B | 40 | 5140 | 40 | 128 | 2M | 1.0·10⁻⁴ |
| **175B ("GPT-3")** | **175.0B** | **96** | **12288** | **96** | **128** | **3.2M** | **0.6·10⁻⁴** |

- Context **2048** tokens; d_ff = 4·d_model.
- **All trained on 300 billion tokens.**
- **The pattern:** bigger models use **bigger batches** and **smaller learning rates**. Batch sizes were chosen with the *gradient noise scale*.

### 2.1 Counting parameters
Each block has attention (4d² for Q, K, V and output) and an MLP (2 × d × 4d = 8d²): **12d² per layer**.

```
175B: 12 × 96 × 12288² = 173.9B,   + embeddings (50257 + 2048) × 12288 = 0.64B   →  174.6B ≈ 175B ✓
```

**Our check:** every size is within 2% of its label.

**Two quirks in the paper's own table:**
- XL lists 24 heads × 128 = 3072, but d_model = 2048;
- 13B lists 40 × 128 = 5120, but d_model = 5140.

### 2.2 Compute (Appendix D)
**Training FLOPs ≈ 6 × N × D**: each parameter does about 2 FLOPs per token forward and 4 backward.

```
6 × 175·10⁹ × 300·10⁹ = 3.15·10²³ FLOPs
1 petaflop/s-day = 10¹⁵ × 86,400 = 8.64·10¹⁹ FLOPs  →  3.15·10²³ / 8.64·10¹⁹ ≈ 3,640 petaflop/s-days
```

Figure 2.2's message: following scaling laws (paper 051), the large models were trained on **relatively few tokens per parameter**. GPT-3 2.7B used about the same compute as RoBERTa-Large (355M) because it saw far fewer tokens.

---

## 3. The data (Section 2.2, Table 2.2, Appendix A)

| dataset | tokens | weight in the training mix | epochs over 300B tokens |
|---|---|---|---|
| Common Crawl (filtered) | 410B | 60% | 0.44 |
| WebText2 | 19B | 22% | 2.9 |
| Books1 | 12B | 8% | 1.9 |
| Books2 | 55B | 8% | 0.43 |
| Wikipedia | 3B | 3% | 3.4 |

**Sampling is by quality, not size.**
- Common Crawl is 85% of the tokens but only 60% of what's sampled, and is seen less than once.
- Wikipedia is seen 3.4 times.
- This trades a little overfitting for higher-quality data.
- **A small inconsistency:** the rounded weights sum to 101%, and weight × 300B / size gives 3.47 epochs for WebText2 and 3.0 for Wikipedia, slightly off the paper's own epoch column. Our demo prints both.

### 3.1 Cleaning Common Crawl (Appendix A)
**Step 1: quality filter.**
- A logistic-regression classifier learns to tell high-quality reference text (WebText, Wikipedia, books) from raw Common Crawl.
- Each document is **kept if** `np.random.pareto(α = 9) > 1 − score`.
- A Pareto(α) (Lomax) variable exceeds t with probability (1 + t)^(−α). So:

  | classifier score | P(keep) |
  |---|---|
  | 1.0 | 100% |
  | 0.99 | (1.01)⁻⁹ = 91% |
  | 0.9 | (1.1)⁻⁹ = 42% |
  | 0.5 | (1.5)⁻⁹ = 2.6% |
  | 0.1 | (1.9)⁻⁹ = 0.3% |

- **Why random?** It mostly keeps high-scoring text, but occasionally lets in odd-looking documents, which keeps diversity.

**Step 2: fuzzy de-duplication** with **MinHash LSH** (10 hashes).
- **Jaccard similarity** of two documents' shingle sets (sets of 5-word sequences) = |A ∩ B| / |A ∪ B|.
- **A MinHash slot** = the minimum hash value over a document's shingles, for one hash function.
- **Key fact:** P(two documents' MinHash slots agree) = their Jaccard similarity. So the fraction of agreeing slots estimates similarity, without comparing the full texts.
- **Our demo:** true Jaccard 0.85, MinHash estimate 0.80 with 10 hashes.

**Step 3:** add the curated datasets (WebText2, two book corpora, Wikipedia).

**Totals:** 45 TB of compressed Common Crawl became **570 GB** after filtering, about 400B tokens.

---

## 4. Evaluation details (Section 2.4)

**Multiple choice** (Section 2.4): give K demonstrations, then score each possible completion by language-model likelihood.
- **Per-token normalisation:** average log-probability per token, so long answers aren't penalised for having more tokens.
- **Unconditional normalisation** (ARC, OpenBookQA, RACE): score log P(completion | context) − log P(completion | "Answer: "). This removes answers that are just generically likely.

**Example:**

| answer | log P(answer \| context) | tokens | log P(answer \| "Answer:") |
|---|---|---|---|
| "Paris" | −2 | 1 | −6 |
| "the city of light" | −5 | 4 | −12 |

- **By total:** Paris wins (−2 > −5).
- **Per token:** "the city of light" wins (−1.25 > −2).
- **Unconditional:** "the city of light" wins (−5 + 12 = 7 > −2 + 6 = 4).

**Other details:**
- **Free-form tasks:** beam search (width 4, length penalty 0.6), as in GPT-2.
- **Binary tasks:** options get meaningful names ("True" / "False") and are scored like multiple choice.

---

## 5. Results (Section 3), with the headline numbers

### 5.1 Scaling
- Validation loss follows a **smooth power law in compute** over many orders of magnitude, with only slight deviations (Figure 3.1).
- Downstream accuracy improves smoothly with size in all three settings.
- **The gap between zero-, one- and few-shot grows with size:** larger models are better *in-context learners*.

### 5.2 Selected results (175B)

| task | zero-shot | one-shot | few-shot | reference |
|---|---|---|---|---|
| LAMBADA (acc) | 76.2 | 72.5 | **86.4** | SOTA 68.0 |
| TriviaQA (acc) | 64.3 | 68.0 | **71.2** | fine-tuned closed-book SOTA 68.0 |
| StoryCloze | 83.2 | 84.7 | 87.7 | SOTA 91.8 |
| HellaSwag | 78.9 | 78.1 | 79.3 | SOTA 85.6 |
| SuperGLUE (avg, 32 examples) | | | **71.8** | fine-tuned BERT-Large 69.0; SOTA 89.0 |

**Translation:** few-shot GPT-3 beats prior *unsupervised* machine translation by about 5 BLEU when translating **into** English.

### 5.3 New tasks invented to probe in-context learning (Section 3.9)

**Arithmetic, 175B (Table 3.9, accuracy %):**

| | 2D+ | 2D− | 3D+ | 3D− | 4D+ | 4D− | 5D+ | 5D− | 2D× | 1DC |
|---|---|---|---|---|---|---|---|---|---|---|
| zero-shot | 76.9 | 58.0 | 34.2 | 48.3 | 4.0 | 7.5 | 0.7 | 0.8 | 19.8 | 9.8 |
| one-shot | 99.6 | 86.4 | 65.5 | 78.7 | 14.0 | 14.0 | 3.5 | 3.8 | 27.4 | 14.3 |
| **few-shot** | **100.0** | **98.9** | **80.4** | **94.2** | 25.5 | 26.8 | 9.3 | 9.9 | 29.2 | 21.3 |

The authors checked that few-shot answers were rarely in the training data, so these are not memorised.

**Word scrambling (Table 3.10, few-shot %):**

| task | accuracy |
|---|---|
| cycled letters | 37.9 |
| anagrams (all but first/last letters) | 15.1 |
| anagrams (all but first/last two letters) | 39.7 |
| random insertion | 67.2 |
| reversed words | 0.44 |

**Reversed words are almost impossible.** BPE tokens hide the individual letters.

**News generation (Table 3.11):** humans identified GPT-3 175B's ~200-word articles as machine-written only **52%** of the time, near chance; for a deliberately bad control model the figure was 86%.

---

## 6. Contamination (Section 4)
- The training data is a big slice of the internet, so some test sets may have been seen.
- The authors tried to remove all overlaps, but **a bug left some in**, and retraining was unaffordable.
- **Their response:**
  - mark every test example as **"dirty"** if any of its 13-grams appears in the training data (fewer for short examples);
  - re-evaluate on the **clean** subset;
  - for most benchmarks the scores barely changed;
  - a few results were flagged or dropped.

---

## 7. Limitations the paper admits (Section 5)
- **Text synthesis:** samples still repeat themselves and lose coherence over long passages.
- **Comparisons:** it is weak at comparison tasks such as WiC and NLI ("are these two sentences the same?").
- **Objective:** autoregressive only, with no bidirectionality; the objective treats every token as equally important.
- **Sample efficiency:** it sees far more text in pre-training than a human sees in a lifetime.
- **Unclear mechanism:** it is unclear whether few-shot learning *learns* new tasks or *recognises* tasks seen during training.
- **Cost:** expensive to run.
- **Broader impacts** (Section 6): misuse, bias in gender, race and religion, and energy use.

---

## 8. Why it works (the intuition)

1. **Pre-training covers a huge space of tasks.** Web text contains countless implicit "here are some examples, now continue" patterns, so the model learns to continue patterns it is shown.
2. **Prediction needs the context.** To predict the next token of a list of translations, you must infer *what the list is doing* from the examples above. That inference is in-context learning.
3. **Scale makes the inference better.** Larger models represent more patterns and use their context more effectively. That's why the few-shot advantage grows with size.

---

## 9. What our code found

All numbers come from `demo.py` (about 5 s) and `test_gpt3.py` (10 tests, about 0.4 s).

1. **Table 2.1:**
   - 12·L·d² plus embeddings reproduces every size within 2% (175B → 174.6B);
   - 6ND gives 3,646 petaflop/s-days for 175B (paper: about 3,640);
   - the paper's table has two inconsistent head counts (XL, 13B).
2. **In-context learning, measured exactly:**
   - every test sequence draws its own random token distribution, so prediction is only possible by learning it from the context;
   - the loss falls with the number of tokens seen (nats per token):

     | tokens seen | d = 8 model | d = 32 model | Bayes-optimal |
     |---|---|---|---|
     | 1 | 2.27 | 2.17 | 2.13 |
     | 8 | 1.78 | 1.72 | 1.71 |
     | 47 | 1.62 | 1.52 | 1.50 |

   - The wider model sits within 0.02 nats of the Bayes-optimal learner, and closer than the narrow one.

     **This is learning without weight updates.** The trained network runs a near-optimal estimation procedure in its forward pass.
3. **Prompts and scoring:** exact zero-, one- and few-shot prompt layouts. The three multiple-choice scoring rules give different answers on the Paris example, as in Section 4.
4. **Data hygiene:**
   - **Pareto filter:** keeps 91% / 42% / 2.5% / 0.2% of documents scoring 0.99 / 0.9 / 0.5 / 0.1;
   - **MinHash:** dedup removes the near-duplicate;
   - **13-gram check:** flags a copied example.

**Honest note:** our first two attempts at an in-context learning demo failed.
- (a) Learning y = (a·x + b) mod 10 rules from examples stayed near chance.
- (b) Recalling a per-sequence random mapping stayed exactly at chance (loss stuck at ln 8) for 5000 steps.
- Tiny transformers learn these copy-and-compute skills only after long plateaus. The Dirichlet task needs only *averaging* over the context, which a 1-layer model learns in seconds.

**Not run (too heavy):**
- E1: power-law loss vs compute for 6 sizes;
- E2: zero / one / few-shot accuracy vs size on arithmetic and word scrambling;
- E3: in-context curves vs size;
- E4: quality filter + Pareto + MinHash on real text;
- E5: clean vs dirty accuracy.

---

## 10. Check yourself

1. What is the difference between few-shot learning in GPT-3 and fine-tuning?
   <details><summary>Answer</summary>Few-shot puts K examples in the prompt and runs one forward pass, with no weights changed. Fine-tuning runs gradient descent on the labelled examples, changing the weights.</details>
2. Estimate the training FLOPs of a 6.7B model trained on 300B tokens, in petaflop/s-days.
   <details><summary>Answer</summary>6 × 6.7·10⁹ × 3·10¹¹ = 1.21·10²² FLOPs; divided by 8.64·10¹⁹ that is ≈ 140 PF-days.</details>
3. A document's classifier score is 0.8. What is its chance of passing the Pareto(α = 9) filter?
   <details><summary>Answer</summary>P(Lomax > 0.2) = 1.2⁻⁹ ≈ 19.4%.</details>
4. Two documents' MinHash signatures agree in 7 of 10 slots. What is the estimated Jaccard similarity?
   <details><summary>Answer</summary>0.7.</details>
5. Why does GPT-3 struggle with reversing the letters of a word?
   <details><summary>Answer</summary>It sees BPE tokens (chunks of letters), not individual letters, so it doesn't directly "see" the characters it must reverse.</details>
6. Why normalise multiple-choice scores by the number of tokens?
   <details><summary>Answer</summary>Every extra token multiplies in another probability below 1, so longer answers get lower total probability even when they are right. The per-token average removes that bias.</details>
