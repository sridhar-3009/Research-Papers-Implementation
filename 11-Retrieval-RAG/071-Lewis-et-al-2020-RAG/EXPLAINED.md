# Retrieval-Augmented Generation (RAG), explained simply

**Paper:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (Facebook AI Research, UCL, NYU), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020.

**In one sentence:** combine a **parametric memory** (a seq2seq language model, BART) with a **non-parametric memory** (a dense vector index of Wikipedia, searched by DPR). Treat "which document to read" as a **hidden variable** that is summed out, and train everything end to end from input/output pairs alone.

---

## 1. Why retrieval?
- **Language models store knowledge in their weights** (parametric memory). That has drawbacks:
  - it is **hard to update:** new facts need retraining;
  - **you can't see where an answer came from;**
  - models **hallucinate** facts they don't really know.
- **A retriever plus an index** (non-parametric memory) fixes these:
  - knowledge sits in plain text you can read, cite and **replace**;
  - the generator just has to **read** it.
- **Earlier hybrids** (REALM, ORQA) were limited to **extractive** QA: they could only copy a span from the passage. RAG makes the hybrid a general-purpose **generator**. It can write free text, generate questions and classify claims.

---

## 2. The model

### Two components
1. **Retriever p_η(z | x): DPR** (paper 070).
```
p_η(z | x) ∝ exp( d(z)ᵀ q(x) ),   d(z) = BERT_d(z),  q(x) = BERT_q(x)
```
   Top-k documents come from a FAISS index over **21M** 100-word Wikipedia passages (Dec 2018 dump).
2. **Generator p_θ(y_i | x, z, y_<i): BART-large** (400M parameters). The input is the question concatenated with the document.

### The document is a latent variable
- **No training label says which document to read.** RAG **sums over** the top-k documents, weighted by the retriever's probabilities.
- **Two ways to do it:**

**RAG-Sequence:** use the **same document for the whole answer.**
```
p(y | x) ≈ Σ_{z ∈ top-k} p_η(z | x) · Π_i p_θ(y_i | x, z, y_<i)
```
**RAG-Token:** each **token** may come from a **different document.**
```
p(y | x) ≈ Π_i  Σ_{z ∈ top-k} p_η(z | x) · p_θ(y_i | x, z, y_<i)
```

### Worked example (from our demo)
- **Setup:** 2 documents with p(z|x) = [0.7, 0.3].
- **Document 1** supports token 1 (probability 0.9) but not token 2 (0.1). **Document 2** is the reverse (0.2, 0.8).
- **RAG-Sequence:** 0.7·(0.9·0.1) + 0.3·(0.2·0.8) = 0.063 + 0.048 = **0.111**.
- **RAG-Token:** (0.7·0.9 + 0.3·0.2) · (0.7·0.1 + 0.3·0.8) = 0.69 · 0.31 = **0.214**.
- **Result:** RAG-Token can stitch an answer together from several documents. RAG-Sequence needs **one** document that supports everything.

### Training
- **The loss:** minimise −log p(y | x) with Adam.
- **What gets updated:** the **question encoder BERT_q** and **BART**. The gradient reaches the retriever through p_η(z|x): documents that help the generator get pushed up.
- **What stays frozen:** the document encoder and the index. Updating them would require re-indexing 21M passages, which REALM did; RAG found it unnecessary.
- **k:** 5 or 10 documents per example during training.

### Decoding
- **RAG-Token:** a normal beam search on the mixed next-token distribution Σ_z p(z|x) p(y_i|x, z, y_<i).
- **RAG-Sequence:** the likelihood doesn't split per token, so:
  - **"Thorough decoding":** run beam search **per document** to get candidate answers. Score every candidate under **every** document, doing extra forward passes where needed, and sum.
  - **"Fast decoding":** skip those extra passes, assuming p(y|x, z) ≈ 0 when y wasn't in document z's beam.

---

## 3. Results

### Open-domain QA (Table 1, exact match)

| Model | NQ | TriviaQA | WebQuestions | CuratedTREC |
|---|---|---|---|---|
| T5-11B closed-book | 34.5 | – / 50.1 | 37.4 | – |
| REALM | 40.4 | – | 40.7 | 46.8 |
| DPR (extractive) | 41.5 | 57.9 / – | 41.1 | 50.6 |
| **RAG-Token** | 44.1 | 55.2 / 66.1 | **45.5** | 50.0 |
| **RAG-Sequence** | **44.5** | **56.8 / 68.0** | 45.2 | **52.2** |

- **It generates answers, not just copies them.** When the correct answer is in **no** retrieved document, RAG still gets **11.8%** of NQ right. An extractive reader would get 0%.

### Generation and classification (Table 2)
- **MS-MARCO (abstractive QA):** RAG-Sequence beats BART by 2.6 BLEU and 2.6 ROUGE-L, and **hallucinates less**.
- **Jeopardy question generation:** RAG-Token is best.
  - Human evaluation over 452 pairs: **RAG more factual in 42.7%** of cases vs BART in **7.1%**.
  - RAG is also judged more specific.
- **FEVER fact verification:** within 4.3% of complex pipeline systems (3-way), without retrieval supervision. The top retrieved document comes from a gold article 71% of the time.
- **Diversity:** RAG generations are more diverse than BART's.

### Ablations (Table 6, NQ dev EM)

| Retriever | RAG-Token | RAG-Sequence |
|---|---|---|
| BM25 | 29.7 | 31.8 |
| Frozen DPR | 37.8 | 41.2 |
| **Learned DPR** | **43.5** | **44.0** |

- **Learned retrieval improves every task.** The exception is BM25 on FEVER, whose claims are entity-centric and so suit word overlap.

### Other findings
- **Index hot-swapping:** RAG asked "Who is {position}?" about 82 world leaders who changed between 2016 and 2018.
  - Matched indexes: **70%** (2016 index, 2016 leaders) and **68%** (2018 index, 2018 leaders).
  - Mismatched indexes: **12%** and **4%**.
  - **Knowledge can be updated by swapping the index**, without retraining.
- **More documents at test time:** NQ improves monotonically with k for RAG-Sequence; RAG-Token peaks at k = 10.
- **Parametric and non-parametric memory work together.** In the Jeopardy example (Figure 2), the document posterior is high on the document that mentions "The Sun Also Rises" when generating "Sun". It flattens once the title has started, because BART's own memory completes it.

---

## 4. Why it matters
- **"RAG" became the name for a whole pattern:** retrieve relevant text and condition the generator on it. It is the backbone of document question answering, enterprise search assistants and citation-giving chatbots.
- **It established the core ideas:**
  - non-parametric memory you can update (hot-swap);
  - a retriever trained only through the generator's loss;
  - documents as latent variables.
- **What modern practice usually does differently:** it puts retrieved text straight into an LLM's prompt (often with no joint training), but the motivations are the same: freshness, provenance, and fewer hallucinations. Atlas (paper 072) pushes the joint-training idea further.

---

## 5. What our code found

### Setup
- **World:** the toy Wikipedia of paper 070 (500 people × 4 facts + 1,500 filler passages).
- **Retriever:** DPR pre-trained on **overlap-style questions only**. It knows "where was X born" but not the paraphrase "hometown".
- **Generator:** a tiny MLP over (document words, question words, previous token), trained from scratch.
- **Training:** RAG fine-tuning on 1,120 question/answer pairs, k = 5, with **no document labels**.

| Model | EM paraphrase | EM overlap | Gold in top-5 (para / overlap) |
|---|---|---|---|
| Closed-book (no retrieval) | 9.0% | 8.3% | – |
| RAG, BM25 retriever | 22.9% | 73.1% | 75.6% / 100% |
| RAG, frozen DPR | 11.5% | 79.2% | 9.8% / 97.1% |
| **RAG, retriever learned end to end** | **57.5%** | **81.9%** | **75.8%** / 96.0% |

- **Closed-book can only recall what it memorised.** It gets 66% of its own training questions but guesses on held-out facts.
- **Learned retrieval wins, as in the paper's Table 6.** Training the question encoder through the **generator's loss alone** taught it the paraphrases: gold-in-top-5 rose from 9.8% to 75.8%.
- **Honest note:** our from-scratch generator needed **200 warm-up steps** (generator only) before the retriever could learn. Without them, early random gradients wrecked the retriever: recall fell from 72% to 1% in a trial run. The paper's pre-trained BART can read from step 1.

**Test-time k** (paraphrase EM):

| k | 1 | 2 | 5 | 10 |
|---|---|---|---|---|
| EM | 61.0% | 60.8% | 57.5% | 56.0% |

This is the opposite of the paper's trend: our tiny generator is distracted by extra documents.

**Index hot-swap** (every person changes job, instrument and team; we re-encode the new passages with the same frozen encoder and leave the model untouched):

| | Old facts | New facts |
|---|---|---|
| **Old index** | **93.8%** | 0.6% |
| **New index** | 3.9% | **88.8%** |

Swapping the index updates what the model "knows". This is the same pattern as the paper's 70 / 68 vs 12 / 4.

**Two facts from two documents** ("where was X born and what does X work as?" → "kyoto pilot"):
- **Setup:** a **fixed reader** isolates the marginalisation. It gives 0.9 to a value its document states, and guesses uniformly otherwise. Both gold passages are in the top 5 for 72% of questions.

| Method | Exact match |
|---|---|
| RAG-Sequence, thorough decoding | 10.0% |
| RAG-Sequence, fast decoding | 4.0% |
| **RAG-Token** | **73.0%** |

- **Why:** no single passage has both facts, so RAG-Sequence must guess one half. RAG-Token takes "kyoto" from the birth passage and "pilot" from the job passage.
- **In the posterior table** (Figure 2 analogue), the job passage's weight for token 2 rises from a prior of 0.003 to 0.041.
- **With a trained tiny generator** this task failed for both models: with 300 examples it **memorised** the answers instead of learning to read. `experiments.py` E4 retries with 2,000 people.

**`experiments.py`:**
- **E1:** Table 6, multi-seed, both modes;
- **E2:** test-time k;
- **E3:** warm-up and retriever learning-rate fragility;
- **E4:** the two-fact task with a trained generator;
- **E5:** partial hot-swaps;
- **E6:** the real `facebook/rag-sequence-nq` / `rag-token-nq` models on NQ.
- E6 needs large downloads. None were run here.

---

## 6. Check yourself

1. What are the "parametric" and "non-parametric" memories in RAG?
<details><summary>Answer</summary>Parametric: BART's weights, i.e. knowledge learned in training. Non-parametric: the dense index of 21M Wikipedia passages that the retriever searches, which can be read and replaced.</details>

2. Write the RAG-Sequence and RAG-Token probabilities.
<details><summary>Answer</summary>RAG-Sequence: Σ_z p(z|x) Π_i p(y_i|x, z, y_<i). RAG-Token: Π_i Σ_z p(z|x) p(y_i|x, z, y_<i).</details>

3. p(z|x) = [0.5, 0.5]; doc 1 gives tokens (0.8, 0.2), doc 2 gives (0.2, 0.8). Compute both.
<details><summary>Answer</summary>Sequence: 0.5·0.16 + 0.5·0.16 = 0.16. Token: (0.5·0.8 + 0.5·0.2)·(0.5·0.2 + 0.5·0.8) = 0.5·0.5 = 0.25.</details>

4. How does the retriever learn without labels for which document is relevant?
<details><summary>Answer</summary>The loss −log Σ_z p(z|x) p(y|x, z) is differentiable in the retriever scores. Documents under which the generator assigns high probability to the right answer get their p(z|x) increased.</details>

5. Why keep the document encoder frozen?
<details><summary>Answer</summary>Changing it would change every passage vector, requiring the 21M-passage index to be rebuilt periodically (as REALM does). RAG found that fine-tuning only the query encoder works well.</details>

6. What is the difference between thorough and fast decoding?
<details><summary>Answer</summary>Both beam-search each document separately to get candidates. Thorough decoding then scores each candidate under every document (extra forward passes); fast decoding assumes zero probability under documents whose beam didn't produce it.</details>

7. What did the hot-swap experiment show?
<details><summary>Answer</summary>Replacing the 2018 Wikipedia index with a 2016 one, without retraining, changes the answers to match that time: 70% / 68% with matched indexes vs 12% / 4% mismatched. Knowledge can be updated by swapping documents.</details>

8. Why did BM25 work best on FEVER?
<details><summary>Answer</summary>FEVER claims are entity-centric (they name the article's subject), so exact word overlap finds the right article well.</details>

9. How can RAG answer correctly when the answer isn't in any retrieved document?
<details><summary>Answer</summary>The generator also has parametric knowledge. Retrieved passages can give clues that trigger what BART already knows (11.8% of such NQ cases), which an extractive reader can't do.</details>

10. In our toy, why did fine-tuning the retriever help paraphrased questions so much?
<details><summary>Answer</summary>The pre-trained retriever had never seen words like "hometown", so it often missed the gold passage (9.8% in top-5). The generator's loss rewarded retrieving passages it could answer from, so the question encoder learned the paraphrases (75.8% in top-5) and exact match rose from 11.5% to 57.5%.</details>
