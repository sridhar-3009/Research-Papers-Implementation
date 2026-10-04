# Atlas, explained simply

**Paper:** Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, Edouard Grave (Meta AI, ENS/PSL, Inria, UCL), *Atlas: Few-shot Learning with Retrieval Augmented Language Models*, 2022 (JMLR 2023).

**In one sentence:** pre-train a retrieval-augmented language model **together with its retriever** on unlabelled text. The result is an excellent **few-shot** learner: with 11B parameters and **64 examples** it gets **42.4%** on Natural Questions, beating the **540B**-parameter PaLM (39.6%) with 50× fewer parameters.

---

## 1. The question
- **Large language models are good few-shot learners** (GPT-3, PaLM), but they need huge parameter counts to **memorise** world knowledge.
- **Can a smaller model look facts up instead?** Earlier retrieval models (REALM, RAG; papers 070–071) were mostly tested with **lots of fine-tuning data**. Atlas asks whether retrieval also brings **few-shot** ability, and what it takes.
- **The answer:** yes, but the language model and retriever must be **pre-trained jointly**, so the language model already knows how to *use* documents before it sees the 64 examples.

---

## 2. The architecture

### Retriever: Contriever
- **A dual encoder** (like DPR, paper 070):
  - a transformer encodes the query and each document **separately**;
  - each vector is the **average** of the last layer's token outputs;
  - the score is the **dot product** s(d, q).
- **No labelled data:** Contriever is pre-trained **unsupervised** with a contrastive (MoCo) loss. Two random crops of the same document should embed close together; other documents should be far apart.

### Language model: T5 with Fusion-in-Decoder (FiD)
- **The encoder sees each retrieved document separately:** "question + document k" is encoded on its own for each of the K documents.
- **The decoder sees them all at once:** the K encoder outputs are **concatenated**, and the decoder cross-attends over the whole concatenation while generating the answer.
- **Why FiD:** putting all documents into one long input would cost **quadratic** self-attention in K. FiD's encoder cost is **linear** in K, and the decoder can still **fuse** evidence across documents.

---

## 3. Training the retriever from the language model (Section 2.2)
- **The goal:** if the language model finds a document useful, the retriever should rank it higher. No document labels are needed, only (query, output) pairs.
- **Retriever distribution** over the top-K documents, with temperature θ:
```
p_retr(d_k | q) = exp(s(d_k, q)/θ) / Σ_i exp(s(d_i, q)/θ)
```
- **The four losses:** each builds a target from the language model (with no gradient into the LM) and pulls p_retr toward it.

| Loss | Target / objective | Intuition |
|---|---|---|
| **ADist** (attention distillation) | per document: Σ over its tokens of α_n·‖v_n‖, averaged over heads, layers and output tokens, then softmax | the decoder pays attention to useful documents. The value norm ‖v_n‖ is included because a token with large attention but a tiny value vector contributes little |
| **EMDR²** | maximise log Σ_k p_LM(a \| q, d_k)·p_retr(d_k) | documents are latent variables, as in RAG |
| **PDist** (perplexity distillation) | p_k = softmax_k( log p_LM(a \| q, d_k) ) | how well does each document **alone** let the LM produce the answer? |
| **LOOP** (leave-one-out) | p_k = softmax_k( −log p_LM(a \| q, D \ {d_k}) ) | how much **worse** does the answer get when d_k is removed? |

- **For ADist, PDist and LOOP:** minimise KL(target ‖ p_retr).
- **Result:** in the paper all four perform **similarly**. **PDist** was adopted because it is stable and cheaper than LOOP.

### Worked PDist example
- **Setup:** three retrieved documents give log p_LM(a|q, d) = −1, −3 and −3.
- **Target:** softmax([−1, −3, −3]) = [e⁻¹, e⁻³, e⁻³]/Σ = [0.368, 0.050, 0.050]/0.468 = **[0.787, 0.106, 0.106]**.
- **Effect:** if the retriever currently gives [0.2, 0.5, 0.3], the KL pushes the first document's score up.

---

## 4. Joint pre-training (Section 2.3)
- **The data:** self-supervised, from unlabelled text. The "query" is a piece of text with something hidden; the model must retrieve documents that help fill it in.
- **The paper compared three pretext tasks** (64-shot average):

| Pretext task | 64-shot average |
|---|---|
| Prefix language modelling: continue the text | 40.1 |
| **Masked language modelling:** fill in masked spans | **42.4** |
| Title-to-section generation | 40.8 |

  Masked LM was slightly best and was adopted.
- **Keeping the index fresh:** the document encoder changes during pre-training, so the stored vectors go stale. Atlas **re-indexes** every 1,000 steps, about 30% overhead.
- **Ways to treat the retriever during fine-tuning:**
  - **full re-indexing;**
  - **re-ranking:** retrieve L documents with the stale index, re-embed only those, keep the top K;
  - **query-side fine-tuning:** update only the query encoder, so the index never changes.
- **Table 4:**
  - a **fixed retriever** is clearly worse;
  - re-ranking is about as good as full re-indexing;
  - query-side fine-tuning is **best at 64 shots** (less over-fitting), while full fine-tuning wins with more data.

---

## 5. Results
- **The ablations (Table 1)** show:
  - the **closed-book** baseline (T5 without retrieval, same pre-training) is poor;
  - models **without joint pre-training** are much worse at few-shot;
  - joint **retriever** training strongly improves the pre-training (masked-LM) metric, but the effect is "less marked" at 64 shots and "almost non-existent" at 1,024.

| Benchmark | Atlas-11B | Comparison |
|---|---|---|
| NQ, 64-shot | **42.4%** (45.1% with a Wikipedia-only index) | PaLM-540B 39.6%, Chinchilla-70B 35.5% |
| NQ, full data | 60.4%, **64.0%** with the Dec-2018 index | new state of the art (+8.1) |
| MMLU 5-shot | beats GPT-3 by 4 points | GPT-3 has 15× more parameters and 10× more pre-training compute |
| MMLU zero-shot (de-biased) | 47.1% | > GPT-3 5-shot 43.9% |

- **Temporal updating (TempLAMA, Table 11):**
  - trained on 2017 answers with a 2017 index: Atlas **57.7%** vs closed-book T5 **12.1%**;
  - **swapping in a 2020 index** without retraining lifts 2020 accuracy to **53.1%**;
  - T5 cannot be updated (3.6% on 2020 even when trained on 2020 answers).
- **Index compression:** product quantisation shrinks the Wikipedia index from **49 GB to 4 GB** (fp16) with negligible accuracy loss.

---

## 6. Product quantisation (PQ) in one paragraph
- **How it works:**
  - split each d-dimensional vector into m sub-vectors;
  - for each sub-vector position, run k-means to get 2^b centroids;
  - store, for each vector, only the m centroid **indices** (b bits each).
- **Example:** d = 64 floats (256 bytes) with m = 16 and b = 8 becomes **16 bytes** (16× smaller).
- **Search** compares the query against reconstructed vectors, or faster against precomputed query–centroid tables.
- **The trade-off:** fewer bits mean coarser vectors and lower recall.

---

## 7. Why it matters
- **Retrieval can stand in for scale:** an 11B model with a good index beats a 540B closed-book model on knowledge-heavy tasks.
- **Few-shot retrieval needs pre-training:** retrieval-augmented models need retrieval-aware pre-training to become good few-shot learners.
- **Practical benefits:** updatable knowledge (swap the index), interpretable evidence (the retrieved passages), and compressible memory (PQ).
- **The "retriever trained from the reader" losses** (especially PDist and EMDR²) are widely reused.

---

## 8. What our code found

### Setup
- **Corpus:** 300 invented people × 4 facts, with **each fact written twice** in two different templates, plus 900 filler passages (3,300 in all).
- **Masked-LM pre-training:** hide the value in one passage (e.g. "kamarka was born in `<mask>` …" → "hanoi"). The passage itself is excluded from retrieval, so the answer must come from the **other** passage.
- **Reader:** a one-head FiD. Each document's tokens are encoded with the query; one decoder query attends over all documents' tokens.
- **Retriever:** pooled word embeddings with per-word weights. The **unsupervised start** uses idf weights from corpus counts (no labels): 97% masked-LM recall@5; 78% / 76% on overlap / paraphrase QA queries.
- **Honest note:** our Contriever-style training on random crops made it **worse** (48%), so the main pipeline starts from the idf retriever. The weakened one is reused to show retriever learning.

### 64-shot QA
Pre-training is 800 masked-LM steps. Fine-tuning uses 64 questions (half paraphrased) with query-side PDist. The test is 800 questions about 100 people not used in fine-tuning.

| Model | Masked-LM acc | Recall@5 | EM overlap | EM paraphrase |
|---|---|---|---|---|
| Closed-book (same pre-training, no retrieval) | 0.20 | – | 19.0% | 18.0% |
| Retrieval, **no** joint pre-training | – | 0.97 | 7.2% | 7.8% |
| Joint pre-training, fixed retriever | 0.90 | 0.97 | 48.5% | 34.2% |
| **Joint pre-training, PDist retriever** | 0.87 | 0.96 | **53.8%** | **37.8%** |

- **Joint pre-training is what makes few-shot work:** 48–54% vs 7% without it. This is the paper's central finding.
- **Closed-book is weak:** it only knows facts it memorised.
- **PDist vs fixed** is a small, single-seed difference. This retriever already finds the right passage for 97% of masked-LM queries.

### The reader teaching a weak retriever
Starting from the crop-trained retriever (48% masked-LM recall@5), after 600 joint steps:
- **EMDR²:** recall **73%**, accuracy 0.36. The reader pulled useful passages up the ranking.
- **PDist:** 43%, accuracy 0.28. It **did not help** here: from a weak start, the untrained reader's per-document perplexities are nearly flat.

### The targets on one example
With the trained reader, for "kamarka was born in `<mask>` …" → hanoi (top-5 documents: the other birthplace passage, a "met … in delhi" filler, a "letter … about hanoi" filler, the sitar passage, and an unrelated passage):

| Loss | Target over the 5 documents |
|---|---|
| PDist | [0.321, 0.018, 0.321, 0.146, 0.195]: favours the two passages that mention hanoi |
| LOOP | [0.247, 0.014, 0.247, 0.247, 0.247]: nearly flat, since dropping one of five documents barely matters, except that it down-weights the delhi filler |
| ADist | [0, 1, 0, 0, 0]: our one-head attention points at the **delhi filler**, so ADist was the unreliable target in our toy |

### Index swap
Everyone changes job, instrument and team; the same model is used with the index rebuilt.

| | Old answers | New answers |
|---|---|---|
| Old index | **56.3%** | 4.0% |
| New index | 27.0% | **32.7%** |

- **The swap moves answers to the new facts, but only partly.** With the new index the model still gives the **old** answer 27% of the time: our reader memorised facts during pre-training and sometimes trusts that memory over the passage.
- **The paper's Atlas conditions more faithfully** on its index (57.7% → 53.1%).

### PQ compression (recall@5 of the fine-tuned model)

| Setting | Recall@5 |
|---|---|
| float32 (256 B) | 86% |
| 16 B (16×) | 85% |
| 8 B (32×) | 79% |
| 4 B (64×) | 72% |
| 2 B (128×) | 21% |

16× costs almost nothing; harder compression degrades quickly.

**`experiments.py`:**
- **E1:** all four losses from both starts, 3 seeds;
- **E2:** fine-tuning strategies, including re-ranking;
- **E3:** 16 … 1,024 shots;
- **E4:** the number of documents k;
- **E5:** partial index swaps;
- **E6:** a full PQ grid;
- **E7:** real Contriever plus a FLAN-T5 FiD on NQ with 64 examples.
- None were run here.

---

## 9. Check yourself

1. What does Fusion-in-Decoder do, and why?
<details><summary>Answer</summary>The encoder processes each (question, document) pair separately; the decoder cross-attends over all encoded documents concatenated. Encoder cost grows linearly with the number of documents (instead of quadratically for one long input), and the decoder can still combine evidence across documents.</details>

2. Compute the PDist target for log p_LM = [−2, −2, −4].
<details><summary>Answer</summary>softmax: e⁻², e⁻², e⁻⁴ = 0.135, 0.135, 0.018, summing to 0.289, giving [0.468, 0.468, 0.063].</details>

3. What does LOOP measure, and why can it be nearly flat?
<details><summary>Answer</summary>How much worse the answer becomes when one document is removed from the set. If several documents carry the answer (or none does), removing any single one changes little, so all scores are similar.</details>

4. Why does ADist multiply attention by ‖v_n‖?
<details><summary>Answer</summary>A token's contribution to the attention output is α_n·v_n. A large weight on a token with a tiny value vector contributes little, so α_n‖v_n‖ is a better relevance measure than α_n alone.</details>

5. What is the main finding about pre-training?
<details><summary>Answer</summary>Few-shot performance depends on joint retrieval-augmented pre-training: the language model learns to use and aggregate retrieved documents. Without it, a retrieval model with 64 examples does badly (in our toy, 7% vs 48–54%).</details>

6. Name three ways to handle the retriever during fine-tuning.
<details><summary>Answer</summary>Full fine-tuning with re-indexing; re-ranking (retrieve L with the stale index, re-embed and re-score them, keep K); query-side fine-tuning (update only the query encoder, so the index stays valid). A fixed retriever is the baseline.</details>

7. Why is query-side fine-tuning attractive with 64 examples?
<details><summary>Answer</summary>No re-indexing is needed (cheap), and fewer trainable retriever parameters means less over-fitting on tiny data. The paper found it slightly better than full fine-tuning at 64 shots.</details>

8. How does PQ store a 64-dimensional vector in 16 bytes?
<details><summary>Answer</summary>Split it into 16 sub-vectors of 4 dimensions. For each position, learn 256 centroids by k-means and store the 1-byte index of the nearest centroid: 16 indices × 1 byte.</details>

9. In the TempLAMA experiment, why can't closed-book T5 be updated like Atlas?
<details><summary>Answer</summary>Its knowledge is in its weights; changing it needs retraining with new data. Atlas reads facts from its index, so swapping the 2017 index for a 2020 one changes the answers (to 53.1% on 2020) without any training.</details>

10. In our toy, the model still gave old answers 27% of the time after the index swap. Why?
<details><summary>Answer</summary>During pre-training it saw every fact many times and partly memorised them in its weights. When the retrieved passage disagrees with that memory, it sometimes follows the memory. That shows parametric and retrieved knowledge competing.</details>
