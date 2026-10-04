# Dense Passage Retrieval (DPR), explained simply

**Paper:** Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih (Facebook AI, University of Washington, Princeton), *Dense Passage Retrieval for Open-Domain Question Answering*, EMNLP 2020.

**In one sentence:** to find the Wikipedia passages that answer a question, turn the question and every passage into **vectors** with two BERT encoders and pick the passages with the **largest dot product**. Trained with a simple trick ("in-batch negatives") on just question–passage pairs, this **beats the keyword search BM25** by 9–19 points of top-20 accuracy and set a new state of the art for open-domain QA.

---

## 1. The problem: open-domain question answering
- **The task:** "Who plays Thoros of Myr in Game of Thrones?" must be answered from **all of Wikipedia**, with no passage given.
- **The usual pipeline: retriever then reader.**
  1. A **retriever** picks a few passages (say 20–100) out of millions.
  2. A **reader** (a BERT model) reads them and extracts the answer span.
- **The retriever matters most:** if the right passage isn't retrieved, the reader can't answer.

### The classic retriever: BM25 (sparse, keyword matching)
- **The idea:** score a passage by how many question words it contains, giving **rare words more weight**.
- **The formula** (one term per question word t):
```
BM25(q, d) = Σ_t idf(t) · tf(t, d)·(k1 + 1) / ( tf(t, d) + k1·(1 − b + b·|d|/avgdl) )
idf(t) = log(1 + (N − df(t) + 0.5)/(df(t) + 0.5))
```
  - **tf(t, d):** how often t appears in passage d.
  - **df(t):** how many passages contain t.
  - **N:** the number of passages.
  - **|d| / avgdl:** the passage length relative to average.
  - **k1:** saturates repeated words; **b:** normalises for length. The paper tuned k1 = 0.9 and b = 0.4.
- **Weakness:** no understanding of **synonyms or paraphrases**. The paper's example: "Who is the bad guy in lord of the rings?" should retrieve "…best known for portraying the villain Sauron in the Lord of the Rings trilogy". Keyword search can't tell that "bad guy" means "villain"; a dense encoder can learn it.
- **Strength:** exact rare words and names ("Thoros of Myr") match precisely.

### Worked BM25 example (from our tests)
- **Setup:** passages "a b c", "a a d", "e f g h"; query "a"; N = 3; df(a) = 2.
- **idf** = log(1 + (3 − 2 + 0.5)/(2 + 0.5)) = log(1.6) = 0.470.
- **For "a a d":** tf = 2, |d| = 3, avgdl = 10/3.
  - Score = 0.470 · 2·1.9/(2 + 0.9·(0.6 + 0.4·0.9)) = 0.470 · 3.8/2.864 = **0.624**.
- **"a b c"** (tf = 1) scores lower, and **"e f g h"** scores 0.

---

## 2. DPR: dense vectors instead of keywords

### The model
- **Two independent BERT-base encoders:** E_Q for questions and E_P for passages. The vector is the **[CLS] token's output**, d = 768.
- **Similarity is a dot product:**
```
sim(q, p) = E_Q(q) · E_P(p)                                     (Eq. 1)
```
- **Why a dot product:** it is **decomposable**. All 21 million passage vectors can be computed **once, offline**. At question time:
  - encode only the question;
  - find the largest dot products with **FAISS**, a library for fast nearest-neighbour search over billions of vectors.
  - A cross-attention model that reads question and passage together would be more accurate per pair, but it would have to run on all 21M passages for every question.

### Training: metric learning with negatives
For each question q_i there is a positive passage p_i⁺ and some negatives p⁻. The loss is a softmax over them:
```
L = −log  e^{sim(q_i, p_i⁺)} / ( e^{sim(q_i, p_i⁺)} + Σ_j e^{sim(q_i, p_j⁻)} )                (Eq. 2)
```
The paper tries three kinds of negatives:
1. **Random** passages;
2. **BM25** passages: top keyword matches that **don't** contain the answer (hard negatives);
3. **Gold** passages: the positives of **other questions**.

### The key trick: in-batch negatives
- **The setup:** take B questions and their B gold passages. Encode them into matrices Q (B×d) and P (B×d), then compute
```
S = Q Pᵀ      (B × B)
```
- **Row i** scores question i against all B passages. Its positive is column i; the other B − 1 columns are **free negatives**.
- **The loss is cross-entropy** with target i for row i. You get B² question–passage pairs for the cost of encoding 2B texts.
- **The best model** adds **one BM25 negative per question**, which is shared as a negative for every question in the batch. With B = 128, each question has 127 + 128 = 255 negatives.

**Worked example:** B = 4 with 4 extra BM25 negatives gives S of size 4 × 8.
- If all 8 scores in a row are equal, the loss is ln 8 = 2.079.
- Training pushes the diagonal score up and the rest down, and the loss toward 0.

### Setup
- **Corpus:** English Wikipedia (Dec 2018), split into **100-word passages**: **21,015,324** passages.
- **Datasets:** Natural Questions, TriviaQA, WebQuestions, CuratedTREC and SQuAD.
- **Training:** batch 128 plus 1 BM25 negative per question; Adam at learning rate 1e-5 with linear warm-up; dropout 0.1; up to 40 epochs (100 for the small datasets).
- **Hybrid:** BM25(q, p) + λ·sim(q, p), re-ranking the union of each method's top 2000.

---

## 3. Results

### Retrieval (Table 2, top-k accuracy: a top-k passage contains the answer)

| Dataset | BM25 top-20 | DPR top-20 | BM25 top-100 | DPR top-100 |
|---|---|---|---|---|
| Natural Questions | 59.1 | **78.4** | 73.7 | **85.4** |
| TriviaQA | 66.9 | **79.4** | 76.7 | **85.0** |
| WebQuestions | 55.0 | **73.2** | 71.1 | **81.4** |
| CuratedTREC | 70.9 | **79.8** | 84.1 | **89.1** |
| SQuAD | **68.8** | 63.2 | **80.0** | 77.2 |

- **SQuAD is the exception.** Its questions were written by people **looking at the passage**, so they share many words with it, which favours BM25. SQuAD also covers only 500+ articles.
- **The hybrid** BM25 + DPR helps on some datasets, but not on NQ (76.6 vs 78.4 top-20).

### Ablations (Table 3, NQ dev, top-20)

| Training | Top-20 |
|---|---|
| 7 random / 7 BM25 / 7 gold negatives (no in-batch) | 64.3 / 63.3 / 63.1 |
| In-batch gold, batch 8 / 32 / 128 | 69.1 / 70.8 / 73.0 |
| In-batch 128 + 1 BM25 negative each | **78.0** |

- **In-batch negatives help, and more of them help more.** One BM25 hard negative helps a lot; two don't help further.
- **Sample efficiency:** DPR trained on just **1,000** NQ questions already beats BM25 (Figure 1).
- **Similarity:** dot product ≈ L2 distance > cosine. Triplet loss ≈ NLL.

### End to end (Table 4)
- With a BERT reader, NQ exact match is **41.5 vs 33.3** for ORQA, the previous best.

### Speed
- **DPR with FAISS:** **995 questions/s** (top-100).
- **BM25/Lucene:** 23.7 questions/s per CPU thread.
- **The cost is up front:** encoding 21M passages took about 8.8 hours on 8 GPUs, and building the FAISS index 8.5 hours, vs about 30 minutes for a Lucene index.

---

## 4. Why it matters
- **Dense retrieval showed it could beat BM25** with simple training. No special pre-training was needed (unlike ORQA and REALM).
- **It is the retriever inside RAG** (paper 071), Atlas (072), and most "chat with your documents" systems. Today's embedding models are its descendants: Contriever, E5, BGE, OpenAI embeddings.
- **The lessons carried over:**
  - in-batch negatives plus hard negatives became the standard recipe for training embedding models;
  - dense and sparse retrieval complement each other, so hybrid search is still common.

---

## 5. What our code found

### The toy corpus
- **3,500 passages:** 500 invented people with 4 facts each (birthplace, job, instrument, team), plus 1,500 filler passages that mention people in passing.
- **Two question styles:**
  - **overlap** ("where was X born ?"), like SQuAD;
  - **paraphrase** ("what is the hometown of X ?"), like Natural Questions.
- **Training:** 1,120 questions about 400 people.
- **Testing:** their held-out facts, plus 100 people never asked about in training.
- **The encoders:** two independent bag-of-words encoders, d = 128, starting from the **same** random weights. This mimics both BERTs starting from the same checkpoint; with unrelated random starts the model only memorised its training questions.

| Test set | BM25 top-1 / top-5 | DPR top-1 / top-5 | BM25 + 1.0·DPR top-1 / top-5 |
|---|---|---|---|
| Seen people, overlap | 81.2% / 95.8% | 86.7% / 96.0% | 99.4% / 100% |
| Seen people, **paraphrase** | 19.0% / 67.7% | **81.7% / 92.1%** | 98.3% / 99.8% |
| Unseen people, overlap | **83.0% / 95.5%** | 48.5% / 74.5% | 98.8% / 100% |
| Unseen people, paraphrase | 19.5% / 57.5% | 43.8% / 68.5% | 98.2% / 99.8% |

- **Paraphrases:** DPR crushes BM25 (82% vs 19% top-1). It learned that "hometown" means "born" and "profession" means "works as". BM25 can only match the name, which also appears in the person's other facts and in filler.
- **Rare names:** for people never seen in training, DPR drops to 48% on overlap questions, where BM25 gets 83%. An exact rare word is BM25's strength (the paper's SQuAD and "Thoros of Myr" points).
- **The hybrid** (λ = 1.0, picked on training questions) gets ≥ 98% top-1 everywhere. In the paper the hybrid helped less.

**Training schemes** (top-1 / top-5 on seen-people questions):

| Scheme | Top-1 | Top-5 |
|---|---|---|
| 7 random negatives | 71.1% | 89.7% |
| 7 BM25 negatives | **8.9%** | 18.9% |
| In-batch, batch 8 | 27.5% | 56.4% |
| In-batch, batch 32 | 91.4% | 98.5% |
| In-batch, batch 128 | 86.0% | 96.6% |
| In-batch 128 + 1 BM25 | 84.2% | 94.1% |

**Honest differences from the paper:**
- **BM25 negatives alone fail here.** In our toy the top BM25 non-answer is usually **another fact about the same person**, so training never has to match the name against the rest of the corpus.
- **Batch 32 beats 128.** With a fixed 40 epochs, small batches get 4× more steps; E1 in `experiments.py` equalises steps.
- **The extra BM25 negative doesn't help.**

**Sample efficiency** (top-5 on paraphrased questions):
- 50 questions: 16.2%; 200: 31.7%; 1,120: 92.1%; BM25: 67.7%.
- DPR needs the full set to beat BM25. The paper's encoders start from pre-trained BERT; ours don't.

**End-to-end exact match** with a rule-based reader (k = 1 / 5):

| Test set | BM25 | DPR | Hybrid |
|---|---|---|---|
| Seen people, paraphrase | 19.0% / 66.0% | 81.2% / 90.8% | 98.3% / 99.8% |
| Unseen people, overlap | 83.0% / 95.5% | 45.8% / 69.8% | 98.8% / 100% |

The reader can only answer from what was retrieved, so **retrieval quality is QA quality.**

**Speed:** dense search is one matrix product over pre-computed vectors, about 580k q/s vs about 3.4k q/s for our Python BM25.

**`experiments.py`:**
- **E1:** Table 3 with equal steps, multi-seed;
- **E2:** sample efficiency;
- **E3:** dot vs cosine vs L2;
- **E4:** shared vs separate initialisation;
- **E5:** a real BERT DPR on Natural Questions vs BM25.
- E1–E4 run in minutes on a CPU; E5 needs a GPU. None were run here.

---

## 6. Check yourself

1. Why does DPR use a dot product rather than a model that reads question and passage together?
<details><summary>Answer</summary>A dot product is decomposable: passage vectors are computed once offline and indexed (FAISS), so a question needs one encoder pass plus a fast nearest-neighbour search. A cross-attention model would need a full pass for every (question, passage) pair, 21M per question.</details>

2. With B = 128 and one BM25 negative per question, how many negatives does each question see?
<details><summary>Answer</summary>127 other gold passages plus 128 BM25 negatives (all shared), so 255.</details>

3. Why is the in-batch trick efficient?
<details><summary>Answer</summary>One matrix product S = QPᵀ scores all B² pairs, reusing the 2B encodings already computed. The negatives cost no extra encoder passes.</details>

4. Compute idf for a word that appears in 2 of 3 passages.
<details><summary>Answer</summary>log(1 + (3 − 2 + 0.5)/(2 + 0.5)) = log(1.6) ≈ 0.470.</details>

5. Why does BM25 beat DPR on SQuAD?
<details><summary>Answer</summary>Annotators wrote questions while reading the passage, so questions share many words with it (favouring keyword match), and the data comes from only 500+ articles (a biased training distribution).</details>

6. What is a "hard negative", and why is it useful?
<details><summary>Answer</summary>A passage that looks relevant (a high BM25 score, matching many question words) but does not contain the answer. It forces the encoder to learn finer distinctions than random negatives, which are easy to reject.</details>

7. In our toy, why did DPR fail on people never seen in training?
<details><summary>Answer</summary>The question encoder's embedding for an unseen name was never trained to match the passage encoder's. Only the shared initialisation links them, and it drifts during training. BM25 matches the exact rare word.</details>

8. Why might combining BM25 and DPR help?
<details><summary>Answer</summary>They fail differently: BM25 misses paraphrases, while DPR can miss exact rare terms. A weighted sum ranks a passage highly if it is good under either, as in our toy, where the hybrid got ≥ 98% top-1 everywhere.</details>

9. What does "top-20 accuracy" measure?
<details><summary>Answer</summary>The fraction of questions for which at least one of the 20 retrieved passages contains the answer string.</details>

10. What is the trade-off in DPR's speed?
<details><summary>Answer</summary>Search is very fast (995 q/s vs 23.7 for Lucene), but building the index is expensive: encoding 21M passages took about 8.8 GPU-hours × 8 GPUs, plus 8.5 hours of FAISS indexing, vs about 30 minutes for an inverted index. Updating the corpus also needs re-encoding.</details>
