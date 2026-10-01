# Vinyals, Toshev, Bengio & Erhan (2015), explained from scratch

**Paper:** *Show and Tell: A Neural Image Caption Generator*
**Authors:** Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan (Google)
**Published at:** CVPR 2015 (arXiv:1411.4555)

Read Paper 027 (Seq2Seq) first. This paper takes the same recipe and swaps the encoder for a CNN (Papers 013–016).

---

## The big idea in one line

> **Captioning is "translating" an image into a sentence. Use a pretrained CNN as the encoder, so that the image becomes one vector. Feed that vector once into an LSTM language model, which then writes the sentence word by word. Train the whole thing end to end to maximize log p(sentence | image).**

---

## 1. Why this was new (Sections 1–2)

- **Earlier systems were pipelines:**
  - detect objects and attributes;
  - fill a template ("There is a ___ on the ___") or stitch phrases together.

  They were hand-designed and rigid.
- **Other systems ranked existing captions** instead of writing new ones.
- **NIC (Neural Image Caption) is one network, trained by SGD.** It writes new sentences, and the vision and language parts can each be pretrained on more data.
- The closest work, Mao et al.'s m-RNN, feeds the image into the RNN at **every step**. NIC feeds it **once**, at the start.

---

## 2. The model (Section 3)

**The objective:**
```
θ* = argmax Σ_(I,S) log p(S | I; θ)                     (1)
log p(S | I) = Σ_t log p(S_t | I, S_0, …, S_{t−1})        (2)   chain rule over the words
```

**Unrolled (Eqs. 10–12, Figure 3):**
```
x_{−1} = CNN(I)                                  the image, only at t = −1
x_t    = W_e S_t      t = 0 … N−1                word embeddings; S_0 = start word, S_N = stop word
p_{t+1} = LSTM(x_t)
```

**The loss (Eq. 13):** L(I, S) = −Σ_{t=1..N} log p_t(S_t). The output at t = −1 is not scored; it only "tells the LSTM what is in the picture".

**Points to note:**
- **The image and the words live in the same 512-d space.** The CNN's top layer maps the image there, and W_e maps the words.
- **The image is fed only once.** "Feeding the image at each time step as an extra input yields inferior results, as the network can explicitly exploit noise in the image and overfits more easily."

**The LSTM (Eqs. 4–9):**
```
i_t = σ(W_ix x_t + W_im m_{t−1})      f_t = σ(W_fx x_t + W_fm m_{t−1})      o_t = σ(W_ox x_t + W_om m_{t−1})
c_t = f_t ⊙ c_{t−1} + i_t ⊙ tanh(W_cx x_t + W_cm m_{t−1})
m_t = o_t ⊙ c_t                       ← as printed: no tanh on c_t (the usual LSTM has o ⊙ tanh(c))
p_{t+1} = Softmax(m_t)
```
- **What the missing tanh means:**
  - With tanh, the output m is bounded in (−1, 1).
  - As printed, m = o·c can be as large as the cell value c, which itself grows by up to 1 per step (i·tanh(…) ≤ 1). So the output is **unbounded**.
- **It still trains** (our demo uses the literal form), but the bounded version is safer. The authors' later open-source code uses it.

### 2.1 The loss on a 3-word caption, by hand
Caption "a red circle" = S₀ = <start>, S₁ = "a", S₂ = "red", S₃ = "circle", S₄ = <stop>. Suppose the model gives the correct next word these probabilities:
```
p₁("a") = 0.9    p₂("red") = 0.5    p₃("circle") = 0.8    p₄(<stop>) = 0.95
L = −(ln 0.9 + ln 0.5 + ln 0.8 + ln 0.95) = 0.105 + 0.693 + 0.223 + 0.051 = 1.073 nats
```
- **The weak link** is "red". Only the image can tell the colour, so that's where the image information has to flow through the LSTM's memory, all the way from t = −1.
- **Scoring stops at the stop word.** The <stop> term teaches the model when a caption ends (Paper 027, section 2).

---

## 3. Inference

- **Sampling:** draw S₁ from p₁, feed it in, draw S₂, and so on until the stop word. The captions are diverse, but some are poor.
- **Beam search:** keep the k best partial sentences (Paper 027, section 5). **The paper uses k = 20; greedy search (k = 1) loses about 2 BLEU.**
- **N-best lists:** the beam's finished hypotheses, ranked. They show the model's alternative descriptions (Table 3).

---

## 3a. Ranking with a generative model: the Bayes view (Tables 4–5)

A captioner gives p(S | I). Retrieval tasks need two kinds of ranking:
- **Image search** (given a caption S, rank the images): with a uniform prior over images, Bayes' rule says
  ```
  p(I | S) = p(S | I) p(I) / p(S)  ∝  p(S | I)                 (p(S) is the same for every image)
  ```
  so **ranking images by p(S | I) is exactly right.**
- **Image annotation** (given an image I, rank the captions): ranking by p(S | I) is **biased**.
  - **The bias:** short, generic captions ("a man") have high probability for **every** image.
  - **The fix:** divide by the caption's overall probability p(S) = average over images of p(S | I'):
    ```
    score(S, I) = log p(S | I) − log [ (1/#images) Σ_{I'} p(S | I') ]          ("normalized", like a PMI)
    ```
  - **What it rewards:** captions that are **specifically** likely for this image.
  - This is our reading of the paper's "we normalized our scores similar to [21]".
- **Our demo:**
  - **raw annotation R@1 is 37.5%;** **normalized, 100%**;
  - **normalization can't change image search:** for a fixed caption it subtracts the same number from every image's score. Our test checks this.
- **The metrics:**
  - **R@K** = the % of queries whose correct answer is in the top K;
  - **median rank** = the median position of the first correct answer (lower is better).

---

## 3b. The automatic metrics

- **BLEU-n** (Paper 026, section 5.1): clipped n-gram precision with a brevity penalty. **Table 2 reports BLEU-1**, i.e. unigram precision only. That's lenient: a bag of the right words scores well in any order.
- **Human BLEU:** score each of the 5 human captions against the other 4 and average. The paper then adds back the average gain of having 5 references instead of 4.
- **CIDEr-D:**
  - represent each caption as a **TF-IDF vector** of its n-grams. An n-gram's weight is its count in the caption × log(#images / #images whose references contain it), so words common to *all* captions (like "a") weigh little;
  - average the cosine similarity with each reference over n = 1…4, with a Gaussian penalty on the length difference;
  - × 10 (and × 100 in Table 1).

  **It rewards saying what is distinctive about this image.**

---

## 4. Training details (Section 4.3.1)

**The main problem is overfitting:** the good datasets have fewer than 100,000 images.

**What helped:**
- a CNN **pretrained on ImageNet** ("helped quite a lot");
- dropout;
- ensembles of models (worth a few BLEU points).

**What didn't help:**
- initializing W_e from word2vec (Paper 019) gave no gain;
- updating the CNN weights "had a negative impact", so they stayed fixed.

**Other settings:**
- SGD with a fixed learning rate and no momentum;
- 512-d embeddings and LSTM;
- words seen at least 5 times;
- model selection by perplexity.

---

## 5. Results (Section 4.3)

### Table 1: MSCOCO development set
| | BLEU-4 | METEOR | CIDEr |
|---|---|---|---|
| NIC | **27.7** | 23.7 | **85.5** |
| Random | 4.6 | 9.0 | 5.1 |
| Nearest neighbour | 9.9 | 15.7 | 36.5 |
| Human | 21.7 | 25.2 | 85.4 |

### Table 2: BLEU-1
| | Pascal (transfer) | Flickr30k | Flickr8k | SBU |
|---|---|---|---|---|
| Previous best | 25 | 56 | 58 | 19 |
| **NIC** | **59** | **66** | **63** | **28** |
| Human | 69 | 68 | 70 | – |

- **NIC beats humans on BLEU-4 and CIDEr on MSCOCO**, but human raters still clearly prefer human captions (Figure 4: NIC averages 2.37–2.72 out of 4; ground truth averages 3.89). **"BLEU is not a perfect metric."**
- **Transfer and data size:**
  - training on Flickr30k (4× more data) gives +4 BLEU over training on Flickr8k;
  - transferring from MSCOCO costs about 10 BLEU because of domain mismatch;
  - an MSCOCO-trained model run on SBU drops from 28 to 16.
- **Diversity (Table 3):**
  - the best caption is a sentence from the training set **80%** of the time;
  - **about half** of the 15-best list is novel;
  - the BLEU agreement among the 15-best is 58, similar to humans.
- **Ranking (Tables 4–5):** NIC is a generative model, but it ranks by p(S|I) as well as models built for ranking. On Flickr8k it gets annotation R@1 20 and median rank 6, and image search R@1 19 and median rank 5.
- **Embeddings (Table 6):**
  - "horse" is close to "pony" and "donkey";
  - "car" is close to "van" and "cab";
  - the authors argue this helps the CNN learn related visual features.

---

## 6. Why it matters

- **It turned image captioning into a sequence problem** and set the template for "vision encoder + language decoder". This line leads from Show, Attend and Tell (attention over image regions, Paper 028's idea) to CLIP-style and multimodal LLMs.
- **It made the metric debate concrete:** when the model "beats humans" on BLEU but loses with human raters, that tells you something is wrong with the metric. CIDEr and METEOR were reported partly for this reason.

---

## 7. What our code found

**Scale note:**
- At your request, nothing heavy was run.
- `experiments.py` reproduces on Flickr8k (fixed GoogLeNet features):
  - E1: Table 2's BLEU-1, plus BLEU-4 and CIDEr-D, beam 20 vs greedy;
  - E2: Table 4's ranking;
  - E3: image once vs every step;
  - E4: diversity and novelty;
  - E5: Table 6's neighbours.

**The demo** uses a toy "image world" (24 kinds of object, by size, colour and shape, with 5 caption styles each). 4 kinds are **held out of training**. It trains in about a second.

- **It captions held-out combinations correctly,** for example "a small blue square" for an image it never saw. The beam's N-best list then contains **novel** sentences that never appeared in training, as in the paper's Table 3. The same is checked in a test.
- **Beam 5 vs greedy:** BLEU-4 is 96.6 vs 93.5. This matches the paper's "greedy costs about 2 BLEU".
- **The model "beats humans" on BLEU:**
  - BLEU-1 is 98 vs a human 81;
  - human BLEU-4 is **0**, because the 5 reference styles share no 4-gram.

  This is the paper's Section 4.3.6 warning in miniature: BLEU rewards matching the typical style, not being right.
- **Ranking by p(S|I):**
  - **raw** image annotation has only 37.5% R@1, because short, generic captions have high probability for every image;
  - **dividing by the caption's prior** (our reading of "normalized our scores similar to [21]") lifts this to 100%;
  - image search reaches 52% R@1, limited by captions that truly fit several images.
- **The learned W_e puts colours next to colours, shapes next to shapes, and "big" next to "small",** as in Table 6.

**Our notes on the text:**
- **Eq. (8) is m_t = o_t ⊙ c_t, with no tanh.** We implement it as printed. `cell_tanh=True` gives the standard form, which the authors' later open-source code (im2txt) uses.
- **The paper doesn't give a learning rate.** experiments.py uses im2txt's SGD lr 2.0 with clipping at 5.

---

## 8. Check yourself

1. Write Eqs. 10–13. When does the image enter the LSTM, and why only then?
2. Why doesn't the loss include p₀?
3. What did the authors find about fine-tuning the CNN and about word2vec initialization?
4. How does beam search differ from sampling? What did beam 1 cost?
5. NIC beats humans on BLEU-4 but loses with human raters. What does that say about BLEU?
6. How can a generative captioner do retrieval (Tables 4–5)? Why normalize the annotation scores?
