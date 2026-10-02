# GPT (Generative Pre-Training), explained simply

**Paper:** Alec Radford, Karthik Narasimhan, Tim Salimans & Ilya Sutskever, *Improving Language Understanding by Generative Pre-Training*, OpenAI, 2018.

**In one sentence:** first train a Transformer to **predict the next word** on thousands of books (no labels needed). Then, with very few changes, **fine-tune** it on each labelled task: classification, entailment, similarity, question answering. One model and one recipe beat custom-built models on 9 of 12 benchmarks.

---

## 1. The big idea

### 1.1 The problem: labels are scarce, text is everywhere
- Supervised NLP needs labelled examples, which are expensive.
- Unlabelled text (books, the web) is nearly free.
- **Question:** can a model learn from raw text something that helps *every* task?

Earlier answers:
- **Word vectors** (word2vec, GloVe): they transfer only word-level meaning, not how sentences fit together.
- **LSTM language models + fine-tuning** (ULMFiT, Dai & Le): they work, but LSTMs struggle with long-range structure.
- **ELMo:** uses a pre-trained LM's hidden states as *extra features* inside a new, task-specific architecture. Every task needs new design and new parameters.

### 1.2 GPT's recipe
1. **Generative pre-training:** a 12-layer Transformer decoder learns to predict the next token on BooksCorpus (7,000 unpublished books with long contiguous stories).
2. **Discriminative fine-tuning:** add a single linear layer W_y and train on the labelled task.
3. **Input transformations:** structured inputs (pairs of sentences, question + answers) are written as *one token sequence* with special delimiter tokens. The same network handles every task without architectural changes.

---

## 2. Stage 1: unsupervised pre-training (Section 3.1)

### 2.1 The objective (Eq. 1)

```
L₁(U) = Σ_i log P(u_i | u_{i−k}, …, u_{i−1}; Θ)
```

- Maximise the log-probability of each token given the previous k tokens (here k = 512).
- This is ordinary next-word prediction, the same "language model" as in papers 028–034.

**Worked example:** the sentence "the cat sat".
- If the model gives P(cat | the) = 0.2 and P(sat | the cat) = 0.5, these tokens contribute log 0.2 + log 0.5 = −1.609 − 0.693 = **−2.303** to L₁.
- Training pushes this up (closer to 0).

### 2.2 The network (Eq. 2)

```
h₀ = U W_e + W_p                       (token embedding + learned position embedding)
h_l = transformer_block(h_{l−1})       for l = 1 … n
P(u) = softmax(h_n W_eᵀ)               (output re-uses the token embedding matrix: "tied")
```

**Each block** is masked multi-head self-attention plus a position-wise feed-forward layer.
- **Masked** means token i can only look at tokens 1…i (paper 034's decoder, without the encoder).
- **Residual connections and layer norm** (post-LN, as in the original Transformer).

**Specifications (Section 4.1):**

| choice | value |
|---|---|
| layers | 12 |
| width | d = 768 |
| heads | 12 |
| FFN inner size | 3072 |
| activation | **GELU** |
| positions | **learned** (not sinusoidal) |
| context | 512 tokens |
| vocabulary | BPE with 40,000 merges |
| dropout | 0.1 (residual, embedding, attention) |
| init | N(0, 0.02), "sufficient because LayerNorm is used everywhere" |
| optimiser | Adam, max learning rate 2.5·10⁻⁴, linear warm-up over 2000 updates, cosine decay to 0 |
| weight decay | the decoupled ("modified L2") version, w = 0.01, on all non-bias, non-gain weights |
| training | 100 epochs, batches of 64 × 512 contiguous tokens |
| result | perplexity **18.4** on BooksCorpus |

**Parameter count, by hand:**
- **token embeddings:** 40,478 × 768 = 31.1M;
- **positions:** 512 × 768 = 0.4M;
- **per block:**
  - attention: 4 × 768² + biases = 2.36M;
  - FFN: 2 × 768 × 3072 + biases = 4.72M;
  - layer norms: about 3k;
  - total about **7.09M**;
- **12 blocks:** 85.0M;
- **total ≈ 116.5M**, the famous "117M". Our test computes it exactly.

---

## 3. Stage 2: supervised fine-tuning (Section 3.2)

Given a labelled example (x₁ … x_m, y):

```
P(y | x₁…x_m) = softmax(h_l^m W_y)                 (Eq. 3)   h_l^m = final hidden state at the LAST token
L₂(C) = Σ_(x,y) log P(y | x₁…x_m)                  (Eq. 4)
L₃(C) = L₂(C) + λ · L₁(C),  λ = 0.5                 (Eq. 5)
```

- **Why the last token?** Because of the causal mask, only the last position has seen the whole input.
- **New parameters:** just W_y and embeddings for the special tokens.
- **Auxiliary LM (Eq. 5):** keep predicting the next word *while* fine-tuning. This:
  - **regularises:** the network keeps its general language knowledge;
  - **speeds up convergence.**
- **Fine-tuning settings:** learning rate 6.25·10⁻⁵, batch 32, **3 epochs** usually enough, warm-up over 0.2% of training then linear decay, classifier dropout 0.1.

---

## 4. Input transformations (Section 3.3, Figure 1)

The pre-trained model only knows single contiguous texts. Instead of new architectures, **rewrite every input as one sequence**. ⟨s⟩ is a start token, $ a delimiter and ⟨e⟩ an extract token, all with new learned embeddings.

| task | sequence(s) | how the output is used |
|---|---|---|
| classification | ⟨s⟩ text ⟨e⟩ | linear on h at ⟨e⟩ |
| entailment | ⟨s⟩ premise $ hypothesis ⟨e⟩ | linear on h at ⟨e⟩ |
| similarity | ⟨s⟩ A $ B ⟨e⟩ **and** ⟨s⟩ B $ A ⟨e⟩ | the two ⟨e⟩ states are **added**, then linear (no natural order) |
| multiple choice / QA | ⟨s⟩ context $ answer_k ⟨e⟩ for each k | one score per answer, **softmax across answers** |

**Example (entailment):** "⟨s⟩ a man sleeps $ a person rests ⟨e⟩". Attention lets the hypothesis tokens look back at the premise.

---

## 5. Results (Section 4.2)

### Natural language inference (Table 2, accuracy)

| | MNLI-m | MNLI-mm | SNLI | SciTail | QNLI | RTE |
|---|---|---|---|---|---|---|
| previous best | 80.6 | 80.1 | 89.3 | 83.3 | 82.3 | **61.7** |
| **GPT** | **82.1** | **81.4** | **89.9** | **88.3** | **88.1** | 56.0 |

**On RTE** (only 2,490 examples) GPT lost to a multi-task biLSTM.

### QA and commonsense (Table 3)
| | Story Cloze | RACE |
|---|---|---|
| previous best | 77.6 | 53.3 |
| **GPT** | **86.5** (+8.9) | **59.0** (+5.7) |

### Similarity and classification (Table 4)
| | CoLA (mc) | SST-2 (acc) | MRPC (F1) | STS-B (pc) | QQP (F1) | GLUE |
|---|---|---|---|---|---|---|
| previous best | 35.0 | 93.2 | 86.0 | 81.0 | 66.1 | 68.9 |
| **GPT** | **45.4** | 91.3 | 82.3 | **82.0** | **70.3** | **72.8** |

The metrics are Matthews correlation (mc), accuracy, F1 and Pearson correlation (pc).

**Overall:** state of the art on **9 of 12** datasets, from STS-B (5.7k examples) to SNLI (550k).

---

## 6. Analysis (Section 5)

### 6.1 Transferring more layers helps (Figure 2, left)
- Copy only the embeddings, then 1, 2, …, 12 pre-trained layers, and fine-tune.
- **Every layer adds accuracy**, up to **+9%** on MultiNLI for full transfer.
- So each layer holds something useful, not just the embeddings.

### 6.2 Zero-shot behaviour (Figure 2, right)
**Hypothesis:** to get better at predicting text, the language model has to learn skills the tasks need. The authors tested it with **no fine-tuning at all**:

| task | zero-shot heuristic |
|---|---|
| **SST-2** | append "very", then compare P("positive") vs P("negative") as the next word |
| **CoLA** | the average token log-probability, thresholded |
| **RACE** | pick the answer with the highest average log-probability given the document and question |
| **Winograd (DPRD)** | substitute each candidate for the pronoun, and keep the one making the rest of the sentence most probable |

- These scores **rise steadily during pre-training**.
- The Transformer's curves are steadier than an LSTM's.
- Pre-training really does learn task-relevant skills.

### 6.3 Ablations (Table 5, average over 9 tasks)
| model | avg score |
|---|---|
| Transformer with auxiliary LM (full) | 74.7 |
| without pre-training | **59.9** (−14.8) |
| without auxiliary LM | 75.0 |
| LSTM with auxiliary LM | 69.1 (−5.6) |

- **Pre-training is the big win.**
- **The auxiliary LM** helps on large datasets (NLI, QQP) but not small ones, so the average is even slightly higher without it.
- **The Transformer beats the LSTM** on all but MRPC.

---

## 7. Why it works (the intuition)

1. **Language modelling is a universal teacher.** To predict the next word in a story you must track grammar, meaning, who did what, and sentiment. These are exactly the skills downstream tasks need.
2. **Transformers keep long-range information.** Attention connects distant tokens directly, so the skills learned span sentences and paragraphs. Books give long contiguous text to learn from.
3. **Change the input, not the model.** Writing every task as a sequence lets fine-tuning reuse *all* pre-trained layers instead of bolting on new architectures.
4. **Fine-tuning moves only a little.** The general model is close to good for each task, so a few epochs with a small learning rate suffice.

---

## 8. What our code found

All numbers come from `demo.py` (about 11 s) and `test_gpt.py` (8 tests, about 1 s).

**The toy world:** synthetic reviews like "the acting was brilliant . the plot was dull . …". 80% of a review's sentences share its sentiment. The unlabelled reviews end with "overall it was very good/bad .".

1. **Size:** our GPT class with the paper's settings has **116.5M** parameters, the "117M".
2. **The mechanics are exact:**
   - the output layer is tied to W_e;
   - the model is causal;
   - the LM loss equals next-token cross-entropy and ignores padding;
   - the four input transformations match Figure 1;
   - the classifier reads the ⟨e⟩ position;
   - L₃ = L₂ + λL₁.
3. **Zero-shot appears from pre-training alone:**
   - after 1000 pre-training steps on 5,000 unlabelled reviews, the "very good vs very bad" heuristic classifies **89.0%** of test reviews correctly, with no labels;
   - **honest note:** at step 300 it was still at chance (50.6%), so in our toy the skill appeared **abruptly**, not gradually as in the paper's Figure 2.
4. **Fine-tuning with only 40 labels** (3 seeds):

   | | mean accuracy |
   |---|---|
   | pre-trained, λ = 0.5 | **80.1%** |
   | pre-trained, λ = 0 | **82.1%** |
   | no pre-training | **73.1%** |

   - Pre-training helps.
   - The auxiliary LM made no clear difference on this tiny dataset, as the paper found for small datasets.
   - With 40 labels, fine-tuning did *not* beat the zero-shot score.
5. **Layer transfer:**

   | layers transferred | mean accuracy |
   |---|---|
   | embeddings only | 50.2% |
   | 1 layer | 57.4% |
   | 2 layers | **75.3%** |

   - More layers, more accuracy, as in Figure 2 (left).
   - **Odd detail:** with embeddings only, every seed predicted a single class, worse than training from scratch. We did not investigate why.

**Not run (too heavy):** pre-training on 20 Gutenberg books with BPE, then
- E2: Table 5 on SST-2 / RTE / MRPC / COPA;
- E3: layers transferred;
- E4: zero-shot heuristics over checkpoints;
- E5: the LSTM comparison.

---

## 9. Check yourself

1. Why does the classifier read the hidden state at the *last* token?
   <details><summary>Answer</summary>The attention is causal, so only the last position has attended to the whole input sequence.</details>
2. Write the input for a similarity task between "he left" and "he departed".
   <details><summary>Answer</summary>Two sequences: ⟨s⟩ he left $ he departed ⟨e⟩ and ⟨s⟩ he departed $ he left ⟨e⟩. Their final ⟨e⟩ states are added before W_y.</details>
3. With λ = 0.5, a classification loss of 0.4 nats and an LM loss of 3.0 nats per token, what is the total fine-tuning loss?
   <details><summary>Answer</summary>0.4 + 0.5 × 3.0 = 1.9.</details>
4. How many parameters does the token embedding take with 40,478 tokens and d = 768? Why doesn't the output layer add another copy?
   <details><summary>Answer</summary>31.1M. The output layer reuses W_e (it computes h W_eᵀ), so no new matrix is needed.</details>
5. How could a pure language model do sentiment classification with no fine-tuning?
   <details><summary>Answer</summary>Append "very" and compare the probabilities it gives to "positive" vs "negative" (or "good" vs "bad") as the next word. The LM has learned which continuation fits the text.</details>
6. Which ablation hurt the most, and by how much?
   <details><summary>Answer</summary>Removing pre-training: the average score fell from 74.7 to 59.9 (−14.8).</details>
