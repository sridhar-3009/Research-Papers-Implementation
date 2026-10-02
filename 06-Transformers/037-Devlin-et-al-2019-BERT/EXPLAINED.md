# Devlin, Chang, Lee & Toutanova (2019), explained from scratch

**Paper:** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
**Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (Google AI Language)
**Published at:** NAACL 2019 (arXiv:1810.04805)

Read Paper 034 (the Transformer) first; Paper 019 (word2vec) helps for the idea of learning from unlabelled text. This guide:
- explains **why** a language model normally can't look both ways, and how masking fixes that;
- works through the input format and the 80/10/10 masking rule with numbers;
- counts BERT-base's 110M parameters;
- shows how one pre-trained network is fine-tuned for classification, question answering and multiple choice.

---

## 0. The whole idea in one line

> **Pre-train one Transformer encoder on huge amounts of plain text by hiding 15% of the words and asking it to fill them in (using the words on BOTH sides), plus guessing whether two text spans are consecutive. Then add one tiny output layer and fine-tune the whole thing on each task. One recipe set new records on 11 NLP tasks.**

---

## 1. Background: two ways to reuse pre-trained language knowledge (Sections 1–2)

- **Feature-based (ELMo):** run a pre-trained model, and feed its vectors as **fixed features** into a separate task model.
- **Fine-tuning (OpenAI GPT):** pre-train a whole network as a language model, then continue training **all** its weights on the task with a small output layer added.
- **The limitation both share:** standard language models are **unidirectional**. GPT predicts each word from the words to its **left** only. ELMo trains separate left-to-right and right-to-left LSTMs and just concatenates them, which is "shallow" bidirectionality.
- **Why that hurts:** in "the bank of the river", deciding what "bank" means requires the words **after** it. For question answering, every token of the passage should see the question and the rest of the passage.

### 1.1 Why a normal LM can't simply look both ways
- **A language model predicts word i from its context.** If the context included word i itself (directly or through other layers), the model could just copy it.
- **In a deep bidirectional Transformer,** every position attends to every other position in every layer. Word i's information reaches all other positions after one layer, so by layer 2 "each word could indirectly see itself". Predicting the next word would be trivial.
- **So left-to-right LMs must mask out the future (causal attention, Paper 034), and lose the right context.**

---

## 2. The model and its input (Section 3)

### 2.1 Architecture
- A **Transformer encoder** (Paper 034's left half), with **bidirectional** self-attention:

  | model | layers L | hidden H | heads A | parameters |
  |---|---|---|---|---|
  | BERT-base | 12 | 768 | 12 | 110M (same size as GPT, on purpose) |
  | BERT-large | 24 | 1024 | 16 | 340M |

- **Feed-forward size 4H;** GELU activation; learned position embeddings up to 512 tokens.

### 2.2 Counting BERT-base's parameters (vocabulary 30,522)
```
embeddings   token 30,522·768 + position 512·768 + segment 2·768 + LayerNorm      ≈ 23.8M
each layer   attention 4·768² + 4·768  +  FFN 2·768·3072 + 3072 + 768  +  2 LayerNorms ≈ 7.09M
12 layers                                                                          ≈ 85.1M
pooler       768·768 + 768                                                         ≈ 0.6M
MLM head     dense 768² + LayerNorm + output bias (output weights TIED to the token embeddings)
total                                                                              ≈ 110.1M ✔
```

### 2.3 One input format for every task (Figure 2)
```
[CLS] my dog is cute [SEP] he likes play ##ing [SEP]
segment:  A  A  A  A  A  A    B  B     B    B     B
position: 0  1  2  3  4  5    6  7     8    9    10
```
- **Every input embedding = token embedding + segment embedding (A or B) + position embedding.**
- **[CLS]** starts every sequence. Its final hidden vector **C** summarizes the whole input for classification.
- **[SEP]** separates sentence A from sentence B. A single sentence is just A alone.
- **A "sentence" means any span of text,** not necessarily a linguistic sentence.

### 2.4 WordPiece tokens (30,000 vocabulary)
- **Words are split into frequent pieces:** "playing" → "play" + "##ing" ("##" marks a piece that continues a word).
- **Tokenizing uses greedy longest-match-first:** take the longest piece in the vocabulary that starts the word, then repeat on the rest.
- **The benefit:** no unknown words (any word is spelled from pieces), with a modest vocabulary.
- **Our demo:** "jumped" → "jump" + "##e" + "##d", since "##ed" wasn't frequent enough in our tiny corpus to become a piece.

---

## 3. Pre-training task 1: the masked language model (Section 3.1)

### 3.1 The idea
- **Hide some words and predict them from both sides** (the "Cloze" task, Taylor 1953).
- **Mask 15% of the WordPiece tokens at random,** and predict only those, with cross-entropy over the vocabulary.
- **Unlike a denoising autoencoder,** BERT doesn't reconstruct the whole input, only the chosen positions.

### 3.2 The 80/10/10 rule, and why
- **The problem:** [MASK] never appears during fine-tuning. A model trained only on [MASK] inputs learns representations tuned to an input it will never see again.
- **The fix:** for each chosen position:
  - **80%:** replace it with [MASK];
  - **10%:** replace it with a **random** token;
  - **10%:** leave it **unchanged**.
- **The model must predict the original in all three cases.** Since any token might be a corrupted or kept target, it has to keep a good contextual representation of **every** token.
- **The arithmetic:** 15% × 80% = 12% of tokens are [MASK], 1.5% random and 1.5% unchanged-but-predicted.
- **Our measurement** over 102,400 tokens: chosen 15.0%; of those, [MASK] 80.0%, unchanged 9.7%, random 10.3%.

### 3.3 The cost
- **Only 15% of tokens give a training signal per pass** (a left-to-right LM learns from 100%).
- **The paper's Figure 5:** MLM converges a little more slowly at first, then beats the LTR model by a wide margin on downstream accuracy.

### 3.4 Why bidirectionality matters, shown on a toy (our demo)
- **The toy:** a sequence of (marker, letter) pairs, where each **marker is fixed by the letter to its right**.
- **The result:**
  - **masked LM (bidirectional):** 100% of markers recovered;
  - **left-to-right LM:** 9.5%, i.e. chance (10%).
- **The point:** a left-to-right model *cannot* use the right context, however long it trains.

---

## 4. Pre-training task 2: next sentence prediction, NSP (Section 3.1)

- **Many tasks are about the relation between two texts** (question/answer, premise/hypothesis).
- **The NSP task:**
  - pick sentence A from a document;
  - **50%** of the time B is the real next span (IsNext), and 50% of the time a random span from another document (NotNext);
  - classify from C, the final [CLS] vector.
- **BERT reaches 97–98% on NSP.**
- **Ablation (Table 5): removing NSP hurt QNLI, MNLI and SQuAD.**
  - Later work (RoBERTa, 2019) found NSP unnecessary when trained on longer contiguous text, so it has been debated since.
- **Our toy:** 77% after 500 small updates.

### 4.1 The total pre-training loss
```
loss = mean masked-LM cross-entropy  +  mean NSP cross-entropy
```

### 4.2 Data and schedule (Appendix A.2)
- **Data:** BooksCorpus (800M words) + English Wikipedia (2,500M words), **document-level**, so that long contiguous spans exist.
- **Batches:** 256 sequences × 512 tokens = 128,000 tokens per batch, for **1M steps** (about 40 epochs).
- **Optimizer:** Adam, lr 1e-4, β = (0.9, 0.999), weight decay 0.01, **10,000-step warm-up then linear decay**; dropout 0.1; GELU.
- **Length schedule:** 90% of the steps use length 128 (attention is quadratic, so it's cheaper). The last 10% use 512, to learn the later position embeddings.
- **Hardware:** BERT-base took 4 days on 16 TPU chips.

---

## 5. Fine-tuning: one model, many tasks (Sections 3.2, 4)

- **Start from the pre-trained weights, add one small layer, and train everything end to end.**
- **Cost:** about an hour on one TPU for most tasks.
- **Hyperparameters:** batch 16 or 32, lr ∈ {5e-5, 3e-5, 2e-5}, 2–4 epochs.

| Task type | Input | New parameters | Output |
|---|---|---|---|
| **sentence (pair) classification** (GLUE) | [CLS] A [SEP] (B [SEP]) | W (K × H) | log softmax(C Wᵀ) |
| **span QA** (SQuAD) | [CLS] question [SEP] passage [SEP] | start vector S, end vector E (H each) | P(start = i) ∝ e^{S·T_i}, P(end = j) ∝ e^{E·T_j}; best span max_{j ≥ i} S·T_i + E·T_j |
| **SQuAD 2.0 (may have no answer)** | same | same | "no answer" = the span at [CLS], score s_null = S·C + E·C; answer only if the best span > s_null + τ |
| **multiple choice** (SWAG) | 4 sequences [CLS] context [SEP] ending [SEP] | a vector v | softmax over v·C of the 4 choices |
| **token tagging** (NER) | the sentence | a per-token classifier | labels from each T_i |

**Why one network handles pairs well:** concatenating A and B and running self-attention over both means every token of A attends to every token of B in every layer. This is **bidirectional cross-attention for free**, which earlier models built as a separate component.

---

## 6. Results (Section 4)

### Table 1: GLUE test (single model)
| System | MNLI | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Avg |
|---|---|---|---|---|---|---|---|---|---|
| previous best | 80.6 | 66.1 | 82.3 | 93.2 | 35.0 | 81.0 | 86.0 | 61.7 | 74.0 |
| OpenAI GPT | 82.1 | 70.3 | 87.4 | 91.3 | 45.4 | 80.0 | 82.3 | 56.0 | 75.1 |
| **BERT-base** | 84.6 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 | 79.6 |
| **BERT-large** | **86.7** | **72.1** | **92.7** | **94.9** | **60.5** | **86.5** | **89.3** | **70.1** | **82.1** |

- **BERT-base vs GPT:** the same size and almost the same architecture. The difference is mostly **bidirectionality** (+4.5 points on average).
- **BERT-large wins everywhere,** even on tiny datasets (RTE has 2,500 examples). BERT-large fine-tuning was sometimes unstable on small sets, so they took the best of several restarts.

### SQuAD and SWAG
| Task | Metric | BERT | Previous best |
|---|---|---|---|
| SQuAD 1.1 | test F1, single model | **91.8** (+TriviaQA) | 91.7 (an ensemble) |
| SQuAD 1.1 | ensemble | **93.2** | |
| SQuAD 2.0 | test F1 | **83.1** | 78.0 (+5.1) |
| SWAG | test accuracy | **86.3** | GPT 78.0; humans 88.0 |

### Table 5: ablation of the pre-training tasks (dev)
| Model | MNLI | QNLI | MRPC | SST-2 | SQuAD F1 |
|---|---|---|---|---|---|
| BERT-base | 84.4 | 88.4 | 86.7 | 92.7 | 88.5 |
| no NSP | 83.9 | 84.9 | 86.5 | 92.6 | 87.9 |
| LTR & no NSP | 82.1 | 84.3 | 77.5 | 92.1 | 77.8 |
| + BiLSTM on top | 82.1 | 84.1 | 75.7 | 91.6 | 84.9 |

**Bidirectionality is the big win** (SQuAD 87.9 → 77.8 without it). A BiLSTM added on top of an LTR model helps SQuAD but not GLUE.

### Table 6: bigger is better, even for tiny tasks
| (L, H, A) | 3/768/12 | 6/768/12 | 12/768/12 | 24/1024/16 |
|---|---|---|---|---|
| MLM perplexity | 5.84 | 4.68 | 3.99 | 3.23 |
| MNLI | 77.9 | 81.9 | 84.4 | 86.6 |
| MRPC (3.6k examples) | 79.8 | 84.8 | 86.7 | 87.8 |

"The first work to demonstrate convincingly that scaling to extreme model sizes also leads to large improvements on very small scale tasks, provided that the model has been sufficiently pre-trained."

### Feature-based use (Section 5.3, Table 7)
- **The setup:** NER (CoNLL-2003) with frozen BERT features.
- **The result:** concatenating the **top four layers** reaches 96.1 dev F1, only 0.3 behind full fine-tuning (96.4).

### Masking mix (Appendix C.2)
- **What was compared:** fine-tuning is fairly robust to the mix of [MASK], random and unchanged replacements.
- **Where it matters:** the feature-based approach is hurt most by always using [MASK] (100%/0/0), which is exactly the mismatch problem.

---

## 7. Why it matters

- **"Pre-train once, fine-tune everywhere"** became the standard NLP workflow (2018–2020): RoBERTa, ALBERT, ELECTRA, DeBERTa, T5.
- **Encoder-only BERT-style models** still power search ranking, embeddings, classification and retrieval (Paper 070's DPR uses two BERT encoders).
- **The generative side** went the other way: GPT-style decoder-only models (Papers 048–050) won generation and in-context learning, and eventually most tasks.

---

## 8. What our code found

**Scale note:**
- At your request, nothing heavy was run on this laptop.
- `experiments.py` pre-trains a **mini-BERT** on public-domain Gutenberg books (a stand-in for BooksCorpus) with our own WordPiece vocabulary, then fine-tunes on GLUE SST-2 and RTE:
  - E1: pre-trained vs from scratch;
  - E2: Table 5's ablation;
  - E3: Table 6's sizes;
  - E4: Appendix C.2's masking mixes;
  - E5: feature-based vs fine-tuning.

**Checked (tests and demo, ~5 seconds):**
- **Sizes:** BERT-base **110.1M**, BERT-large **336.2M** (paper: 110M / 340M).
- **WordPiece** is greedy longest-match-first ("unaffable" → un + ##aff + ##able), with [UNK] when no split exists.
- **Packing** gives exactly Figure 2's tokens and segments; truncation removes from the longer sentence.
- **Masking:** 15.0% chosen, then 80.0 / 9.7 / 10.3% (mask / unchanged / random); special tokens are never chosen.
- **NSP pairs:** 50/50, with the "next" really being next and "random" really from another document.
- **Bidirectional vs left-to-right:** changing a later token changes an earlier token's representation only in the bidirectional model. Padding never changes real tokens' outputs.
- **The heads:**
  - the loss is MLM + NSP;
  - the MLM output is tied to the token embeddings;
  - the classification, span (including SQuAD 2.0's no-answer rule), multiple-choice and four-layer feature heads have the right shapes and logic.
- **Schedule:** warm-up + linear decay; weight decay is not applied to biases or LayerNorm weights.
- **Toy marker task:** masked LM **100%** vs left-to-right **9.5%** (chance 10%).

**An honest negative result:**
- **The test:** on a tiny synthetic task (compare two letters, 64 labelled examples), 400 steps of toy MLM pre-training did **not** help fine-tuning (79% vs 83% from scratch).
- **Why:** pre-training pays off when the unlabelled data teaches structure the task needs, at a scale our toy doesn't have. E1 tests the real claim on SST-2/RTE with a book corpus.

---

## 9. Check yourself

1. Why can't a deep bidirectional Transformer be trained as an ordinary language model?
2. Write BERT's input for the pair ("the cat sat", "it was tired") with segments and positions.
3. What fractions of all tokens become [MASK], random and unchanged-but-predicted? Why not always [MASK]?
4. What is NSP? What did removing it do in Table 5?
5. Count the parameters of one BERT-base layer.
6. How does BERT score an answer span in SQuAD? How does SQuAD 2.0 say "no answer"?
7. BERT-base and GPT have the same size. What makes BERT better on GLUE?
8. What is the difference between fine-tuning and the feature-based approach? How close did they come on NER?
