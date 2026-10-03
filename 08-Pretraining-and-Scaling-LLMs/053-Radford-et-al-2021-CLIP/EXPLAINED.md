# CLIP: Learning Transferable Visual Models From Natural Language Supervision, explained simply

**Paper:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger & Ilya Sutskever (OpenAI), *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021.

**In one sentence:** train an image encoder and a text encoder together so that a picture and its own caption land close to each other in a shared space. After training on 400 million (image, text) pairs from the internet, you can classify images into *any* set of categories, without a single labelled training example, by writing the category names as sentences ("a photo of a dog.") and asking which sentence is closest to the picture.

---

## 1. The big idea

### 1.1 The problem with labels
- **The old recipe:** classic vision models (ResNet, ViT, paper 038) learn from a fixed list of labels, e.g. ImageNet's 1000 classes.
- **The limits:**
  - **Cost:** labels are expensive; ImageNet needed 25,000 workers for 14 million images.
  - **Closed vocabulary:** the model only knows those 1000 words. To recognise "a stop sign covered in snow", you must collect new labels and train a new classifier.
- **Meanwhile, the internet has billions of images that already come with text:** alt-text, titles, captions. The text is noisy, but it is free and far richer than one label out of 1000.

### 1.2 What NLP had already shown
- GPT (048), GPT-2 (049) and GPT-3 (050) learned from raw web text and could then do new tasks **zero-shot**: you describe the task, with no fine-tuning.
- **CLIP's question:** can vision get the same thing by learning from web text **about** images?

### 1.3 Earlier attempts were weak
- **Visual N-Grams (Li et al. 2017):** learned from image text and reached only **11.5%** zero-shot on ImageNet.
- **VirTex and ConVIRT** (2020) learned good features from captions, but at small scale.
- **CLIP** = a simplified ConVIRT, trained at scale: **400M pairs**, **76.2%** zero-shot on ImageNet. That matches the original ResNet-50 without using any of its 1.28 million labelled training images.

---

## 2. The data: WIT (Section 2.2)
- **400 million** (image, text) pairs collected from public internet sources.
- **Coverage:** they searched for pairs whose text contains one of **500,000 queries**:
  - every word appearing at least 100 times in English Wikipedia;
  - common bigrams;
  - Wikipedia article titles;
  - WordNet synsets.
- **Balance:** at most 20,000 pairs per query.
- **Size:** about as many total words as GPT-2's WebText.

---

## 3. Which training objective? (Section 2.3, Figure 2)
They tried three ways to learn from (image, caption) pairs and compared how fast each one's ImageNet zero-shot accuracy improves per training image:

| Objective | What the image model must do | Efficiency |
|---|---|---|
| Transformer language model | generate the **exact** caption, word by word | slowest |
| Bag-of-words prediction | predict **which words** appear (order ignored) | **3×** faster than the LM |
| Bag-of-words **contrastive** (CLIP) | pick **which caption** in the batch belongs to this image | another **4×** faster |

**Why predicting is wasteful:**
- A photo of a dog could have been captioned "my best friend!", "Rex at the beach, 2016" or "golden retriever puppy".
- Predicting the **exact** words is extremely hard, and mostly about things you can't see in the image.
- The contrastive task asks something easier and more useful: *which of these N captions goes with this image?* To answer, the model only needs to know what the image is about, not how someone phrased it.

---

## 4. The model and the loss (Section 2.4–2.5, Figure 3)

### 4.1 Two encoders and a shared space
- **Image encoder**, one of:
  - a ResNet with tweaks: its final average pooling is replaced by an **attention pooling** layer, whose query is the average-pooled feature;
  - a Vision Transformer (paper 038) with an extra LayerNorm before the Transformer.
- **Text encoder:**
  - a GPT-2-style Transformer (paper 049): 63M parameters, 12 layers, width 512, 8 heads;
  - lower-cased BPE with 49,152 tokens, at most 76 tokens;
  - the text is wrapped in [SOS] … [EOS], and the top layer's activation **at [EOS]** is the text's feature;
  - causal (masked) attention, kept so a language-model objective could be added later.
- **Projections:** each feature is mapped into a shared d-dimensional space by a **linear** projection, with no bias and no nonlinear MLP. (They found that nonlinear projection heads made no difference here.)
- **Normalisation:** both embeddings are L2-normalised to length 1.

### 4.2 The symmetric contrastive loss
Take a batch of N (image, text) pairs.
- Iᵢ = normalised embedding of image i; Tⱼ = normalised embedding of text j.
- **Similarity:** cosine similarity Iᵢ·Tⱼ, a number between −1 and 1.
- **Logits:** Sᵢⱼ = (Iᵢ·Tⱼ) × s, where s = exp(t) is a learned scale.
- **The task:** the **diagonal** pairs (i, i) are the real ones. The other N² − N pairs are mismatches, the **negatives**.

**Loss:**
```
loss_image = cross_entropy(rows of S,    target = the diagonal)   # each image picks its caption among N
loss_text  = cross_entropy(columns of S, target = the diagonal)   # each caption picks its image among N
loss       = (loss_image + loss_text) / 2
```

The paper's pseudocode (Figure 3), which our `clip_loss` follows line for line:
```
logits = np.dot(I_e, T_e.T) * np.exp(t)
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss   = (loss_i + loss_t)/2
```

**Worked example (N = 2):** the cosine similarities are
```
          text 1  text 2
image 1    0.9     0.1
image 2    0.2     0.8
```
**With scale s = 1** (logits = cosines), each loss term is the negative log of the correct entry's softmax probability:
- Row 1: softmax(0.9, 0.1) gives 1/(1+e^−0.8) = **0.690** for the correct pair, so the loss is −ln 0.690 = 0.371.
- Row 2: softmax(0.2, 0.8) gives 1/(1+e^−0.6) = 0.646, so 0.437.
- Column 1: (0.9, 0.2) gives 1/(1+e^−0.7) = 0.668, so 0.403.
- Column 2: (0.1, 0.8) gives 0.668 again, so 0.403.
- **Total:** ((0.371 + 0.437)/2 + (0.403 + 0.403)/2)/2 = **0.404**.

**With scale s = 10** (logits 9, 1 / 2, 8):
- Row 1: 1/(1+e^−8) = 0.99966, so 0.0003.
- Row 2: 1/(1+e^−6) = 0.9975, so 0.0025.
- Each column: 1/(1+e^−7), so 0.0009.
- **Total:** ≈ **0.0012**.

### 4.3 Why the temperature matters
- **Cosines are trapped in [−1, 1].** Without a scale, the best possible gap between the correct logit and a wrong one is 2.
- For N = 2, that floor is a loss of −ln(1/(1+e^−2)) = **0.127**. With N = 32,768 it would be far worse: the correct pair could never stand out against 32,767 rivals.
- **The scale s = 1/τ sharpens the softmax.** τ is the "temperature".
- **CLIP learns it:**
  - it is stored as t = log s, so it stays positive and changes multiplicatively;
  - it starts at s = 1/0.07 ≈ **14.3**;
  - it is **clipped at 100** for training stability. The trained models reach that ceiling.
- This removes a hyperparameter they would otherwise have had to tune.

### 4.4 Why the batch is huge
- Each image is compared with **N − 1 negatives**. More negatives make a harder, more informative task. Telling a dog from 32,767 other captions forces finer distinctions than telling it from 7.
- CLIP used batch **32,768**.

**The rest of the training recipe:**
- 32 epochs over WIT;
- Adam with decoupled weight decay;
- cosine learning-rate schedule;
- mixed precision.

**Compute:**
- the biggest ResNet (RN50x64) took **18 days on 592 V100 GPUs**;
- the biggest ViT (ViT-L/14) took **12 days on 256 V100 GPUs**;
- the best model is ViT-L/14 fine-tuned one extra epoch at 336-pixel resolution, called **ViT-L/14@336px**.

---

## 5. Zero-shot classification (Section 3.1)

### 5.1 The text encoder writes a classifier
To classify an image into K classes:
1. For every class name, build a sentence such as "a photo of a **dog**.", embed it with the text encoder, and normalise. This gives K vectors W₁ … W_K.
2. Embed the image: I.
3. **Score:** class k gets s × (I·W_k). The prediction is the largest score; a softmax over the scores gives probabilities.

**This is exactly a linear classifier** (no bias, normalised weights, a temperature), but its weights were **generated from text** instead of learned from labelled examples.
- The paper calls the text encoder a **hypernetwork**: a network that outputs another network's weights.
- **New classes cost nothing:** just type their names.

### 5.2 Prompt engineering
**The bare class name is a bad query:**
1. **Distribution mismatch:** in training, text is rarely a single word, so it is usually a sentence about the image. "A photo of a {label}." alone gives **+1.3%** on ImageNet.
2. **Polysemy:** "crane" could be a bird or a machine, and "boxer" a dog breed or an athlete. Context helps: "a photo of a boxer, **a type of pet**."
3. **Task hints:**
   - "a satellite photo of a {}";
   - "a photo of a {}, a type of food";
   - "a photo of the number {}" (for digits, as in our demo).

### 5.3 Prompt ensembling
- **The idea:** write many templates ("a photo of a big {}", "a blurry photo of a {}", "art of the {}", … **80** for ImageNet).
- **How:** embed each template for each class and **average the normalised embeddings**, then renormalise.
- **Cost:** because the averaging happens in embedding space, the ensemble is **still a single vector per class**. Classification costs exactly the same as with one prompt.
- **Gain:** ensembling adds **+3.5%** on ImageNet; together with prompt engineering, almost **+5%**.

**Tiny example:** two templates give the "dog" embeddings (0.8, 0.6) and (0.6, 0.8).
- Their mean is (0.7, 0.7), with length 0.99.
- Renormalised, it is (0.707, 0.707).
- The two templates' quirks partly cancel, leaving the direction they agree on.

### 5.4 Headline zero-shot numbers (Table 1)
| | aYahoo | ImageNet | SUN |
|---|---|---|---|
| Visual N-Grams | 72.4 | 11.5 | 23.0 |
| **CLIP** | **98.4** | **76.2** | **58.5** |

Across a **27-dataset suite** (Figure 5):
- **Overall:** zero-shot CLIP beats a **fully supervised** logistic regression on ResNet-50 features on **16 of the 27** datasets, including ImageNet.
- **Fine-grained tasks:**
  - **strong** on Stanford Cars and Food101, by over 20%;
  - **weak**, by over 10%, on Flowers102 and FGVC Aircraft, and on specialised or abstract tasks: satellite images, tumour detection, counting objects, MNIST (88%; a linear classifier on raw pixels beats it).

---

## 6. Representation learning: linear probes (Section 3.2)
- **Linear probe:** freeze the image encoder, then fit a logistic regression on its features using labelled data.
- **Results:**
  - CLIP's features beat the best ImageNet models' features on most of the paper's 27 datasets.
  - **Zero-shot CLIP ≈ a 4-shot linear probe on the same features** (Figure 6). Four labelled examples per class are needed before the probe catches up with "just typing the names".
  - On average, zero-shot CLIP matches a **16-shot** linear classifier on BiT-M features (a ResNet-152x2 trained on ImageNet-21K).
- **Why few-shot can be worse than zero-shot:**
  - **The prior:** zero-shot uses language to tell the model what the concept is.
  - **One example is ambiguous:** is it about the object, the background, or the colour? The probe must guess from a handful of pixels' worth of evidence.

---

## 7. Robustness to distribution shift (Section 3.3)

### 7.1 The puzzle
- **ImageNet models are fragile:** they are much worse on new test sets of the same classes:
  - ImageNetV2, re-collected the same way;
  - sketches;
  - renditions (cartoons, paintings);
  - ObjectNet (odd viewpoints);
  - ImageNet-A (adversarially filtered natural images).
- **Possible reason:** they exploit patterns that only hold on ImageNet (backgrounds, textures, photographer habits).

### 7.2 Effective robustness
- **The trend:** across many ImageNet-trained models, accuracy under shift is predictable from ImageNet accuracy. It is roughly a straight line after a **logit transform**, logit(p) = ln(p/(1−p)).
- **Effective robustness:** how far a model sits **above** that line.
- **Relative robustness:** how much it improves raw accuracy under shift.

### 7.3 CLIP's numbers (Figure 13)
Zero-shot CLIP (ViT-L/14@336px) vs a ResNet-101 with the **same** 76.2% ImageNet accuracy:

| Dataset | ResNet-101 | Zero-shot CLIP | Δ |
|---|---|---|---|
| ImageNet | 76.2 | 76.2 | 0 |
| ImageNetV2 | 64.3 | 70.1 | +5.8 |
| ImageNet-R (renditions) | 37.7 | 88.9 | +51.2 |
| ObjectNet | 32.6 | 72.3 | +39.7 |
| ImageNet Sketch | 25.2 | 60.2 | +35.0 |
| ImageNet-A | 2.7 | 77.1 | +74.4 |

- **The paper's summary:** zero-shot CLIP **reduces the gap between ImageNet accuracy and shifted accuracy by up to 75%**.
- **Over these five shifts:**
  - the ResNet loses 43.7 points on average;
  - zero-shot CLIP loses 2.5.

### 7.4 Adapting to ImageNet hurts robustness (Figure 14)
- **Adaptation:** fit a logistic regression on CLIP features using ImageNet's labels.
- **ImageNet:** it improves by **9.2% to 85.4%**.
- **Under shift:** **average** accuracy slightly **drops**. The gains are ImageNet-specific.
- **This supports the intuition:** a zero-shot model *cannot* exploit patterns that only hold in one dataset, because it never trained on that dataset.
- **The authors' caveat:** this does not prove that supervised training *causes* the gap. CLIP's diverse data might be the main reason.

---

## 8. Limitations the paper admits (Sections 5–7)
- **Not state of the art everywhere:**
  - zero-shot CLIP is usually competitive with a linear probe on ResNet-50, far below the best task-specific models;
  - they estimate about **1000×** more compute would be needed to reach the state of the art zero-shot.
- **Weak at abstract, systematic tasks:**
  - counting;
  - distances;
  - fine distinctions such as car models or aircraft variants;
  - truly out-of-distribution data: handwritten MNIST at 88%.
- **Can't generate captions.** It can only choose between texts you provide.
- **Biased and potentially harmful:** web data carries social biases.
  - They show misclassification of people into crime-related and non-human categories.
  - The rates depend heavily on which class names you offer (Section 7).
- **Not truly zero-shot in their own research process:** they looked at the 27 validation datasets many times while developing the model.

---

## 9. Why it works (the intuition)
- **Natural language is a huge label space.** Instead of 1000 categories, every caption is its own category. The model must learn whatever visual distinctions people *write about*: objects, styles, scenes, text in images, actions, emotions.
- **Contrastive learning only asks what matters.** "Which caption is yours?" can be answered from the image's content alone; the exact phrasing doesn't matter.
- **Scale and diversity.** 400M pairs from 500K queries cover far more than any labelled dataset. Diversity is also the likely source of the robustness: no single dataset's quirks dominate.
- **A shared space makes text into classifiers.** Because images and texts are compared by a dot product, any text becomes a classifier weight vector. That is what turns representation learning into zero-shot transfer.

---

## 10. What our code found
Everything below runs in a few seconds on a laptop CPU. It is a cartoon of CLIP, not a reproduction.

**Exact checks (tests):**
- **Logits:** they are exactly s × cosine; s starts at 1/0.07 and is clipped at 100.
- **The loss** equals the paper's pseudocode:
  - with all-zero logits it is exactly **log N**, the chance level;
  - for a perfect diagonal it is ≈ 0.
- **The prompt ensemble** is the renormalised mean of normalised embeddings.
- **The causal text Transformer:** changing a token *after* [EOS] does not change the [EOS] feature.
- **Effective robustness:** the logit-linear fit is checked on a synthetic trend.
- **Training:** a tiny model learns to match 10 image prototypes to their captions, taking the loss from 2.3 to under 0.3.

**The demo** (MNIST painted red, green or blue; captions such as "a red handwritten seven", "seven , red"; 13K-parameter model; 800 steps of batch 128, 2.4 s):

| Query | Zero-shot accuracy |
|---|---|
| digits, bare name "seven" | 87.9% |
| digits, "a photo of the digit seven" | 85.4% |
| digits, ensemble of 15 templates | **88.1%** |
| colours, "a photo of something red" | 100% |

- **No label was ever used for training.** The classifier was written by the text encoder.
- **Ensembling beats the single prompt** (+2.7), as in the paper.
- **Honest difference: the bare name does as well as the ensemble here.** Our text encoder is a bag of words (it averages word vectors), so extra words like "photo" only dilute the digit word. CLIP's Transformer reads whole sentences, so a single word is out-of-distribution for it.
- **Held-out combinations** (red 7s, green 3s and blue 5s never appeared in any caption): **54.7%** vs 91.9% on combinations seen in training.
  - That is well above chance (10%), so some composition happens.
  - It is much worse than for seen pairs: our tiny model partly learned "red 7" as its own thing.
- **Contrastive vs bag-of-words prediction** (same steps, same image encoder): **88.1% vs 48.6%**.
  - **Caveat:** this matches the paper's direction, but it is not a controlled measurement of the 4× efficiency claim.
  - The predictor also spends capacity predicting colours and filler words, and its zero-shot readout (the most likely digit word) differs from CLIP's.
  - `experiments.py` E1 measures it properly, with learning curves.
- **Robustness:** we only print the paper's numbers. A real test needs real distribution shifts; `experiments.py` E4 uses CIFAR corruptions.

**`experiments.py` (written, not run on this laptop)** does five things on CIFAR-10 or CIFAR-100 with noisy synthetic captions (synonyms, distractor words, colour words):
- **E1:** the captioner, bag-of-words prediction, bag-of-words contrastive, and CLIP, as learning curves;
- **E2:** bare name vs prompt vs ensemble;
- **E3:** zero-shot vs k-shot linear probes;
- **E4:** supervised vs linear-probe vs zero-shot under blur, noise, greyscale and sketch shifts;
- **E5:** fixed vs learned temperature, nonlinear projection, and batch 64 vs 512.

---

## 11. Check yourself

1. Why does CLIP's loss have *two* cross-entropies?
<details><summary>Answer</summary>One over rows (each image must pick its caption among N) and one over columns (each caption must pick its image among N). Averaging them makes the space symmetric: images find texts and texts find images. That is needed both for zero-shot classification (image to class text) and for retrieval in either direction.</details>

2. With batch N and random embeddings, what loss do you expect?
<details><summary>Answer</summary>log N. Every logit is about equal, so the softmax gives 1/N to the right answer and −log(1/N) = log N. For N = 128 that is 4.85, which our demo prints.</details>

3. Why is a learned temperature needed when the embeddings are normalised?
<details><summary>Answer</summary>Cosines lie in [−1, 1], so without scaling the correct logit can beat a wrong one by at most 2. The softmax then can't become confident, especially among thousands of negatives. Multiplying by s (up to 100) lets the correct pair dominate. Learning t = log s avoids tuning it by hand.</details>

4. In what sense is zero-shot CLIP a linear classifier?
<details><summary>Answer</summary>Its scores are s·(I·W_k) with W_k = the normalised text embedding of class k's prompt. That is a linear layer with no bias whose weight rows come from the text encoder (a hypernetwork) instead of from training on labelled images.</details>

5. Why average prompt embeddings rather than average the predicted probabilities?
<details><summary>Answer</summary>Averaging embeddings gives a single vector per class, so classifying costs the same as with one prompt; it can be cached. Averaging probabilities would require one forward pass per template. (Both work, but the embedding average is free at test time.)</details>

6. Why did predicting the caption learn more slowly than contrastive matching?
<details><summary>Answer</summary>The exact words of a caption are highly unpredictable from the image ("Rex at the beach, 2016"). The predictor spends effort modelling phrasing, names and dates. Matching only asks the model to tell which caption fits, which needs only the image's content. The paper measured 3× (bag of words vs exact caption) and another 4× (contrastive vs bag of words).</details>

7. Zero-shot CLIP matches a 4-shot linear probe on its own features. Why isn't 1-shot already better than zero-shot?
<details><summary>Answer</summary>One example per class is ambiguous: the probe can't tell which features matter (object? background? colour?). Zero-shot uses a language description that pinpoints the concept. Only with several examples does the data-driven probe overtake the language prior.</details>

8. What is "effective robustness"?
<details><summary>Answer</summary>The accuracy under distribution shift *above* what the logit-linear trend of ImageNet-trained models predicts at the same ImageNet accuracy. ResNet-101 and zero-shot CLIP both score 76.2 on ImageNet, but on ImageNet-R they score 37.7 vs 88.9, so CLIP sits far above the trend.</details>

9. Fitting a linear probe on ImageNet raised CLIP's ImageNet accuracy by 9.2%. What happened under shift, and why does it matter?
<details><summary>Answer</summary>Average accuracy on the shifted datasets slightly *dropped*. The gain came from ImageNet-specific patterns that don't transfer, which suggests that training on one distribution trades robustness for in-distribution accuracy.</details>

10. In our demo, why does the bare class name work as well as the ensemble, unlike in the paper?
<details><summary>Answer</summary>Our text encoder averages word vectors (a bag of words), so "photo", "of", "the" just dilute the class word's vector, and a single word is not out-of-distribution for it. CLIP's Transformer was trained on sentences, so a lone word is unusual input. It is also ambiguous (polysemy), so context helps there.</details>
