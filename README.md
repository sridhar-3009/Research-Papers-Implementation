# Implementing 103 AI Research Papers, Explained in Plain Language

This repository rebuilds **103 of the most important research papers in artificial intelligence**, from the first artificial neuron in 1943 to today's large language models (the technology behind ChatGPT), and the engineering, safety and economics around them.

For every paper there are three things:

1. **A plain-language explanation** of what the paper says and why it mattered, written for someone who has never studied AI.
2. **Working code written from scratch**, so you can see the idea actually run instead of only reading about it.
3. **An honest check**: does our small version behave the way the paper says it should? Where it doesn't, we say so.

You do **not** need to be an expert to use this. If you can read a little Python, or are just curious, you can follow along from paper 001 to paper 103.

> **Where to start:** read this page first. Then open [READING_ORDER.md](READING_ORDER.md) for the full learning path, and open any paper's folder to read its `EXPLAINED.md`.

---

## Contents

1. [What is this project, really?](#1-what-is-this-project-really)
2. [Who is it for?](#2-who-is-it-for)
3. [What you will find inside each paper's folder](#3-what-you-will-find-inside-each-papers-folder)
4. [How to run it on your own computer, step by step](#4-how-to-run-it-on-your-own-computer-step-by-step)
5. [A two-minute introduction to the words you will see](#5-a-two-minute-introduction-to-the-words-you-will-see)
6. [The story of AI in 16 stages, and every paper explained](#6-the-story-of-ai-in-16-stages-and-every-paper-explained)
7. [How honest are the results?](#7-how-honest-are-the-results)
8. [Other files in this repository](#8-other-files-in-this-repository)

---

## 1. What is this project, really?

Research papers are how new ideas in AI are shared. But papers are written for experts. They are short, full of mathematics, and they skip the steps the authors found obvious. Many people who want to understand AI never get past the first page.

This project takes each paper and does three slow, careful things:

- **Reads it closely**, section by section, and writes down what it actually claims (with the exact numbers from its tables).
- **Rebuilds the method in code from scratch**, usually with only basic tools (NumPy, sometimes PyTorch), instead of calling a ready-made library that hides how it works.
- **Tests the code against the paper.** Small automatic tests check that each piece is correct (for example, that a gradient computed by hand matches one measured numerically), and a short demo shows the paper's main effect happening on a small example.

The papers are arranged in an order where **each one only needs the ones before it**. So if you go from 001 to 103, you never meet an idea that hasn't been explained earlier.

**An important limit:** the original papers often trained enormous models on huge datasets with thousands of computers. That is impossible on a laptop. So every paper here is rebuilt at **toy scale**: tiny models, small or synthetic datasets, a few seconds of computation. The goal is to see the *idea* working, not to match the paper's final scores. Where a full-size experiment would be needed, the code for it is written (in `experiments.py`) but was **not run**, and the explanation says so.

---

## 2. Who is it for?

- **Curious beginners** who want to understand how AI actually works, one idea at a time.
- **Students** who want a guided path through the classic papers, with code they can run and change.
- **Engineers** who use AI tools and want to know what is happening underneath (attention, fine-tuning, quantization, retrieval, monitoring).
- **Anyone who reads AI news** and wants to know what words like "transformer", "RLHF", "RAG" or "LoRA" really mean.

What you need:

- **To read the explanations:** nothing. They are written for beginners, with worked examples using small numbers you can follow by hand.
- **To run the code:** a computer with Python 3.10 or newer. A laptop is enough; nothing here needs a graphics card (GPU).

---

## 3. What you will find inside each paper's folder

Every paper lives in its own folder, numbered 001 to 103, inside a folder for its stage. For example:

```
01-Foundations/                              ← a stage (there are 16, in learning order)
   001-McCulloch-Pitts-1943/                 ← one paper
      EXPLAINED.md                           ← start here: the paper in plain language
      CODE_EXPLAINED.md                      ← a guided tour of the code
      mcculloch_pitts.py (name varies)       ← the implementation itself
      demo.py                                ← a short demonstration you can run
      test_*.py                              ← automatic checks that the code is correct
      experiments.py                         ← bigger experiments (from paper 007 on; written, not run)
      *.pdf                                  ← the original paper (kept on your computer only, not on GitHub)
   002-Rosenblatt-1958/
   ...
```

What each file is for, in more detail:

| File | What it gives you |
|---|---|
| **EXPLAINED.md** | The paper explained simply: the problem it solved, the key idea, the important formulas with every symbol explained, a worked example with small numbers, the paper's own results, what *our* code found (including where it disagreed with the paper), why the paper matters today, and a short quiz ("Check yourself") with hidden answers. |
| **CODE_EXPLAINED.md** | A map of the code: which file does what, how each function connects to an equation or figure in the paper, how to run everything, and ideas to try next. |
| **The main `.py` file** | The method itself, written from scratch with comments. |
| **demo.py** | Runs in seconds to about half a minute and prints the paper's main effect on a small example, with short messages saying what each number means. |
| **test_*.py** | Small automatic tests. They prove things like "this gradient is correct" or "this reproduces the number in Table 2". They finish in about a second. |
| **experiments.py** | Larger experiments that would take minutes to hours, written so you can run them if you have the time or hardware. They were **not** run while building this repository. |

---

## 4. How to run it on your own computer, step by step

**Step 1. Install Python.** You need version 3.10 or newer. Check with:

```bash
python3 --version
```

**Step 2. Download this repository.**

```bash
git clone <this repository's address>
cd "Implementing Research Papers"
```

**Step 3. Install the libraries.** Paper 001 needs nothing extra. Later papers use these common, free libraries:

```bash
pip install numpy scipy scikit-learn torch torchvision matplotlib pytest
```

**Step 4. Pick a paper and run its demo.** For example, the very first one:

```bash
cd 01-Foundations/001-McCulloch-Pitts-1943
python3 demo.py
```

The demo prints its results with short explanations. Compare what you see with the "What our code found" section of that paper's `EXPLAINED.md`.

**Step 5. Run the tests (optional).** From inside the same folder:

```bash
python3 -m pytest -q
```

You should see something like `12 passed`. That means every check in that folder succeeded.

**Step 6. Try the bigger experiments (optional, can be slow).**

```bash
python3 experiments.py --quick        # a short version of every experiment
python3 experiments.py --only e2      # just one experiment
python3 experiments.py --report-only  # print saved results without re-running
```

A few later papers reuse code from an earlier paper's folder (for example, several language-model papers reuse the tiny LLaMA model from paper 056). Run them from their own folder and they will find what they need automatically.

---

## 5. A two-minute introduction to the words you will see

You don't need to memorise these. Come back here whenever a word is unfamiliar. Each paper's `EXPLAINED.md` explains its own terms in more depth.

| Word | Plain meaning |
|---|---|
| **Neural network** | A program made of many simple units ("neurons") connected together. Each connection has a number (a **weight**) that says how strongly one unit influences another. |
| **Training / learning** | Adjusting the weights, a little at a time, so the network's answers get closer to the right ones. |
| **Loss** | A single number measuring how wrong the network currently is. Training tries to make it smaller. |
| **Gradient** | For each weight, the direction and amount it should change to reduce the loss fastest. |
| **Backpropagation** | The method for computing all the gradients efficiently, by passing the error backwards through the network (paper 004). |
| **Overfitting** | When a model memorises its training examples instead of learning the general rule, so it does well on old examples and badly on new ones. |
| **Layer / deep network** | Units are organised in layers; "deep" means many layers stacked on top of each other. |
| **Convolution (CNN)** | A way of processing images by sliding the same small pattern detector across the whole picture (papers 013–016). |
| **Recurrent network (RNN, LSTM)** | A network that reads a sequence (like a sentence) one step at a time, keeping a memory of what it has read so far (papers 021–027). |
| **Embedding** | Turning a word (or image, or anything) into a list of numbers, so that similar things get similar numbers (paper 019). |
| **Attention** | Letting a model, at each step, look back at whichever parts of the input are most relevant right now (paper 028). |
| **Transformer** | A network built almost entirely from attention. It is the basis of modern language models (paper 034). |
| **Language model / LLM** | A model trained to predict the next word in text. "Large" language models are huge versions trained on enormous amounts of text (papers 048–058). |
| **Pretraining / fine-tuning** | First learn general knowledge from lots of data (pretraining), then adjust the model for a specific job with a smaller amount of data (fine-tuning). |
| **Generative model** | A model that creates new things (images, text, sound) rather than just labelling them (papers 039–047). |
| **Reinforcement learning (RL)** | Learning by trial and error from rewards, like training a pet with treats (papers 066–069, 095–097). |
| **RLHF / alignment** | Using human (or AI) judgements of which answer is better to teach a model to be helpful and harmless (papers 066–069). |
| **Retrieval / RAG** | Letting a model look things up in a collection of documents before answering, instead of relying only on memory (papers 070–072). |
| **Quantization** | Storing a model's numbers with fewer bits (for example 4 bits instead of 16), so it uses less memory and runs faster (papers 074–078). |
| **Parameters** | Another word for the weights. "7B parameters" means 7 billion adjustable numbers. |
| **Toy / synthetic data** | Small, made-up data designed so the effect a paper describes can be seen clearly in seconds. |

---

## 6. The story of AI in 16 stages, and every paper explained

Each stage below starts with a short story of **what problem the stage is about** and how its papers connect. Then each paper gets three short parts:

- **The idea:** what the paper is about, in everyday words.
- **What we built:** what the code in its folder does.
- **What we saw:** what happened when we ran it, including any surprises.

Click a paper's title to open its folder.

---

### Stage 01 · Foundations: what is a neural network?

**The story.** In the 1940s, scientists wondered whether thinking could be explained by simple on/off brain cells wired together. This stage follows that question: first a neuron as a tiny logic switch, then a network that can *learn*, then the discovery that a single layer has serious limits, and finally the method (backpropagation) that lets networks with many layers learn. Everything later in this repository is built on these ideas.

#### 001 · [A Logical Calculus of the Ideas Immanent in Nervous Activity](01-Foundations/001-McCulloch-Pitts-1943/) — McCulloch & Pitts, 1943
- **The idea:** a brain cell can be treated as a tiny switch that "fires" when enough of its inputs are on. Wire enough of these switches together and they can compute any logical rule. This was the first mathematical model of a neuron.
- **What we built:** a simulator for these networks, the examples from the paper's first figure, networks that remember things using loops, and a "compiler" that turns a written logic rule into a working network.
- **What we saw:** every example in the paper works, including the famous illusion where touching something cold for a moment can feel hot.

#### 002 · [The Perceptron](01-Foundations/002-Rosenblatt-1958/) — Rosenblatt, 1958
- **The idea:** the first network that **learns from examples**. It is shown pictures, guesses a category, and strengthens or weakens its connections depending on whether it was right.
- **What we built:** the paper's full "photo-perceptron" (a model eye connected to randomly wired units), its learning rules, and the probability formulas that predict how well it will do.
- **What we saw:** the formulas match the simulation, and we reproduced the paper's figures showing the difference between memorising the examples it saw and recognising new ones.

#### 003 · [Perceptrons (Introduction)](01-Foundations/003-Minsky-Papert-1969/) — Minsky & Papert, 1969
- **The idea:** a careful proof of what a single-layer perceptron **cannot** do. Some simple-sounding questions, like "is this shape all in one connected piece?", are impossible for it. This book is often blamed for slowing AI research for years.
- **What we built:** the formal perceptron, a perceptron that checks whether a shape is convex, and computer-checked versions of the book's proofs.
- **What we saw:** the proofs hold. We also measured, as an extra, how complicated a perceptron must be to compute "parity" (is the number of on-pixels odd?).

#### 004 · [Learning Representations by Back-Propagating Errors](01-Foundations/004-Rumelhart-Hinton-Williams-1986/) — Rumelhart, Hinton & Williams, 1986
- **The idea:** **backpropagation**, the method that finally let networks with hidden layers learn. The error at the output is passed backwards, layer by layer, to work out how every weight should change. Almost every network since uses it.
- **What we built:** backpropagation for any layered network (checked against a slow numerical method), plus its version for networks that run over time.
- **What we saw:** we reproduced the paper's mirror-symmetry network (with the same neat 1 : 2 : 4 pattern of weights) and its family-tree network, which invents its own features like nationality and generation. **Surprise:** following the paper's exact recipe, the family-tree network often gets stuck, because the learning signal fades as it travels back through the layers.

#### 005 · [Perceptron: Learning, Generalization, Model Selection, Fault Tolerance, and Role in the Deep Learning Era](01-Foundations/005-Du-et-al-2022/) — Du, Leung, Mow & Swamy, 2022
- **The idea:** a modern review of 70 years of perceptron research, tying Stage 01 together.
- **What we built:** the classic learning rules, a multi-layer network with backpropagation, 11 different training methods, and the usual tricks for avoiding overfitting.
- **What we saw:** we reproduced the review's experiment on the classic Iris flower dataset.

---

### Stage 02 · Training deep networks: making learning actually work

**The story.** Backpropagation works on paper, but in practice deep networks trained badly: learning was slow, signals faded or exploded, and models memorised instead of generalising. This stage collects the practical fixes that made deep learning work: preparing the data, starting the weights at sensible values, smarter ways of taking steps downhill, randomly switching off units during training, and keeping each layer's numbers in a healthy range.

#### 006 · [Efficient BackProp](02-Training-Deep-Networks/006-LeCun-et-al-1998-Efficient-BackProp/) — LeCun, Bottou, Orr & Müller, 1998
- **The idea:** a practical handbook of tricks: scale the inputs, choose a good activation function, start the weights carefully, pick the step size wisely.
- **What we built:** every trick, each tested on handwritten digits, plus tools that measure the "shape" of the loss to choose step sizes automatically.
- **What we saw:** the tricks help as described. **Surprise:** the paper's own formula for the largest safe step size is slightly wrong (it gives 2.38; the correct value is 2.0), because it forgets one detail.

#### 007 · [Understanding the Difficulty of Training Deep Feedforward Neural Networks](02-Training-Deep-Networks/007-Glorot-Bengio-2010-Difficulty-Training-Deep-FF/) — Glorot & Bengio, 2010
- **The idea:** explains *why* deep networks got stuck: with badly chosen starting weights, signals shrink or blow up as they pass through layers. Proposes **"Xavier" initialisation**, a simple formula for starting weights that keeps signals steady.
- **What we built:** both ways of starting weights, several activation functions, and tools that watch the signals in every layer.
- **What we saw:** our measurement of how well signals survive (0.49 with the old method vs 0.80 with Xavier) matches the paper's 0.5 vs 0.8.

#### 008 · [On the Importance of Initialization and Momentum in Deep Learning](02-Training-Deep-Networks/008-Sutskever-et-al-2013-Initialization-and-Momentum/) — Sutskever, Martens, Dahl & Hinton, 2013
- **The idea:** **momentum** means letting each training step keep some of the previous step's direction, like a ball rolling downhill. Combined with good starting weights, it lets plain training handle problems thought to need much fancier methods.
- **What we built:** classical momentum and "Nesterov" momentum, the paper's schedule, and its special ways of starting weights.
- **What we saw:** the paper's theorem checks out exactly. **Surprise:** in some settings Nesterov momentum can blow up where the classic version still works.

#### 009 · [Improving Neural Networks by Preventing Co-adaptation of Feature Detectors](02-Training-Deep-Networks/009-Hinton-et-al-2012-Preventing-Co-adaptation/) — Hinton et al., 2012
- **The idea:** the first version of **dropout**: during training, randomly switch off half the units each time, so no unit can rely too much on any other. This makes the network more robust.
- **What we built:** dropout, the weight limit the paper uses, and the "average network" used at test time.
- **What we saw:** by trying all 1,024 possible smaller networks, we proved the test-time network is *exactly* their (geometric) average, as the paper claims.

#### 010 · [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](02-Training-Deep-Networks/010-Srivastava-et-al-2014-Dropout/) — Srivastava et al., 2014
- **The idea:** the full study of dropout: why it works, how to tune it, and variants of it.
- **What we built:** dropout networks, a version that uses random noise instead of on/off switches, and the paper's comparison with other methods.
- **What we saw:** we confirmed the paper's mathematical result that, for a simple model, dropout is the same as a well-known penalty on large weights.

#### 011 · [Adam: A Method for Stochastic Optimization](02-Training-Deep-Networks/011-Kingma-Ba-2015-Adam/) — Kingma & Ba, 2015
- **The idea:** **Adam**, the most widely used training method today. It adapts the step size for every weight separately, based on how that weight's gradient has behaved recently.
- **What we built:** Adam and six related methods from scratch.
- **What we saw:** our Adam gives the same results as PyTorch's official version, and the paper's guarantees (such as the limit on step size) hold.

#### 012 · [Batch Normalization](02-Training-Deep-Networks/012-Ioffe-Szegedy-2015-Batch-Normalization/) — Ioffe & Szegedy, 2015
- **The idea:** inside the network, keep each layer's numbers centred and of similar size during training. This made training much faster and more stable.
- **What we built:** batch normalisation by hand, including how it learns and how it behaves at test time.
- **What we saw:** our hand-written version matches PyTorch's, and the paper's example of a problem it fixes behaves as described.

---

### Stage 03 · Seeing: networks for images

**The story.** Images are huge (millions of numbers) and the same object can appear anywhere in a picture. Convolutional networks solve this by sliding small pattern detectors across the image. This stage follows them from reading handwritten cheques in the 1990s to the 2012 breakthrough on ImageNet that started the modern deep-learning boom, and on to networks hundreds of layers deep.

#### 013 · [Gradient-Based Learning Applied to Document Recognition (LeNet-5)](03-CNNs-and-Vision/013-LeCun-et-al-1998-LeNet5-Gradient-Based-Learning/) — LeCun et al., 1998
- **The idea:** **LeNet-5**, a convolutional network that read handwritten digits on bank cheques. It introduced the building blocks still used today: convolution, pooling and shared weights.
- **What we built:** LeNet-5 exactly as described, down to its unusual connection pattern and exactly 60,000 adjustable numbers.
- **What we saw:** the details match the paper, including why one of its loss choices makes all outputs collapse to the same answer.

#### 014 · [ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)](03-CNNs-and-Vision/014-Krizhevsky-et-al-2012-AlexNet/) — Krizhevsky, Sutskever & Hinton, 2012
- **The idea:** **AlexNet** won the 2012 ImageNet competition by a huge margin, using a deep convolutional network, a simple activation called ReLU, dropout, and graphics cards. This is often called the start of the deep-learning revolution.
- **What we built:** AlexNet with its original two-GPU design, its normalisation trick and its data-enlarging tricks (cropping, flipping, colour changes).
- **What we saw:** our model has exactly the expected number of parameters (about 61 million). The full ImageNet training is written but not run.

#### 015 · [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)](03-CNNs-and-Vision/015-Simonyan-Zisserman-2015-VGG/) — Simonyan & Zisserman, 2015
- **The idea:** **VGG** showed that simply stacking many small 3×3 filters, making the network deeper, gives better results.
- **What we built:** all six network designs from the paper, with their exact sizes.
- **What we saw:** the sizes match the paper exactly (138 million parameters for the most famous version), and we showed how badly very deep networks train without a careful start.

#### 016 · [Deep Residual Learning for Image Recognition (ResNet)](03-CNNs-and-Vision/016-He-et-al-2016-ResNet/) — He et al., 2016
- **The idea:** **skip connections**: let each block add its output to its input, so the signal has a shortcut path. This made it possible to train networks with over 100 layers, and the idea is now in almost every modern network, including transformers.
- **What we built:** ResNets from 18 to 152 layers, and the very deep small versions (up to 1,202 layers) from the paper.
- **What we saw:** the sizes and amounts of computation match the paper exactly, and we measured how plain networks without shortcuts struggle as they get deeper.

---

### Stage 04 · What networks really learn: surprises

**The story.** Once deep networks worked, researchers found two strange things. They can be fooled by changes to an image so small a person can't see them, and they can perfectly memorise completely random labels. Both papers make you rethink what "learning" means for these models.

#### 017 · [Intriguing Properties of Neural Networks](04-Robustness-and-Generalization/017-Szegedy-et-al-2013-Intriguing-Properties/) — Szegedy et al., 2013
- **The idea:** discovered **adversarial examples**: tiny, invisible changes to an image that make a network confidently give the wrong answer.
- **What we built:** the paper's method for finding the smallest such change, and its way of measuring how sensitive each layer is.
- **What we saw:** on a simple case where the answer is known, the method finds exactly the smallest change.

#### 018 · [Understanding Deep Learning Requires Rethinking Generalization](04-Robustness-and-Generalization/018-Zhang-et-al-2017-Rethinking-Generalization/) — Zhang et al., 2017
- **The idea:** showed that networks can **memorise random labels** perfectly. This means traditional explanations of why they work on new data can't be the whole story.
- **What we built:** all the paper's randomisation tests (shuffled labels, random pixels), and its proof that a fairly small network can fit any labels at all.
- **What we saw:** the proof's construction works as stated.

---

### Stage 05 · Words and sequences: from word vectors to attention

**The story.** Language comes as sequences, and the meaning of a word depends on its neighbours. This stage builds up, one step at a time, to the most important idea in modern AI. First, words become lists of numbers (embeddings). Then networks with memory read sentences word by word (RNNs and LSTMs). Then two networks work together to translate whole sentences (sequence-to-sequence). Finally, **attention** lets the translator look back at the right source words at each step.

#### 019 · [Distributed Representations of Words and Phrases (Word2Vec)](05-Words-and-Sequences/019-Mikolov-et-al-2013-Word2Vec-Negative-Sampling/) — Mikolov et al., 2013
- **The idea:** **Word2Vec** turns each word into a list of numbers so that similar words get similar numbers, learned simply by predicting nearby words. It's famous for arithmetic like *king − man + woman ≈ queen*.
- **What we built:** the model with every training shortcut the paper describes, each checked for correctness.
- **What we saw:** all the training methods compute correct gradients. The full experiment on a large text collection is written but not run.

#### 020 · [Exploiting Similarities among Languages for Machine Translation](05-Words-and-Sequences/020-Mikolov-et-al-2013-Similarities-Among-Languages/) — Mikolov, Le & Sutskever, 2013
- **The idea:** word vectors from two languages have similar shapes, so a simple rotation-like mapping can translate words between them.
- **What we built:** the mapping between languages, the paper's accuracy measures and its baselines.
- **What we saw:** the pieces work as described; the full experiment on real language data is written but not run.

#### 021 · [Long Short-Term Memory (LSTM)](05-Words-and-Sequences/021-Hochreiter-Schmidhuber-1997-LSTM/) — Hochreiter & Schmidhuber, 1997
- **The idea:** the **LSTM**, a memory cell for sequence networks that can keep information for a long time without it fading. It powered speech recognition and translation for two decades.
- **What we built:** the original LSTM (without the "forget gate" added later), with the paper's own learning rule.
- **What we saw:** the memory cell keeps its information perfectly as designed, and we measured a known weakness (its stored value slowly drifts) together with the paper's fix.

#### 022 · [Generating Text with Recurrent Neural Networks](05-Words-and-Sequences/022-Sutskever-et-al-2011-Generating-Text-with-RNNs/) — Sutskever, Martens & Hinton, 2011
- **The idea:** a network that writes text **one character at a time**, using a special design where each input character changes how the memory updates.
- **What we built:** three network designs and a special second-order training method.
- **What we saw:** the training method's maths checks out. The large training run is written but not run.

#### 023 · [Training Recurrent Neural Networks (PhD thesis)](05-Words-and-Sequences/023-Sutskever-2013-PhD-Thesis-Training-RNNs/) — Sutskever, 2013
- **The idea:** a long study of how to make sequence networks learn hard, long-range patterns, including a model that learns videos of bouncing balls.
- **What we built:** the thesis's video model and its improved training method, plus all eight "hard problems" it uses as tests.
- **What we saw:** the methods compute what they should, checked against exact calculations.

#### 024 · [Recurrent Neural Network Regularization](05-Words-and-Sequences/024-Zaremba-et-al-2014-RNN-Regularization/) — Zaremba, Sutskever & Vinyals, 2014
- **The idea:** dropout (from Stage 02) didn't work for LSTMs until this paper showed **where** to apply it: between layers, not on the memory connections.
- **What we built:** a deep LSTM language model with the paper's dropout placement, plus the two wrong placements for comparison.
- **What we saw:** the dropout lands exactly where intended. The full training runs are written but not run.

#### 025 · [An Empirical Exploration of Recurrent Network Architectures](05-Words-and-Sequences/025-Jozefowicz-et-al-2015-Empirical-Exploration-RNN-Architectures/) — Jozefowicz, Zaremba & Sutskever, 2015
- **The idea:** searched through thousands of variations of the LSTM to see which parts matter. A key finding: starting the "forget gate" open makes a big difference.
- **What we built:** all ten memory-cell designs from the paper and the search procedure itself.
- **What we saw:** our general framework reproduces the standard LSTM and GRU exactly.

#### 026 · [Learning Phrase Representations using an RNN Encoder–Decoder](05-Words-and-Sequences/026-Cho-et-al-2014-RNN-Encoder-Decoder/) — Cho et al., 2014
- **The idea:** introduced the **GRU** (a simpler memory cell than the LSTM) and the **encoder–decoder**: one network reads a sentence into a summary, another writes a translation from that summary.
- **What we built:** the GRU and the full encoder–decoder from the paper's appendix.
- **What we saw:** the parts behave as specified; the translation experiments are written but not run.

#### 027 · [Sequence to Sequence Learning with Neural Networks](05-Words-and-Sequences/027-Sutskever-et-al-2014-Seq2Seq/) — Sutskever, Vinyals & Le, 2014
- **The idea:** **sequence-to-sequence**: a deep LSTM reads a whole sentence and another writes the translation. A surprising trick: **reversing the input sentence** makes learning much easier.
- **What we built:** the model, input reversal, and "beam search" (keeping several candidate translations at once).
- **What we saw:** the parts work and beam search finds the best answer on a test case where we know it; the full translation runs are written but not run.

#### 028 · [Neural Machine Translation by Jointly Learning to Align and Translate (Attention)](05-Words-and-Sequences/028-Bahdanau-et-al-2015-Attention-NMT/) — Bahdanau, Cho & Bengio, 2015
- **The idea:** **attention**. Instead of squeezing a whole sentence into one summary, the translator looks back at the most relevant source words at each step. This is the key ingredient of transformers.
- **What we built:** the paper's attention model exactly, and the older model without attention.
- **What we saw:** on reversing 16-symbol sequences, the old model managed 59% while the attention model got 100%, and the attention pattern clearly showed it "looking" at the right positions.

#### 029 · [Show and Tell: A Neural Image Caption Generator](05-Words-and-Sequences/029-Vinyals-et-al-2015-Show-and-Tell/) — Vinyals, Toshev, Bengio & Erhan, 2015
- **The idea:** describe a picture in a sentence by connecting an image network to a sentence-writing LSTM.
- **What we built:** the captioning model and the standard caption-scoring measures.
- **What we saw:** in a small made-up picture world, it wrote correct captions for combinations it had never seen. It also "beat humans" on the automatic score, showing why that score can be misleading.

#### 030 · [Grammar as a Foreign Language](05-Words-and-Sequences/030-Vinyals-et-al-2015-Grammar-as-Foreign-Language/) — Vinyals et al., 2015
- **The idea:** treat working out a sentence's grammatical structure as "translating" it into a bracketed tree.
- **What we built:** converting trees to and from text, the attention model, and the standard scoring tool.
- **What we saw:** on a toy grammar, attention clearly helped (89.8 vs 80.5 on the standard score, with a bigger gap on long sentences).

#### 031 · [Pointer Networks](05-Words-and-Sequences/031-Vinyals-et-al-2015-Pointer-Networks/) — Vinyals, Fortunato & Jaitly, 2015
- **The idea:** use attention to **point** at items in the input, so the output can only be input positions. Useful for sorting, or for drawing the outline around a set of points.
- **What we built:** the model, its baselines, and exact solvers for the geometry problems to compare against.
- **What we saw:** one model could sort lists of different lengths, including a length it had never seen.

#### 032 · [Order Matters: Sequence to Sequence for Sets](05-Words-and-Sequences/032-Vinyals-et-al-2015-Order-Matters/) — Vinyals, Bengio & Kudlur, 2016
- **The idea:** when the input or output is really a *set*, the order you present it in still affects how well the model learns, and you can design for that.
- **What we built:** the paper's order-independent reader and its search over output orders.
- **What we saw:** on sorting, the paper's design (84%) beat the pointer network (75%), as the paper reports.

#### 033 · [Neural Machine Translation in Linear Time (ByteNet)](05-Words-and-Sequences/033-Kalchbrenner-et-al-2016-ByteNet/) — Kalchbrenner et al., 2016
- **The idea:** process sequences with **convolutions** that have growing gaps ("dilations"), instead of step-by-step networks, so the work grows only in proportion to the length.
- **What we built:** the full model with its dilated convolutions.
- **What we saw:** dilation let it learn a pattern 10 steps back that a plain stack could not. **Surprise:** a number in the paper (a "receptive field" of 315) doesn't match its own formula; we get 373.

---

### Stage 06 · Transformers: attention is all you need

**The story.** In 2017 a team showed you could drop the step-by-step memory entirely and build a network out of attention alone. The **transformer** trains faster and scales further than anything before it, and it is the engine inside ChatGPT, BERT and image models alike. This stage covers the original design, ways to make it faster and handle longer inputs, and its use for understanding text (BERT) and for images (ViT).

#### 034 · [Attention Is All You Need (the Transformer)](06-Transformers/034-Vaswani-et-al-2017-Attention-Is-All-You-Need/) — Vaswani et al., 2017
- **The idea:** the **Transformer**. Every word looks at every other word through attention (many "heads" at once), and a trick called positional encoding tells it the word order.
- **What we built:** the complete model from scratch, with its training tricks and beam search.
- **What we saw:** our sizes are close to the paper's (63 vs 65 million parameters for the base model). On a reversing task it got 50 out of 50 right, with one attention head clearly learning the reversal.

#### 035 · [Fast Transformer Decoding: One Write-Head is All You Need](06-Transformers/035-Shazeer-2019-Multi-Query-Attention/) — Shazeer, 2019
- **The idea:** **multi-query attention**: share one set of "keys" and "values" across all heads, which makes text generation much faster with little loss in quality.
- **What we built:** standard, multi-query and the in-between "grouped-query" attention, with the memory cache used during generation.
- **What we saw:** the cache became 8 times smaller and generation on a laptop processor got faster (6.0 → 2.2 milliseconds per step).

#### 036 · [Generating Long Sequences with Sparse Transformers](06-Transformers/036-Child-et-al-2019-Sparse-Transformers/) — Child, Gray, Radford & Sutskever, 2019
- **The idea:** let each position attend to only a structured subset of others, so very long sequences become affordable.
- **What we built:** both of the paper's sparse attention patterns.
- **What we saw:** 11 to 32 times fewer pairs to compute; one pattern copied repeating structure as well as full attention, while the other struggled.

#### 037 · [BERT](06-Transformers/037-Devlin-et-al-2019-BERT/) — Devlin, Chang, Lee & Toutanova, 2019
- **The idea:** **BERT** learns language by filling in hidden words using context from **both** sides, then is fine-tuned for many tasks.
- **What we built:** the full recipe: word pieces, hiding 15% of words, the "is this the next sentence?" task, and heads for classification and question answering.
- **What we saw:** when the missing word depends on what comes after it, BERT-style training recovered it 100% of the time, against 9.5% for a left-to-right model.

#### 038 · [An Image is Worth 16x16 Words (Vision Transformer)](06-Transformers/038-Dosovitskiy-et-al-2021-ViT/) — Dosovitskiy et al., 2021
- **The idea:** cut an image into small patches, treat each patch like a word, and use a plain transformer.
- **What we built:** the Vision Transformer and the paper's tools for looking inside it.
- **What we saw:** with very little data, a convolutional network won easily (100% vs 40.8%). That matches the paper's point that transformers need lots of data because they have fewer built-in assumptions about images.

---

### Stage 07 · Generative models: networks that create

**The story.** Instead of labelling data, these models *create* it: new faces, digits, pictures from a sentence. This stage covers the main families: VAEs (compress and rebuild), GANs (a forger against a detective), pixel-by-pixel models, invertible "flows", and finally DALL·E, which draws images from text descriptions.

#### 039 · [Auto-Encoding Variational Bayes (VAE)](07-Generative-Models/039-Kingma-Welling-2014-VAE/) — Kingma & Welling, 2014
- **The idea:** the **VAE** squeezes data into a small "code" and learns to rebuild it. A clever "reparameterisation trick" makes this trainable.
- **What we built:** the VAE, both of the paper's estimators, and the older methods it beat.
- **What we saw:** the trick made learning signals 8 to 12 times less noisy, and the VAE beat the older "wake-sleep" method on digits.

#### 040 · [Improved Variational Inference with Inverse Autoregressive Flow](07-Generative-Models/040-Kingma-et-al-2016-Inverse-Autoregressive-Flow/) — Kingma et al., 2016
- **The idea:** make a VAE's guesses about the hidden code more flexible by passing them through a chain of invertible steps (a "flow").
- **What we built:** the flow and its building blocks.
- **What we saw:** the simple version was stuck with a fixed error, while the flow brought it close to zero.

#### 041 · [Variational Lossy Autoencoder](07-Generative-Models/041-Chen-et-al-2017-Variational-Lossy-Autoencoder/) — Chen et al., 2017
- **The idea:** control **what** information the code stores (for example, the overall shape but not fine texture) by limiting what the decoder can see.
- **What we built:** decoders that see only a small window, and the paper's information accounting.
- **What we saw:** in a toy with one important global fact, the limited decoder kept exactly that fact in the code, as the paper predicts.

#### 042 · [Generative Adversarial Nets (GAN)](07-Generative-Models/042-Goodfellow-et-al-2014-GAN/) — Goodfellow et al., 2014
- **The idea:** a **forger** network makes fake samples and a **detective** network tries to spot them. Each improves by competing with the other.
- **What we built:** the original GAN and the paper's way of measuring it.
- **What we saw:** the theory holds exactly, and a one-dimensional GAN learned its target. We also reproduced two famous problems: training swings back and forth, and the forger can collapse to producing only a few kinds of output.

#### 043 · [InfoGAN](07-Generative-Models/043-Chen-et-al-2016-InfoGAN/) — Chen et al., 2016
- **The idea:** add "control knobs" to a GAN that end up meaning something (like digit type or slant), without any labels.
- **What we built:** InfoGAN with its information-based training.
- **What we saw:** the knobs grouped the data correctly 100% of the time without labels, against 26% for a plain GAN.

#### 044 · [Wasserstein GAN](07-Generative-Models/044-Arjovsky-et-al-2017-Wasserstein-GAN/) — Arjovsky, Chintala & Bottou, 2017
- **The idea:** use a different way of measuring how far apart two distributions are (the "earth mover's distance"), which gives smoother learning signals and steadier GAN training.
- **What we built:** all the distance measures, the new critic network and its training.
- **What we saw:** the old measure gave a flat, useless signal while the new one gave a steady slope, and the new GAN smoothly moved to the right answer.

#### 045 · [PixelCNN++](07-Generative-Models/045-Salimans-et-al-2017-PixelCNN-plus-plus/) — Salimans et al., 2017
- **The idea:** generate an image one pixel at a time, each pixel predicted from the ones before it, with several improvements over earlier versions.
- **What we built:** the paper's pixel probability model and its network design.
- **What we saw:** the simpler model memorised its training images, while the improved one generalised better.

#### 046 · [Glow](07-Generative-Models/046-Kingma-Dhariwal-2018-Glow/) — Kingma & Dhariwal, 2018
- **The idea:** a fully **invertible** network: you can turn an image into a code and back again exactly. That makes exact probability calculations possible.
- **What we built:** all of Glow's building blocks.
- **What we saw:** the network was exactly invertible, and its fast probability calculation matched the slow exact one to six decimal places.

#### 047 · [Zero-Shot Text-to-Image Generation (DALL·E)](07-Generative-Models/047-Ramesh-et-al-2021-DALL-E/) — Ramesh et al., 2021
- **The idea:** **DALL·E** turns images into "visual words" and trains one transformer on text followed by image, so it can draw a picture from a caption.
- **What we built:** the image-to-token compressor, the text-plus-image transformer, and the trick of generating many images and keeping the best.
- **What we saw:** it composed colours and positions it had never seen together about half the time ("inconsistent", as the paper honestly says), and picking the best of several samples raised accuracy from 83% to 100%.

---

### Stage 08 · Pretraining and scaling: how large language models are made

**The story.** This is where ChatGPT-style models come from. Train a transformer to predict the next word on huge amounts of text, and it picks up grammar, facts and even skills. Make it bigger and feed it more data, and it gets predictably better. This stage covers the GPT series, the "scaling laws" that predict how good a model will be, open models like LLaMA, models for images and text (CLIP), code (Codex) and speech (Whisper), and how GPT-4 was evaluated.

#### 048 · [Improving Language Understanding by Generative Pre-Training (GPT)](08-Pretraining-and-Scaling-LLMs/048-Radford-et-al-2018-GPT/) — Radford et al., 2018
- **The idea:** **GPT**: first learn language by predicting the next word on lots of text, then fine-tune on a small labelled task.
- **What we built:** the model, the pretraining, and the paper's ways of turning tasks into text.
- **What we saw:** with only 40 labelled examples, the pretrained model reached 80–82% against 73% when trained from scratch. It could also classify reviews with **no labels at all** (89%), just by comparing "very good" with "very bad".

#### 049 · [Language Models are Unsupervised Multitask Learners (GPT-2)](08-Pretraining-and-Scaling-LLMs/049-Radford-et-al-2019-GPT-2/) — Radford et al., 2019
- **The idea:** **GPT-2** showed that a big enough language model can do tasks like answering questions **without any fine-tuning**, just from how the question is phrased.
- **What we built:** GPT-2's way of splitting text into tokens, its exact model sizes, and its method for detecting test data leaked into training.
- **What we saw:** in a toy "web", it answered questions about facts it had only seen written as normal prose.

#### 050 · [Language Models are Few-Shot Learners (GPT-3)](08-Pretraining-and-Scaling-LLMs/050-Brown-et-al-2020-GPT3/) — Brown et al., 2020
- **The idea:** **GPT-3** (175 billion parameters) can learn a new task from a few examples written in the prompt (**in-context learning**), with no retraining.
- **What we built:** GPT-3's size and compute formulas, its data filtering and de-duplication, and the prompt formats.
- **What we saw:** a small model clearly learned from the examples in its context, coming within 0.02 of the best possible score on a task designed to measure this.

#### 051 · [Scaling Laws for Neural Language Models](08-Pretraining-and-Scaling-LLMs/051-Kaplan-et-al-2020-Scaling-Laws/) — Kaplan et al., 2020
- **The idea:** a model's error falls smoothly and predictably as you add parameters, data and computing power, following simple "power laws".
- **What we built:** every formula in the paper and a calculator for the best model size for a given budget.
- **What we saw:** the calculator agrees with the paper's own table to within about a factor of 2, and the paper's numbers fit together consistently.

#### 052 · [Training Compute-Optimal Large Language Models (Chinchilla)](08-Pretraining-and-Scaling-LLMs/052-Hoffmann-et-al-2022-Chinchilla/) — Hoffmann et al., 2022
- **The idea:** a correction to the scaling laws: for the best results, train on **far more data**, about 20 words (tokens) per parameter. Models before this were too big for the data they saw.
- **What we built:** all three of the paper's methods for finding the best balance.
- **What we saw:** all three recovered the correct answer from data we generated, and at equal computing cost the "Chinchilla" balance scored better than the older approach.

#### 053 · [Learning Transferable Visual Models from Natural Language Supervision (CLIP)](08-Pretraining-and-Scaling-LLMs/053-Radford-et-al-2021-CLIP/) — Radford et al., 2021
- **The idea:** **CLIP** learns to match pictures with their captions, which lets it recognise new kinds of images just from a text description, with no labelled examples.
- **What we built:** CLIP's training method and its "zero-shot" classifier built from text prompts.
- **What we saw:** trained only on captions, it recognised digits 88% of the time with no digit labels, and matching captions worked far better than predicting the exact caption words (88% vs 49%).

#### 054 · [Evaluating Large Language Models Trained on Code (Codex)](08-Pretraining-and-Scaling-LLMs/054-Chen-et-al-2021-Codex/) — Chen et al., 2021
- **The idea:** **Codex** writes programs. The paper's key contribution is judging code by **running it against tests** ("pass@k"), not by how similar it looks to a reference.
- **What we built:** the fair way of computing pass@k, a safe harness for running generated code, and sampling settings.
- **What we saw:** a buggy program scored high on text similarity while a correct one scored low, which shows why running tests matters.

#### 055 · [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](08-Pretraining-and-Scaling-LLMs/055-Radford-et-al-2023-Whisper/) — Radford et al., 2022
- **The idea:** **Whisper** learns speech recognition from 680,000 hours of imperfectly labelled audio, and becomes robust to noise and accents.
- **What we built:** Whisper's audio processing, model design and output format.
- **What we saw:** a tiny model trained only on clean sound fell apart with any noise, while one trained on varied noise stayed accurate, which is the paper's main lesson.

#### 056 · [LLaMA: Open and Efficient Foundation Language Models](08-Pretraining-and-Scaling-LLMs/056-Touvron-et-al-2023-LLaMA/) — Touvron et al., 2023
- **The idea:** **LLaMA**, a family of openly released models trained on public data, with several small design improvements.
- **What we built:** a tiny LLaMA with all its improvements. Later papers in this repository reuse it.
- **What we saw:** the paper's model sizes, training time and carbon figures match our calculations, and our efficient attention gives the same answer as the slow version.

#### 057 · [GPT-4 Technical Report](08-Pretraining-and-Scaling-LLMs/057-OpenAI-2023-GPT-4-Technical-Report/) — OpenAI, 2023
- **The idea:** the report keeps the model's design secret, but explains how OpenAI **predicted** GPT-4's performance from much smaller models, and how it checked calibration and test-data leaks.
- **What we built:** those prediction and checking methods.
- **What we saw:** predicting a real tiny model's result from smaller ones came within 4.8%, and the leak checker caught copied text but missed paraphrases.

#### 058 · [GPT-4o System Card](08-Pretraining-and-Scaling-LLMs/058-OpenAI-2024-GPT-4o-System-Card/) — OpenAI, 2024
- **The idea:** how a model is evaluated for safety before release: risk categories, scoring rules, and checks such as making sure the voice model only speaks in approved voices.
- **What we built:** the scoring rules and a toy voice-checking system.
- **What we saw:** the voice checker shows a real trade-off: to catch a very similar-sounding voice, it must also wrongly flag many approved clips.

---

### Stage 09 · Reasoning and agents: getting more out of language models

**The story.** Once you have a language model, how do you make it think more carefully, check its own work, and act in the world? This stage covers "think step by step" prompting, a second model that checks answers, and **agents**: models that reason, use tools like search, remember, and plan.

#### 059 · [Chain-of-Thought Prompting Elicits Reasoning](09-Reasoning-and-Agents/059-Wei-et-al-2022-Chain-of-Thought/) — Wei et al., 2022
- **The idea:** show the model examples that **write out their reasoning step by step**, and it reasons better on new problems.
- **What we built:** the paper's prompts and a multi-step toy problem tested in several formats.
- **What we saw:** step-by-step answers stayed at 100% as problems got longer, while direct answers collapsed to around 11%. Like the paper, padding with meaningless dots did not help.

#### 060 · [Large Language Models are Zero-Shot Reasoners](09-Reasoning-and-Agents/060-Kojima-et-al-2022-Zero-Shot-Reasoners/) — Kojima et al., 2022
- **The idea:** simply adding **"Let's think step by step"** to a question makes models reason better, with no examples needed.
- **What we built:** the two-step prompting method and all 16 trigger phrases the paper compares.
- **What we saw:** the reasoning phrase kept accuracy high (83% on the hardest level, against 13% without it), while irrelevant or misleading phrases did not help, matching the paper's pattern.

#### 061 · [Training Verifiers to Solve Math Word Problems](09-Reasoning-and-Agents/061-Cobbe-et-al-2021-Training-Verifiers-Math/) — Cobbe et al., 2021
- **The idea:** generate many candidate solutions, then use a second model (a **verifier**) to pick the one most likely to be right.
- **What we built:** the generator, the verifier and the voting methods.
- **What we saw:** with little training data the verifier made things worse; with more data it helped clearly (75%, then 81% with voting). The paper predicts exactly this.

#### 062 · [ReAct: Synergizing Reasoning and Acting](09-Reasoning-and-Agents/062-Yao-et-al-2023-ReAct/) — Yao et al., 2023
- **The idea:** an agent alternates **thinking** and **acting** (searching, reading) in a loop. This is the basis of most AI agents today.
- **What we built:** a Wikipedia-like environment, the think/act/observe loop, and the paper's fallback strategies.
- **What we saw:** a model answering from memory gave outdated answers when facts changed (3% right), while the acting agent read the pages and got 100%.

#### 063 · [WebGPT: Browser-Assisted Question Answering](09-Reasoning-and-Agents/063-Nakano-et-al-2021-WebGPT/) — Nakano et al., 2021
- **The idea:** a model that **browses the web** to answer questions, trained with human feedback on which answers are better.
- **What we built:** a text browser, a scoring model learned from comparisons, best-of-n selection and reinforcement learning.
- **What we saw:** a classic trap appeared. Pushing too hard on the scoring model made answers *worse* in reality while the score kept rising (the model learned that longer looked better).

#### 064 · [Generative Agents: Interactive Simulacra of Human Behavior](09-Reasoning-and-Agents/064-Park-et-al-2023-Generative-Agents/) — Park et al., 2023
- **The idea:** a town of 25 AI characters with memories, reflections and daily plans, who spread news and organise a party on their own.
- **What we built:** the memory system with its scoring, reflection and planning, plus the town (with simple stand-ins in place of a real language model).
- **What we saw:** news of a party spread to about 9–12 of the 25 agents, and switching off the "importance" part of memory stopped it spreading entirely.

#### 065 · [A Survey on Large Language Model based Autonomous Agents](09-Reasoning-and-Agents/065-Wang-et-al-2023-LLM-Agents-Survey/) — Wang et al., 2023
- **The idea:** a map of the whole agent field: profiles, memory, planning and action.
- **What we built:** the survey's framework as plug-together parts, with five planning strategies.
- **What we saw:** on a six-step task, planning with feedback from the environment succeeded about 95% of the time, against about 25% for planning once and hoping.

---

### Stage 10 · Alignment: teaching models what people want

**The story.** A model that predicts text is not automatically helpful or safe. This stage covers how assistants like ChatGPT are trained: people compare pairs of answers, a "reward model" learns their preferences, and the language model is then trained to score well on it. It also covers using AI feedback guided by written principles instead of human labels, and a simpler method (DPO) that skips the reinforcement-learning step.

#### 066 · [Learning to Summarize from Human Feedback](10-Alignment/066-Stiennon-et-al-2020-Summarize-from-Human-Feedback/) — Stiennon et al., 2020
- **The idea:** train a **reward model** from human comparisons of summaries, then improve the summariser with reinforcement learning.
- **What we built:** the whole pipeline with real (tiny) networks: supervised training, comparisons, reward model and reinforcement learning.
- **What we saw:** quality rose clearly. Without a penalty for drifting too far from the original model, it exploited the reward model by repeating the main topic: a higher score but worse real quality.

#### 067 · [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](10-Alignment/067-Ouyang-et-al-2022-InstructGPT/) — Ouyang et al., 2022
- **The idea:** the full **RLHF** recipe behind ChatGPT: learn from demonstrations, then rankings, then reinforcement learning, while mixing in pretraining so the model doesn't lose old skills.
- **What we built:** every stage with tiny real networks.
- **What we saw:** mixing in pretraining kept the new helpfulness while recovering lost skills. **Honest result:** reinforcement learning did not beat simple supervised training here, because our reward model was too weak.

#### 068 · [Constitutional AI: Harmlessness from AI Feedback](10-Alignment/068-Bai-et-al-2022-Constitutional-AI/) — Bai et al., 2022
- **The idea:** instead of human labels, the AI critiques and revises its own answers using a written list of principles (a "constitution").
- **What we built:** the critique-and-revise loop and AI-generated preference labels, in a small simulated text world.
- **What we saw:** harmful answers fell from 100% to under 1% after a few revisions. Labels that rewarded dodging questions produced an evasive model, while the constitution produced one that explains instead.

#### 069 · [Direct Preference Optimization (DPO)](10-Alignment/069-Rafailov-et-al-2023-DPO/) — Rafailov et al., 2023
- **The idea:** **DPO** gets the same effect as RLHF with one simple training loss, no separate reward model and no reinforcement learning.
- **What we built:** DPO, a full RLHF baseline, and a task where the best possible result can be computed exactly.
- **What we saw:** DPO got close to the best possible (87–98%). **Honest result:** RLHF got even closer here (99–100%), so the paper's claim that DPO beats it was not reproduced on this toy.

---

### Stage 11 · Retrieval: giving models knowledge they can look up

**The story.** Models forget, make things up, and go out of date. Retrieval lets a model search a collection of documents first and answer from what it finds. Search engines did this with keywords; these papers do it with meaning (embeddings), then connect search and answer-writing so both learn together. This is one of the most widely used patterns in AI applications today ("RAG").

#### 070 · [Dense Passage Retrieval for Open-Domain Question Answering](11-Retrieval-RAG/070-Karpukhin-et-al-2020-Dense-Passage-Retrieval/) — Karpukhin et al., 2020
- **The idea:** search by **meaning** using embeddings, instead of matching keywords.
- **What we built:** a toy Wikipedia, the classic keyword search (BM25), the embedding-based retriever, and a mix of the two.
- **What we saw:** for reworded questions, meaning-based search found the right page 82% of the time against 19% for keywords. But for names it had never seen, keywords won (83% vs 49%). Combining both was best everywhere.

#### 071 · [Retrieval-Augmented Generation (RAG)](11-Retrieval-RAG/071-Lewis-et-al-2020-RAG/) — Lewis et al., 2020
- **The idea:** **RAG**: retrieve documents, then write the answer from them, training search and writing together.
- **What we built:** both versions of RAG from the paper, and swapping the document collection without retraining.
- **What we saw:** answering from memory got 9%, while RAG with a learned retriever got 58%. Swapping in an updated collection changed the answers correctly.

#### 072 · [Atlas: Few-shot Learning with Retrieval Augmented Language Models](11-Retrieval-RAG/072-Izacard-et-al-2022-Atlas/) — Izacard et al., 2022
- **The idea:** a retrieval model that learns new tasks from just a few examples, by pre-training the searcher and the reader together.
- **What we built:** the reader, four ways of training the retriever, joint pre-training and index compression.
- **What we saw:** with 64 examples, joint pre-training reached about 50% against 7% without it, and compressing the index 16 times lost almost nothing.

---

### Stage 12 · Efficient fine-tuning and quantization: big models on small hardware

**The story.** Large models need huge amounts of memory. This stage covers two families of tricks used every day by AI engineers. **LoRA** adapts a model by training a tiny add-on instead of all its weights. **Quantization** stores the model's numbers in 8 or 4 bits instead of 16 or 32. Combined (QLoRA), they let you fine-tune a huge model on a single graphics card.

#### 073 · [LoRA: Low-Rank Adaptation of Large Language Models](12-Efficient-Finetuning-and-Quantization/073-Hu-et-al-2022-LoRA/) — Hu et al., 2022
- **The idea:** **LoRA**: freeze the big model and train a tiny "patch" made of two thin matrices. Far fewer numbers to train and store.
- **What we built:** LoRA on our tiny LLaMA, teaching it a new task.
- **What we saw:** with 4,096 trainable numbers LoRA got 97.4%, against 99.8% for full fine-tuning with 133,000. Switching the patch off brought back the original skill that full fine-tuning had destroyed.

#### 074 · [SmoothQuant](12-Efficient-Finetuning-and-Quantization/074-Xiao-et-al-2023-SmoothQuant/) — Xiao et al., 2023
- **The idea:** a few huge values inside the model ruin 8-bit storage. **SmoothQuant** moves that difficulty from the activations to the weights, where it's easier to handle.
- **What we built:** 8-bit quantisation at several levels of detail, and the smoothing trick.
- **What we saw:** naive 8-bit collapsed to about 6–8% accuracy, while SmoothQuant kept the full-precision 98.9%.

#### 075 · [GPTQ: Accurate Post-Training Quantization](12-Efficient-Finetuning-and-Quantization/075-Frantar-et-al-2023-GPTQ/) — Frantar et al., 2023
- **The idea:** **GPTQ** rounds weights to 4 (or fewer) bits one at a time, adjusting the remaining weights to make up for each rounding error.
- **What we built:** GPTQ, the simple rounding baseline, and the slow exact method it approximates.
- **What we saw:** at a harsh 2 bits, simple rounding fell to 75.5% while GPTQ kept 97.7% (full precision: 98.6%). The fast method gave exactly the same result as the slow one.

#### 076 · [AWQ: Activation-aware Weight Quantization](12-Efficient-Finetuning-and-Quantization/076-Lin-et-al-2023-AWQ/) — Lin et al., 2023
- **The idea:** a small fraction of weights matter most, the ones connected to large activations. Protect them by scaling before rounding.
- **What we built:** AWQ's search for the best scaling, and the comparison with GPTQ.
- **What we saw:** protecting just 3 important channels rescued 4-bit accuracy from 27% to 99%, and AWQ kept 98%.

#### 077 · [QLoRA: Efficient Finetuning of Quantized LLMs](12-Efficient-Finetuning-and-Quantization/077-Dettmers-et-al-2023-QLoRA/) — Dettmers et al., 2023
- **The idea:** **QLoRA**: store the frozen model in 4 bits (with a new number format, "NF4") and train LoRA patches on top. A 65-billion-parameter model can then be fine-tuned on one GPU.
- **What we built:** the NF4 format, double quantisation and QLoRA on our tiny model.
- **What we saw:** QLoRA matched full 16-bit fine-tuning (99.3% vs 99.7%), and for a 65B model the memory need falls from about 786 GB to about 43 GB.

#### 078 · [LLM-QAT: Data-Free Quantization Aware Training](12-Efficient-Finetuning-and-Quantization/078-Liu-et-al-2023-LLM-QAT/) — Liu et al., 2023
- **The idea:** train the model *while* it is quantised so it adapts, using text the model generates itself as training data.
- **What we built:** quantisation-aware training and self-generated training data.
- **What we saw:** at very low precision, simple rounding fell to 2.5%, while training on self-generated text recovered 96.7%.

---

### Stage 13 · Systems: making training and inference fast

**The story.** Big models are only possible because of clever engineering: computing attention without wasting memory, splitting a model across many chips, and coordinating thousands of computers. This stage covers those systems ideas, from early distributed training to the techniques used to train today's largest models.

#### 079 · [FlashAttention](13-Systems-Inference-and-Training/079-Dao-et-al-2022-FlashAttention/) — Dao et al., 2022
- **The idea:** compute attention in small tiles that fit in the chip's fast memory, avoiding slow trips to main memory. Exact same answer, much faster.
- **What we built:** standard and tiled attention, with a counter for memory traffic.
- **What we saw:** identical results, with about 9 times less memory traffic at length 1,024 (and the traffic scaling as the paper's theory says).

#### 080 · [Efficiently Scaling Transformer Inference](13-Systems-Inference-and-Training/080-Pope-et-al-2022-Efficiently-Scaling-Transformer-Inference/) — Pope et al., 2022
- **The idea:** how to split a huge model across many chips to answer requests quickly and cheaply.
- **What we built:** the paper's cost model and a simulation of the different ways of splitting the work.
- **What we saw:** we reproduced the paper's table of maximum context lengths (1,333 / 666 / 42,654 vs its 1,320 / 660 / 43,000).

#### 081 · [Parallelized Stochastic Gradient Descent](13-Systems-Inference-and-Training/081-Zinkevich-et-al-2010-Parallelized-SGD/) — Zinkevich et al., 2010
- **The idea:** train separate copies of a model on different machines with no communication, then simply average them at the end.
- **What we built:** the method with many simulated machines, plus checks of its theory.
- **What we saw:** going from 1 to 10 machines helped far more than going from 10 to 100.

#### 082 · [TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](13-Systems-Inference-and-Training/082-Abadi-et-al-2015-TensorFlow-Whitepaper/) — Abadi et al., 2015
- **The idea:** describe a computation as a graph of operations, and let the system work out gradients and where each piece should run.
- **What we built:** a miniature TensorFlow from scratch, with automatic gradients and device placement.
- **What we saw:** exact gradients, and smart placement ran 3 times faster (4.6 vs 15.7 ms) with identical results.

#### 083 · [TensorFlow: A System for Large-Scale Machine Learning](13-Systems-Inference-and-Training/083-Abadi-et-al-2016-TensorFlow-OSDI/) — Abadi et al., 2016
- **The idea:** the production design of TensorFlow, including splitting huge tables across machines and using spare "backup" workers to avoid waiting for slow machines.
- **What we built:** these features on top of our mini-TensorFlow, plus a simulator of slow machines.
- **What we saw:** backup workers sped training up, with the best number close to the paper's.

#### 084 · [GPipe: Pipeline Parallelism](13-Systems-Inference-and-Training/084-Huang-et-al-2019-GPipe/) — Huang et al., 2019
- **The idea:** split a model into stages on different chips, and feed small "micro-batches" through like an assembly line.
- **What we built:** a simulator and a real working pipeline in PyTorch.
- **What we saw:** the pipeline gives exactly the same results as ordinary training, and the speed-up (6.56 with 8 stages) is close to the paper's 6.3.

#### 085 · [PipeDream: Generalized Pipeline Parallelism](13-Systems-Inference-and-Training/085-Narayanan-et-al-2019-PipeDream/) — Narayanan et al., 2019
- **The idea:** a smarter pipeline that keeps every chip busy all the time, with an automatic planner for splitting the model.
- **What we built:** the planner, the schedule, and a real pipelined network.
- **What we saw:** the planner rediscovered the paper's own layout for a famous network. **Surprise:** a shortcut the paper warns against didn't cause problems on our small model.

#### 086 · [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](13-Systems-Inference-and-Training/086-Rajbhandari-et-al-2020-ZeRO/) — Rajbhandari et al., 2020
- **The idea:** stop every GPU from storing a full copy of everything; split the training state across GPUs instead.
- **What we built:** the memory formulas and a simulated multi-GPU training job.
- **What we saw:** the paper's memory figures were reproduced exactly (120 GB → 1.9 GB per GPU), with identical training results.

---

### Stage 14 · Production machine learning: keeping AI working in the real world

**The story.** Building a model is the easy part. Keeping it working for years is harder: data changes, small code changes ripple unexpectedly, and models quietly get worse. These papers explain the hidden costs of real ML systems ("technical debt"), how to test and monitor them, how to detect when incoming data has shifted, and how to document models honestly.

#### 087 · [Machine Learning: The High-Interest Credit Card of Technical Debt](14-Production-ML/087-Sculley-et-al-2014-High-Interest-Credit-Card-Technical-Debt/) — Sculley et al., 2014
- **The idea:** ML systems pile up hidden maintenance costs: "changing anything changes everything", hidden feedback loops, and unused inputs that can break things later.
- **What we built:** each warning in the paper as a small experiment.
- **What we saw:** removing one input shifted other parts of the model, quietly cleaning up an old data field hurt a model that still relied on it, and a fixed decision threshold lost precision after retraining.

#### 088 · [Hidden Technical Debt in Machine Learning Systems](14-Production-ML/088-Sculley-et-al-2015-Hidden-Technical-Debt/) — Sculley et al., 2015
- **The idea:** the famous picture: the ML code is a tiny box inside a huge system of data pipelines, configuration and monitoring.
- **What we built:** feedback-loop simulations, a configuration checker, and monitoring tools.
- **What we saw:** a recommender that only learns from what it shows got stuck on a non-best item in 75% of runs, and in our own tiny pipeline the actual ML was only 19% of the code.

#### 089 · [The ML Test Score: A Rubric for ML Production Readiness](14-Production-ML/089-Breck-et-al-2017-ML-Test-Score/) — Breck et al., 2017
- **The idea:** a checklist of 28 tests (data, model, infrastructure, monitoring) and a score for how production-ready a system is.
- **What we built:** the scoring rule and 26 working automated tests on a toy system.
- **What we saw:** each of 11 deliberately injected bugs was caught by at least one test.

#### 090 · [Rules of Machine Learning](14-Production-ML/090-Zinkevich-Rules-of-ML/) — Zinkevich (Google)
- **The idea:** 43 practical rules, such as "launch with a simple heuristic first" and "log the features you actually used at serving time".
- **What we built:** experiments for the rules that make a measurable claim.
- **What we saw:** a simple heuristic got 44% of the way to the ML model (the guide says about 50%), and correctly weighting sampled data fixed a model that had doubled its predictions.

#### 091 · [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](14-Production-ML/091-Rabanser-et-al-2019-Failing-Loudly/) — Rabanser et al., 2019
- **The idea:** compares ways to detect when incoming data no longer looks like the training data.
- **What we built:** the full detection pipeline with eight ways of summarising data and four statistical tests.
- **What we saw:** like the paper, using the model's own outputs worked well with few samples. Large shifts were easy to catch; small or class-balance shifts were hard.

#### 092 · [Data Distribution Shifts and Monitoring](14-Production-ML/092-Huyen-Data-Distribution-Shifts-and-Monitoring/) — Chip Huyen, 2022 (blog post)
- **The idea:** the different kinds of data shift, how to detect them, and how to respond.
- **What we built:** each kind of shift, monitoring with sensible time windows, and retraining strategies.
- **What we saw:** the most damaging kind ("concept drift", where the right answer changes) was **invisible** to input monitoring; only checking accuracy against real outcomes revealed it.

#### 093 · [Model Cards for Model Reporting](14-Production-ML/093-Mitchell-et-al-2019-Model-Cards/) — Mitchell et al., 2019
- **The idea:** ship every model with a short report card, including how well it works **for different groups of people**, not just on average.
- **What we built:** a model card generator and the paper's two worked examples (smile detection and toxicity scoring).
- **What we saw:** the averages looked fine while specific groups had much higher error rates, which only the group-by-group breakdown revealed.

#### 094 · [Designing Machine Learning Systems](14-Production-ML/094-Huyen-2022-Designing-ML-Systems/) — Chip Huyen, 2022 (book)
- **The idea:** the whole life of an ML system: data, labels, features, evaluation, deployment and testing in production. (The book is paid, so this is built from its public table of contents, without quoting it.)
- **What we built:** a small experiment for each major technique.
- **What we saw:** "data leakage" made a model look 84% accurate on pure noise, an "invariance test" (does changing only a protected attribute change the decision?) caught a biased loan model that flipped 15.9% of decisions, and updating a model daily was 67 times cheaper than retraining from scratch, for the same quality.

---

### Stage 15 · Reinforcement learning *(optional)*

**The story.** Reinforcement learning teaches by reward rather than by example: try something, see how well it went, do more of what worked. These papers show it at large scale: a simple "evolution" method, the team that beat world champions at Dota 2, and AlphaGo, the first program to beat a professional Go player.

#### 095 · [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](15-Reinforcement-Learning-optional/095-Salimans-et-al-2017-Evolution-Strategies/) — Salimans et al., 2017
- **The idea:** nudge all the weights randomly, keep the nudges that score better, repeat. It's simple and easy to spread over thousands of computers.
- **What we built:** the method, its many-computer version, and a balancing-pole game to test it on.
- **What we saw:** it solved the game, and the noise in its learning signal stayed steady as tasks got longer, while the usual RL method's noise grew a lot. The paper predicts exactly that.

#### 096 · [Dota 2 with Large Scale Deep Reinforcement Learning (OpenAI Five)](15-Reinforcement-Learning-optional/096-Berner-et-al-2019-Dota2-OpenAI-Five/) — Berner et al., 2019
- **The idea:** a team of AIs beat the world champions at the video game Dota 2 after 10 months of training, using "surgery" to change the model without starting over.
- **What we built:** model surgery, the training method (PPO) on a small game, and the paper's data-quality experiments.
- **What we saw:** surgery kept learned skill when the model was changed, and using out-of-date data slowed learning the most, as the paper found.

#### 097 · [Mastering the Game of Go with Deep Neural Networks and Tree Search (AlphaGo)](15-Reinforcement-Learning-optional/097-Silver-et-al-2016-AlphaGo/) — Silver et al., 2016
- **The idea:** **AlphaGo** combined networks that suggest moves and judge positions with a look-ahead search.
- **What we built:** the whole AlphaGo pipeline on a small board game that can be solved perfectly, so every part can be checked against perfect play.
- **What we saw:** adding search raised the share of perfect moves from 86% to about 99%. **Honest differences:** on this small game the simple fast policy beat the network, and self-play training helped less than in the paper.

---

### Stage 16 · Extras *(optional)*

**The story.** A mix of interesting side-roads: learning with labels in a contrastive way, a lesson from early feature learning, two early papers on representing relationships, and two papers about AI's impact on society, covering security risks and jobs.

#### 098 · [Supervised Contrastive Learning](16-Extras-optional/098-Khosla-et-al-2020-Supervised-Contrastive-Learning/) — Khosla et al., 2020
- **The idea:** pull together the representations of all images of the same class and push apart those of different classes.
- **What we built:** the paper's losses and its two-stage training, compared with ordinary training.
- **What we saw:** it was much more robust to noise and blur, but less robust to small shifts. With careful tuning, ordinary training tied it.

#### 099 · [The Importance of Encoding Versus Training with Sparse Coding and Vector Quantization](16-Extras-optional/099-Coates-Ng-2011-Encoding-vs-Training-Sparse-Coding/) — Coates & Ng, 2011
- **The idea:** in early feature learning, **how you use** the learned patterns matters more than how carefully you learn them.
- **What we built:** every combination of five ways to learn the patterns and four ways to use them.
- **What we saw:** exactly the paper's finding. With a good way of using them, even randomly chosen patterns worked as well as carefully learned ones.

#### 100 · [Using Matrices to Model Symbolic Relationships](16-Extras-optional/100-Sutskever-Hinton-2008-Matrices-Symbolic-Relationships/) — Sutskever & Hinton, 2008
- **The idea:** represent both things and relationships as small matrices, so relationships can be combined and even learned from definitions, like learning "+3" from "3 plus".
- **What we built:** the model on clock arithmetic and on family trees.
- **What we saw:** results close to the paper's tables, including learning a relationship it was never shown directly.

#### 101 · [Modelling Relational Data using Bayesian Clustered Tensor Factorization](16-Extras-optional/101-Sutskever-Salakhutdinov-Tenenbaum-2009-Bayesian-Clustered-Tensor-Factorization/) — Sutskever, Salakhutdinov & Tenenbaum, 2009
- **The idea:** predict missing facts in a database while also grouping similar things, using careful probability reasoning.
- **What we built:** the model and its comparisons, on made-up data with hidden groups.
- **What we saw:** with very little data the simple method failed badly while the careful probabilistic one kept working, and it found the hidden groups when there was enough data.

#### 102 · [The Malicious Use of Artificial Intelligence](16-Extras-optional/102-Brundage-et-al-2018-Malicious-Use-of-AI/) — Brundage et al., 2018
- **The idea:** a policy report on how AI could be misused, and what researchers and governments should do about it.
- **What we built:** a simple model of how cheaper automation expands attacks, and a "red team" check of a classifier. This is defensive only, with no attack tools.
- **What we saw:** a targeted data-poisoning attack fooled the model and slipped past a common cleaning defence, but a simple check on 100 trusted examples caught it.

#### 103 · [Automation and New Tasks: How Technology Displaces and Reinstates Labor](16-Extras-optional/103-Acemoglu-Restrepo-2019-Automation-and-New-Tasks/) — Acemoglu & Restrepo, 2019
- **The idea:** economics of AI and jobs. **Automation** takes tasks from workers and lowers labour's share of income. **New tasks** create work and raise it. US wages grew slowly after 1987 because automation sped up while new tasks slowed.
- **What we built:** the paper's model of tasks and its method for measuring these effects, tested on data where we know the true answer.
- **What we saw:** automation always lowered labour's share, but whether wages fell depended on how much better the machines were ("so-so automation" lowers wages). The measurement method underestimated both effects, as the paper itself warns.

---

## 7. How honest are the results?

This project tries hard not to oversell. Here is what you should know:

- **Everything runs at small scale.** The original papers used huge models, real datasets and many computers. Here, models are tiny and data is often made up, so the *idea* can be seen in seconds on a laptop. Our numbers are not meant to match the paper's final scores; the paper's own numbers are always quoted next to ours for comparison.
- **Big experiments were written but not run.** From paper 007 on, each `experiments.py` contains the larger experiments. They were not run while building this repository, and each `EXPLAINED.md` states which checks *were* run.
- **Paper claims were checked against the actual paper.** Numbers attributed to a paper were verified by searching its text, not quoted from memory.
- **Where our results disagreed with the paper, we say so.** Some examples you will find in the explanations:
  - paper 006's own step-size formula is slightly wrong (2.38 should be 2.0);
  - paper 033's stated receptive field (315) doesn't match its formula (373);
  - DPO did not beat RLHF on our toy (069);
  - reinforcement learning did not beat supervised training in our InstructGPT toy (067);
  - AlphaGo's mixed evaluation was not the best variant on our small game (097);
  - supervised contrastive learning tied a well-tuned ordinary model (098).
- **Tests prove correctness of the pieces, not the paper's conclusions.** A passing test means, for example, "this gradient is computed correctly" or "this matches the paper's table". The demos and explanations are where the paper's *claims* are examined.

---

## 8. Other files in this repository

| File | What it contains |
|---|---|
| [READING_ORDER.md](READING_ORDER.md) | The full learning path: all 103 papers in order, why each stage comes where it does, a difficulty rating for each paper, and a three-pass method for reading any paper. |
| [TECHNICAL_SUMMARY.md](TECHNICAL_SUMMARY.md) | The detailed technical version of this page: exactly which equations, figures and tables were implemented for each paper, and the precise numbers each demo prints. |
| [SOURCES.txt](SOURCES.txt) | Where each paper's PDF was downloaded from. The PDFs themselves are **not** stored on GitHub (they are copyrighted), so download them from these links if you want to read the originals. |

**Requirements:** Python 3.10 or newer. Paper 001 needs nothing else. Later papers use `numpy`, some use `scipy`, `scikit-learn` or `torch`/`torchvision`, figures use `matplotlib`, and the tests need `pytest`.
