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

Under each paper there is also a **"Read the detailed explanation"** button. Click it to open a longer, step-by-step explanation of every important topic in that paper: what problem existed before it, each key idea explained one at a time (with everyday analogies and small worked examples with real numbers), what our code showed, and why the paper still matters. You can read only the short summaries first and open the details whenever you want to go deeper.

Click a paper's title to open its folder, where `EXPLAINED.md` goes deeper still (full formulas, the paper's own tables, and a quiz).

---

### Stage 01 · Foundations: what is a neural network?

**The story.** In the 1940s, scientists wondered whether thinking could be explained by simple on/off brain cells wired together. This stage follows that question: first a neuron as a tiny logic switch, then a network that can *learn*, then the discovery that a single layer has serious limits, and finally the method (backpropagation) that lets networks with many layers learn. Everything later in this repository is built on these ideas.

#### 001 · [A Logical Calculus of the Ideas Immanent in Nervous Activity](01-Foundations/001-McCulloch-Pitts-1943/) — McCulloch & Pitts, 1943
- **The idea:** a brain cell can be treated as a tiny switch that "fires" when enough of its inputs are on. Wire enough of these switches together and they can compute any logical rule. This was the first mathematical model of a neuron.
- **What we built:** a simulator for these networks, the examples from the paper's first figure, networks that remember things using loops, and a "compiler" that turns a written logic rule into a working network.
- **What we saw:** every example in the paper works, including the famous illusion where touching something cold for a moment can feel hot.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before this paper.** In 1943 nobody knew whether the brain's behaviour could be described with mathematics. Biologists knew that nerve cells (neurons) send electrical spikes, and that a spike is "all or nothing": a neuron either fires fully or not at all. Warren McCulloch (a neuroscientist) and Walter Pitts (a young logician) asked a bold question: if neurons are on/off devices, is the brain a kind of logic machine?

**Topic 1: the neuron as a switch.** Their model neuron has several input wires and one output. Time moves in fixed ticks. At each tick:
- every **excitatory** input that is on counts as +1;
- the neuron fires if the count reaches its **threshold** (say 2);
- if any **inhibitory** input is on, the neuron is blocked completely, whatever the count.

There are no learned weights here; the wiring and thresholds are fixed by design.

**Topic 2: neurons are logic gates.** With the right threshold, one neuron becomes a basic logic gate:

| Gate | How to build it | Example |
|---|---|---|
| AND | two inputs, threshold 2 | fires only if both are on |
| OR | two inputs, threshold 1 | fires if either is on |
| NOT | an always-on input plus an inhibitory input | fires unless the inhibitor is on |

Since any logical rule can be built from AND, OR and NOT, **any logical rule can be built from these neurons**.

**Topic 3: delays and time.** A signal takes one tick to pass through a neuron. So a chain of neurons can say things about the past, like "input A was on two ticks ago". The network's output depends on *when* things happened, not just *what* happened.

**Topic 4: loops give memory.** If a neuron's output is wired back to its own input (a "circle"), once it fires it keeps firing forever. That is one bit of memory. The paper showed that networks with loops can remember, and that with an external tape they can compute anything a Turing machine (an ideal computer) can.

**Topic 5: the heat illusion.** A real puzzle: if a very cold object touches your skin for a moment, you can feel *heat*; if it stays, you feel cold. The paper builds a tiny network with a delay that produces exactly this: a brief "cold" signal triggers the "heat" output, while a longer one triggers "cold". It shows that a perception is the result of the network's computation, not a direct copy of the world.

**Why it matters.** This was the first mathematical model of a neuron, and it introduced the idea that thinking might be computation. What was missing was **learning**: every weight and threshold had to be chosen by hand. The next paper fixes that.

</details>

#### 002 · [The Perceptron](01-Foundations/002-Rosenblatt-1958/) — Rosenblatt, 1958
- **The idea:** the first network that **learns from examples**. It is shown pictures, guesses a category, and strengthens or weakens its connections depending on whether it was right.
- **What we built:** the paper's full "photo-perceptron" (a model eye connected to randomly wired units), its learning rules, and the probability formulas that predict how well it will do.
- **What we saw:** the formulas match the simulation, and we reproduced the paper's figures showing the difference between memorising the examples it saw and recognising new ones.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before this paper.** McCulloch–Pitts neurons could compute, but a person had to design every connection. Frank Rosenblatt wanted a machine that **learns** to recognise patterns from examples, the way animals learn.

**Topic 1: the three layers of the perceptron.**
- **S-units (sensory):** a grid of light sensors, like a simple retina. Each is on or off.
- **A-units (association):** each A-unit is wired to a **random** handful of S-units. It fires if enough of its inputs are on. Rosenblatt deliberately used random wiring, believing the brain's early wiring was largely random too.
- **R-units (response):** the output, for example "this is a square" vs "this is a circle". Each R-unit adds up the A-units connected to it, each with a strength (a **weight**).

**Topic 2: learning by reinforcement.** Learning only changes the weights from A-units to R-units:
- show a picture;
- the perceptron answers;
- if the answer was wrong, the weights of the A-units that were active are increased or decreased so the answer moves toward the right one.

In modern notation the rule is:
```
new weight = old weight + learning rate × (correct answer − given answer) × input
```
**Worked example (learning AND).** Start with weights w1 = w2 = 0, bias b = 0, learning rate 1, and answer 1 if w1·x1 + w2·x2 + b > 0, otherwise 0.
- Input (1,1), target 1. The sum is 0, so the answer is 0: wrong by +1. The update gives w1 = 1, w2 = 1, b = 1.
- Input (0,0), target 0. The sum is 1, so the answer is 1: wrong by −1. The update gives b = 0.
- Input (1,0), target 0. The sum is 1, so the answer is 1: wrong. The update gives w1 = 0, b = −1.

Keep cycling through the examples and it settles on weights that compute AND correctly. Nobody designed those weights; the machine found them.

**Topic 3: memorising vs generalising.** Rosenblatt carefully separated two abilities:
- **P_r:** the chance of answering correctly on pictures it was **trained on**;
- **P_g:** the chance of answering correctly on **new** pictures of the same kind.

He derived probability formulas for both, depending on how many A-units there are and how they are wired. This distinction (training accuracy vs test accuracy) is still the most important idea in machine learning evaluation.

**Topic 4: different learning systems.** The paper compares variants, such as changing every active unit by the same fixed amount, or keeping the total weight constant. They differ in how fast and how stably they learn.

**Why it matters.** This was the first machine that learned from data. It caused great excitement (newspapers claimed it would soon walk, talk and be conscious). Its limits were not yet understood; that is the next paper.

</details>

#### 003 · [Perceptrons (Introduction)](01-Foundations/003-Minsky-Papert-1969/) — Minsky & Papert, 1969
- **The idea:** a careful proof of what a single-layer perceptron **cannot** do. Some simple-sounding questions, like "is this shape all in one connected piece?", are impossible for it. This book is often blamed for slowing AI research for years.
- **What we built:** the formal perceptron, a perceptron that checks whether a shape is convex, and computer-checked versions of the book's proofs.
- **What we saw:** the proofs hold. We also measured, as an extra, how complicated a perceptron must be to compute "parity" (is the number of on-pixels odd?).

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before this paper.** Perceptrons were hyped as general learning machines. Marvin Minsky and Seymour Papert set out to prove mathematically what they could and **could not** learn.

**Topic 1: what a perceptron really computes.** A single-layer perceptron first computes many small **local features** (each looks at only a few pixels), then makes its decision from a **weighted vote**: "yes" if the weighted sum is above a threshold. Geometrically, a weighted vote draws one straight line (or flat plane) through the space of feature values and says "yes" on one side.

**Topic 2: the XOR problem.** XOR ("exclusive or") answers yes for (0,1) and (1,0) and no for (0,0) and (1,1). Plot these four points on a square: the "yes" points are on one diagonal and the "no" points on the other. **No single straight line** separates the two diagonals. So a perceptron that sees the two inputs directly cannot compute XOR.

**Topic 3: the "order" of a perceptron.** The *order* is the largest number of pixels any single feature looks at. Small order means every feature is local. The authors proved:
- **Convexity** ("is this shape free of dents?") has order 3. A shape is convex if, for every pair of its points, the midpoint is also inside it, and each such check only involves 3 pixels.
- **Parity** ("is the number of on-pixels odd?") needs order equal to the **whole image**: some feature must look at every pixel at once. Changing any single pixel flips the answer, so no local feature can help.
- **Connectedness** ("is the shape one piece?") cannot be computed with any fixed small order as the image grows. Whether two far-apart regions are joined can depend on a thin path anywhere in the picture.

**Topic 4: group invariance.** If a question doesn't change when the picture is shifted or rotated, then one can average the perceptron's features over all these transformations without changing what it can compute. This trick makes the impossibility proofs possible.

**Why it matters.** The book showed that one layer of learned weights is fundamentally limited. Many people took this as proof that neural networks were a dead end, and funding moved elsewhere (part of the first "AI winter"). The authors knew that multi-layer networks could escape these limits, but nobody yet knew how to *train* them. Backpropagation (next paper) solved that.

</details>

#### 004 · [Learning Representations by Back-Propagating Errors](01-Foundations/004-Rumelhart-Hinton-Williams-1986/) — Rumelhart, Hinton & Williams, 1986
- **The idea:** **backpropagation**, the method that finally let networks with hidden layers learn. The error at the output is passed backwards, layer by layer, to work out how every weight should change. Almost every network since uses it.
- **What we built:** backpropagation for any layered network (checked against a slow numerical method), plus its version for networks that run over time.
- **What we saw:** we reproduced the paper's mirror-symmetry network (with the same neat 1 : 2 : 4 pattern of weights) and its family-tree network, which invents its own features like nationality and generation. **Surprise:** following the paper's exact recipe, the family-tree network often gets stuck, because the learning signal fades as it travels back through the layers.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before this paper.** Networks with **hidden layers** (layers between input and output) could in principle compute things like XOR and parity. But how should a hidden unit's weights change? The training data only says what the *output* should be. Nobody tells a hidden unit what it should do. This is the "credit assignment problem".

**Topic 1: smooth neurons.** Instead of a hard on/off threshold, each unit uses the smooth **sigmoid** function, σ(x) = 1/(1+e^(−x)). It goes gently from 0 to 1. Because it is smooth, a tiny change in a weight causes a tiny change in the output, so we can ask "which direction should each weight move to reduce the error?"

**Topic 2: the error.** Measure how wrong the network is with E = ½ Σ (output − target)². Training means making E small.

**Topic 3: the chain rule, run backwards.** The effect of a weight on the error passes through a chain of steps (weight → unit's input → unit's output → next unit → … → error). Calculus's **chain rule** says the total effect is the product of the effects of each step. Backpropagation computes these products efficiently:
1. Run the network forwards and remember every unit's output.
2. At the output, compute each unit's "error signal" δ.
3. Pass δ backwards: a hidden unit's δ is the weighted sum of the δs of the units it feeds, times the slope of its own sigmoid.
4. The gradient for a weight is (δ of the unit it feeds into) × (output of the unit it comes from).

**Worked example (one input, one hidden unit, one output, target 1).** Take x = 1, w1 = 0.5, w2 = 1.0.
- Forward pass:
  - hidden h = σ(0.5) = 0.622;
  - output y = σ(1.0 × 0.622) = 0.651;
  - error E = ½(1 − 0.651)² = 0.061.
- Output signal: δ2 = (y − target) × y(1 − y) = (−0.349)(0.227) = −0.079.
- Gradient for w2 = δ2 × h = −0.049.
- Hidden signal: δ1 = δ2 × w2 × h(1 − h) = −0.079 × 1 × 0.235 = −0.019.
- Gradient for w1 = δ1 × x = −0.019.
- Step with learning rate 1: w2 becomes 1.049 and w1 becomes 0.519, and the output rises toward 1.

**Topic 4: momentum and random starts.** The paper adds **momentum** (each change also includes a fraction of the previous change) to speed learning up. Weights must start **random**: if all start equal, all hidden units receive identical updates and stay identical forever.

**Topic 5: hidden units invent features.** The paper's showpieces:
- **Symmetry detection:** a network that says whether a string of bits is a mirror image. Its hidden units learn weights in a neat 1 : 2 : 4 pattern, of opposite sign on the two halves, so a symmetric input cancels to zero.
- **Family trees:** a network told only facts like "Colin's father is James" discovers hidden features meaning *nationality*, *generation* and *which branch of the family*, without ever being told these concepts exist.

**What we found.** The family-tree network often got stuck when we followed the paper's recipe exactly. The error signal shrinks a little at every layer it passes back through (each sigmoid slope is at most 0.25). This is the **vanishing gradient** problem, and Stage 02 is largely about fixing it.

**Why it matters.** Backpropagation is how essentially every neural network today is trained, from image recognisers to ChatGPT.

</details>

#### 005 · [Perceptron: Learning, Generalization, Model Selection, Fault Tolerance, and Role in the Deep Learning Era](01-Foundations/005-Du-et-al-2022/) — Du, Leung, Mow & Swamy, 2022
- **The idea:** a modern review of 70 years of perceptron research, tying Stage 01 together.
- **What we built:** the classic learning rules, a multi-layer network with backpropagation, 11 different training methods, and the usual tricks for avoiding overfitting.
- **What we saw:** we reproduced the review's experiment on the classic Iris flower dataset.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**What this paper is.** A 2022 review that walks through 70 years of perceptron research and connects it to deep learning. It is useful as a summary of Stage 01 and a bridge to Stage 02.

**Topic 1: the perceptron convergence theorem.** If the two classes *can* be separated by a straight line (or plane), the perceptron learning rule is **guaranteed** to find such a line after a limited number of mistakes. The bound is (R/γ)², where:
- R is the size of the largest input;
- γ (the **margin**) is how much empty space there is between the two classes.

Wide gaps mean fast learning. If the classes cannot be separated, the plain rule never settles; variants like the **pocket algorithm** keep the best weights seen so far.

**Topic 2: from one layer to many.** A **multi-layer perceptron (MLP)** stacks layers of units with smooth activations and trains them with backpropagation. The **universal approximation theorem** says that one hidden layer with enough units can approximate any reasonable function as closely as you like, though it may need very many units, and the theorem says nothing about how to find the weights.

**Topic 3: training methods.** The review compares many ways to adjust weights:
- plain gradient descent;
- **momentum**;
- adaptive step sizes;
- second-order methods (which also use the *curvature* of the error, like knowing how steep the slope is getting);
- quasi-Newton (BFGS) and Levenberg–Marquardt.

Faster methods use more memory or computation per step.

**Topic 4: generalisation and model selection.** A big network can memorise its training data (**overfitting**). Standard defences:
- **hold out validation data** and stop training when validation error starts rising (**early stopping**);
- **weight decay** (penalise large weights);
- **cross-validation** (rotate which part of the data is held out) to choose the network size.

**Topic 5: fault tolerance.** Because knowledge is spread over many weights, a network often keeps working if a few weights are damaged or noisy, unlike a normal program where one flipped bit can crash everything. The review discusses training methods that make this robustness stronger.

**Topic 6: role in deep learning.** Every modern network, transformers included, is made of perceptron-like units: a weighted sum followed by a non-linear function. The ideas of this stage are still at the core.

**Our experiment.** On the classic Iris flowers dataset (150 flowers, 3 species, 4 measurements), we reproduced the review's comparison of learning rules and network sizes.

</details>

---

### Stage 02 · Training deep networks: making learning actually work

**The story.** Backpropagation works on paper, but in practice deep networks trained badly: learning was slow, signals faded or exploded, and models memorised instead of generalising. This stage collects the practical fixes that made deep learning work: preparing the data, starting the weights at sensible values, smarter ways of taking steps downhill, randomly switching off units during training, and keeping each layer's numbers in a healthy range.

#### 006 · [Efficient BackProp](02-Training-Deep-Networks/006-LeCun-et-al-1998-Efficient-BackProp/) — LeCun, Bottou, Orr & Müller, 1998
- **The idea:** a practical handbook of tricks: scale the inputs, choose a good activation function, start the weights carefully, pick the step size wisely.
- **What we built:** every trick, each tested on handwritten digits, plus tools that measure the "shape" of the loss to choose step sizes automatically.
- **What we saw:** the tricks help as described. **Surprise:** the paper's own formula for the largest safe step size is slightly wrong (it gives 2.38; the correct value is 2.0), because it forgets one detail.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before this paper.** By the 1990s backpropagation worked, but practitioners found it slow and temperamental. Yann LeCun and colleagues collected the practical knowledge into one handbook.

**Topic 1: stochastic vs batch learning.**
- **Batch learning** computes the gradient over the whole dataset before each step.
- **Stochastic learning** steps after **each example** (or a small "mini-batch").

Stochastic learning is usually much faster on large, redundant datasets: if the data contains 10 copies of everything, batch learning wastes 10× the work, while stochastic learning doesn't. Its noise can even help escape poor solutions.

**Topic 2: shuffle and normalise the inputs.**
- **Shuffle** so consecutive examples come from different classes.
- **Centre each input** to mean 0. If all inputs are positive, all of a unit's weights must move in the same direction (all up or all down) at each step, which forces a slow zig-zag path.
- **Scale each input** to similar spread, so no single input dominates.
- If possible, **decorrelate** the inputs (for example with PCA), so they don't carry the same information twice.

**Topic 3: a good activation function.** Use a function symmetric around zero, such as tanh, rather than the logistic sigmoid (which is always positive, causing the same zig-zag problem one layer up). The paper recommends f(x) = 1.7159 · tanh(2x/3). This is chosen so that f(±1) = ±1 and the outputs keep a variance near 1.

**Topic 4: start the weights sensibly.** Draw initial weights randomly with standard deviation 1/√m, where m is the number of inputs to the unit. Then the sum entering each unit has a spread near 1, which is neither tiny (no learning) nor huge (saturated, flat activation).

**Topic 5: choosing the step size.** Near a minimum, the error looks like a bowl. In one dimension, E = ½·a·w², where a is the bowl's **curvature**:
- step size η = 1/a jumps straight to the bottom in one step;
- η between 1/a and 2/a overshoots but still converges (zig-zagging);
- η > 2/a overshoots more each time and **diverges**.

Example with a = 2: η = 0.5 is perfect, η = 0.9 zig-zags in, η = 1.2 blows up. In many dimensions, the steepest direction (the largest curvature, λ_max, the biggest eigenvalue of the **Hessian**) sets the limit 2/λ_max. The paper shows how to *estimate* λ_max cheaply by repeatedly pushing a random vector through the curvature (power iteration), without computing the whole Hessian.

**What we found.** The paper's own formula for the largest safe step in one example gives 2.38. The correct limit is 2.0, because the formula leaves out one term.

**Why it matters.** Almost every tip in this paper (normalise inputs, careful initialisation, mini-batches, step size from curvature) is still standard practice.

</details>

#### 007 · [Understanding the Difficulty of Training Deep Feedforward Neural Networks](02-Training-Deep-Networks/007-Glorot-Bengio-2010-Difficulty-Training-Deep-FF/) — Glorot & Bengio, 2010
- **The idea:** explains *why* deep networks got stuck: with badly chosen starting weights, signals shrink or blow up as they pass through layers. Proposes **"Xavier" initialisation**, a simple formula for starting weights that keeps signals steady.
- **What we built:** both ways of starting weights, several activation functions, and tools that watch the signals in every layer.
- **What we saw:** our measurement of how well signals survive (0.49 with the old method vs 0.80 with Xavier) matches the paper's 0.5 vs 0.8.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before this paper.** Around 2006, deep networks could only be trained with special "pre-training" tricks. Xavier Glorot and Yoshua Bengio asked *why* plain backpropagation failed on deep networks, by watching what happens inside them during training.

**Topic 1: watching activations.** They plotted the distribution of each layer's outputs during training. With the logistic sigmoid, the **top hidden layer saturated** at 0 almost immediately, and stayed stuck there for a long time. Saturated units have nearly zero slope, so almost no learning signal passes through them.

**Topic 2: signals shrink or grow layer by layer.** Think of a layer as multiplying the signal's spread by some factor. With n inputs, each weight of variance Var(w):
```
Var(output) ≈ n · Var(w) · Var(input)
```
- If n·Var(w) = 1/3, each layer shrinks the variance by 3. After 5 layers that is (1/3)⁵ ≈ 0.004, so the signal has almost vanished.
- If n·Var(w) = 3, it explodes instead.

The same happens to gradients travelling backwards, but there the count that matters is the number of **outputs** of each layer.

**Topic 3: the popular initialisation was wrong.** The common recipe drew weights uniformly from [−1/√n, 1/√n]. That gives Var(w) = 1/(3n), so n·Var(w) = 1/3, which is exactly the shrinking case above.

**Topic 4: Xavier (Glorot) initialisation.**
- For forward signals to keep their size, we want n_in · Var(w) = 1.
- For backward gradients to keep theirs, we want n_out · Var(w) = 1.
- Both can't hold unless n_in = n_out, so take the compromise:
```
Var(w) = 2 / (n_in + n_out)      →   uniform on  [ −√(6/(n_in+n_out)),  +√(6/(n_in+n_out)) ]
```
(A uniform distribution on [−a, a] has variance a²/3, which is where the 6 comes from.)

**Topic 5: activation functions compared.** The paper compares:
- **sigmoid**: bad, because it saturates and is not centred at zero;
- **tanh**: better;
- **softsign**, x/(1+|x|): saturates more gently.

**What we found.** Our measurement of how well signals survive through the layers was 0.49 with the old initialisation and 0.80 with Xavier, matching the paper's roughly 0.5 vs 0.8.

**Why it matters.** "Xavier init" (and He init, its later version for ReLU) is used by default in every deep learning library. Thinking about how the **variance** of signals evolves through layers is a key tool for designing networks.

</details>

#### 008 · [On the Importance of Initialization and Momentum in Deep Learning](02-Training-Deep-Networks/008-Sutskever-et-al-2013-Initialization-and-Momentum/) — Sutskever, Martens, Dahl & Hinton, 2013
- **The idea:** **momentum** means letting each training step keep some of the previous step's direction, like a ball rolling downhill. Combined with good starting weights, it lets plain training handle problems thought to need much fancier methods.
- **What we built:** classical momentum and "Nesterov" momentum, the paper's schedule, and its special ways of starting weights.
- **What we saw:** the paper's theorem checks out exactly. **Surprise:** in some settings Nesterov momentum can blow up where the classic version still works.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before this paper.** Deep networks and recurrent networks were believed to need advanced second-order optimisers (like "Hessian-free" optimisation). Ilya Sutskever and colleagues showed that ordinary gradient descent works well if you add two things: good starting weights and well-tuned **momentum**.

**Topic 1: momentum, the rolling ball.** Plain gradient descent takes a step down the slope each time. Momentum keeps a **velocity** that remembers previous steps:
```
v ← μ·v − η·gradient        θ ← θ + v
```
μ (for example 0.9) is how much of the old velocity is kept. On a long gentle slope, the steps build up. With constant gradient g, the velocity approaches η·g/(1 − μ), which is 10 times a single step when μ = 0.9. In a narrow valley, side-to-side wiggles cancel out while the motion along the valley adds up.

**Topic 2: Nesterov momentum, look before you leap.** Classical momentum measures the gradient where you *are*. **Nesterov's** version first moves by the momentum and measures the gradient where you are *about to be*:
```
v ← μ·v − η·gradient(θ + μ·v)
```
If the ball is about to overshoot, Nesterov sees the slope turning upward and brakes earlier. This makes high momentum (0.99) usable.

**Topic 3: a momentum schedule.** Start with low momentum (when the landscape is rough and the direction keeps changing), then raise it toward 0.99 or 0.995. Lower it again near the end for fine settling.

**Topic 4: initialisation matters as much.** The paper uses **sparse initialisation**: each unit gets only about 15 non-zero incoming weights. That keeps signals from being too small without making all units alike. For recurrent networks, the recurrent weights are scaled so signals neither die out nor explode over time (an idea borrowed from "echo state networks").

**Topic 5: the results.** With these choices, plain momentum methods trained deep autoencoders and recurrent networks on hard long-memory problems about as well as Hessian-free optimisation.

**What we found.** The paper's theorem connecting the two forms of momentum checks out exactly. We also found settings where Nesterov momentum diverges while the classical version still works, a reminder that "usually better" is not "always better".

**Why it matters.** Momentum (in Nesterov or classical form) is in every modern optimiser, including Adam (paper 011). The lesson that "simple methods plus careful details" beat complicated methods appears again and again in deep learning.

</details>

#### 009 · [Improving Neural Networks by Preventing Co-adaptation of Feature Detectors](02-Training-Deep-Networks/009-Hinton-et-al-2012-Preventing-Co-adaptation/) — Hinton et al., 2012
- **The idea:** the first version of **dropout**: during training, randomly switch off half the units each time, so no unit can rely too much on any other. This makes the network more robust.
- **What we built:** dropout, the weight limit the paper uses, and the "average network" used at test time.
- **What we saw:** by trying all 1,024 possible smaller networks, we proved the test-time network is *exactly* their (geometric) average, as the paper claims.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before this paper.** Big networks trained on small datasets overfit: they memorise the training examples. Geoffrey Hinton and colleagues proposed a surprisingly simple fix.

**Topic 1: co-adaptation.** In a trained network, hidden units can become dependent on each other in fragile ways: unit A only works because unit B fixes A's mistakes on particular training examples. These partnerships fit the training set perfectly but don't generalise.

**Topic 2: dropout.** On every training example, **switch off each hidden unit at random with probability 0.5**. The network has to work with any random half of its units, so each unit must be useful on its own and cannot rely on specific partners. Input units can also be dropped, usually with a lower probability like 0.2.

**Topic 3: a huge ensemble for free.** With n hidden units there are 2ⁿ ways to drop units, so 2ⁿ different "thinned" networks. With n = 10 that is 1,024 networks. All share the same weights. Training with dropout trains a different thinned network on every example. Averaging many models (an **ensemble**) usually beats any single one, but running 2ⁿ networks at test time is impossible.

**Topic 4: the test-time shortcut.** At test time, use all units but **halve their outgoing weights**. During training a unit was present half the time, so this keeps the expected input to the next layer the same. For a network with one hidden layer and a softmax output, this is **exactly** the normalised geometric mean of all 2ⁿ thinned networks' predictions.

**Topic 5: max-norm.** The paper also limits each unit's incoming weight vector to a maximum length. When a weight vector grows too long, it is scaled back. This allows large learning rates without blowing up.

**What we found.** With 10 hidden units we actually ran all 1,024 sub-networks and computed their geometric average. It equalled the halved-weight network to rounding error, confirming the claim.

**Why it matters.** Dropout became one of the most used regularisers in deep learning and is still used inside many transformers.

</details>

#### 010 · [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](02-Training-Deep-Networks/010-Srivastava-et-al-2014-Dropout/) — Srivastava et al., 2014
- **The idea:** the full study of dropout: why it works, how to tune it, and variants of it.
- **What we built:** dropout networks, a version that uses random noise instead of on/off switches, and the paper's comparison with other methods.
- **What we saw:** we confirmed the paper's mathematical result that, for a simple model, dropout is the same as a well-known penalty on large weights.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**What this paper is.** The full journal version of dropout. It explains dropout far more thoroughly, tests it on many tasks, and analyses why it works.

**Topic 1: the notation.** Each unit is kept with probability p (so p = 0.5 means half are dropped). For each unit, a random mask r ~ Bernoulli(p) is drawn, and its output y becomes r·y. At test time, weights are multiplied by p. Modern code usually uses "**inverted dropout**" instead: it divides by p during training, so nothing changes at test time.

**Topic 2: choosing p.** Typical choices:
- p = 0.5 for hidden units;
- p = 0.8 for input units (dropping too many inputs destroys information).

Dropout networks should be larger than normal, because fewer units are active at once; roughly n/p units to match n normal ones.

**Topic 3: dropout as noise.** Multiplying by a random 0/1 mask is one kind of noise. The paper also tests **Gaussian dropout**: multiply each unit by a random number from a bell curve with mean 1. It works about as well, which shows that the key ingredient is *noise* that makes units robust.

**Topic 4: dropout for linear regression is weight decay.** For the simplest model (predict y from inputs with a linear function), averaging the dropout loss over all random masks gives exactly:
```
ordinary squared error (with weights scaled by p)  +  p(1−p) × Σ_j (w_j² × Σ_examples x_j²)
```
That is a **penalty on large weights** (like ridge regression). Weights on inputs with large values are penalised more. So in this simple case, dropout is a data-aware version of weight decay.

**Topic 5: effect on features.** The learned features are cleaner and less tangled: on handwritten digits, hidden units trained with dropout look like crisp strokes and spots, while ordinary training gives noisy patterns. Activations are also **sparser** (fewer units strongly on at once).

**Topic 6: wide testing.** The paper shows gains on images (MNIST, CIFAR, ImageNet), speech, text and biology data.

**What we found.** We confirmed the linear-regression equivalence numerically: the average dropout loss matched the penalised formula.

**Why it matters.** This is the reference paper on dropout and a model of how to analyse a simple trick thoroughly.

</details>

#### 011 · [Adam: A Method for Stochastic Optimization](02-Training-Deep-Networks/011-Kingma-Ba-2015-Adam/) — Kingma & Ba, 2015
- **The idea:** **Adam**, the most widely used training method today. It adapts the step size for every weight separately, based on how that weight's gradient has behaved recently.
- **What we built:** Adam and six related methods from scratch.
- **What we saw:** our Adam gives the same results as PyTorch's official version, and the paper's guarantees (such as the limit on step size) hold.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before this paper.** Gradient descent uses one step size for every weight. But some weights get large, noisy gradients and others tiny, rare ones. Methods like **AdaGrad** and **RMSProp** gave each weight its own step size. **Adam** (Diederik Kingma and Jimmy Ba) combined the best of these with momentum.

**Topic 1: two running averages.** For each weight, at every step t with gradient g:
```
m ← β1·m + (1−β1)·g          (average gradient: the "momentum", β1 = 0.9)
v ← β2·v + (1−β2)·g²         (average squared gradient: the "size", β2 = 0.999)
```
- m tells us which direction the gradient usually points.
- v tells us how big gradients usually are.

**Topic 2: bias correction.** m and v start at 0, so in early steps they are too small. Example: after step 1, m = 0.1·g, which is 10× too small. Adam divides by the missing fraction:
```
m̂ = m / (1 − β1^t)          v̂ = v / (1 − β2^t)
```
At t = 1, m̂ = 0.1g / 0.1 = g. That is exactly right.

**Topic 3: the update.**
```
θ ← θ − α · m̂ / (√v̂ + ε)          (α = 0.001, ε = 10⁻⁸)
```
**Worked example.** Step 1, gradient g = 2:
- m = 0.2, v = 0.004;
- after bias correction, m̂ = 2 and v̂ = 4;
- step = 0.001 × 2/√4 = 0.001.

If the gradient had been 200 instead, the step would still be 0.001. The step size is about α no matter how large or small the gradients are. Dividing by √v̂ makes Adam **invariant to the scale** of the gradients.

**Topic 4: a natural step-size bound.** Because m̂/√v̂ is roughly at most 1 in size, each step moves a weight by about α at most. You can choose α by thinking "how far should a weight move per step?". When gradients are noisy and disagree, m̂ is small relative to √v̂, so Adam takes smaller, more careful steps, like an automatic annealing.

**Topic 5: relatives.**
- **AdaGrad** sums all past squared gradients, so its step size only ever shrinks.
- **RMSProp** uses a decaying average (like v) but no momentum and no bias correction.
- **AdaMax** (in the same paper) replaces the squared average with a running maximum.

**What we found.** Our from-scratch Adam produced the same numbers as PyTorch's built-in Adam, and the step-size bound held in our tests.

**Why it matters.** Adam (and its variant **AdamW**, with decoupled weight decay) is the default optimiser for almost all deep learning, including large language models.

</details>

#### 012 · [Batch Normalization](02-Training-Deep-Networks/012-Ioffe-Szegedy-2015-Batch-Normalization/) — Ioffe & Szegedy, 2015
- **The idea:** inside the network, keep each layer's numbers centred and of similar size during training. This made training much faster and more stable.
- **What we built:** batch normalisation by hand, including how it learns and how it behaves at test time.
- **What we saw:** our hand-written version matches PyTorch's, and the paper's example of a problem it fixes behaves as described.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before this paper.** Training deep networks needed small learning rates and careful initialisation. Sergey Ioffe and Christian Szegedy argued that part of the problem is **internal covariate shift**: as earlier layers learn, the distribution of inputs to later layers keeps changing, so later layers chase a moving target.

**Topic 1: normalise inside the network.** For each unit, over the current **mini-batch** of examples:
1. compute the mean μ and variance σ² of that unit's values;
2. normalise: x̂ = (x − μ)/√(σ² + ε);
3. rescale with two learned numbers: y = γ·x̂ + β.

**Worked example.** One unit's values over a batch of 4 are [1, 2, 3, 6].
- Mean = 3.
- Variance = (4 + 1 + 0 + 9)/4 = 3.5, so the standard deviation is 1.871.
- Normalised values: [−1.07, −0.53, 0, 1.60].
- With γ = 2 and β = 0.5, the outputs are [−1.64, −0.57, 0.5, 3.71].

**Topic 2: why γ and β?** Forcing every unit to mean 0 and variance 1 could limit what the network can represent. With γ and β the network can undo the normalisation (γ = σ, β = μ) if that is better. Normalisation becomes a convenient starting point, not a cage.

**Topic 3: backpropagating through the batch statistics.** μ and σ depend on every example in the batch, so the gradient for one example depends on the others. The paper derives the full gradient formulas.

**Topic 4: test time.** At test time you might predict on a single example, so there is no batch to average over. Batch norm keeps **running averages** of μ and σ² during training and uses those fixed values at test time. Forgetting to switch between "training mode" and "evaluation mode" is a classic bug.

**Topic 5: benefits.**
- Much **higher learning rates** become safe.
- Training is far less sensitive to initialisation.
- It acts a bit like a regulariser: the batch statistics add noise.
- The paper reached the same ImageNet accuracy as a strong baseline in **14 times fewer steps**.

**A later twist.** Later research (Santurkar et al., 2018) argued that batch norm helps mainly by making the loss landscape smoother, not by reducing covariate shift. The technique works, but the original explanation is debated.

**What we found.** Our hand-written batch norm (forward, backward and running averages) matches PyTorch's built-in layer.

**Why it matters.** Batch norm made very deep image networks practical (ResNet uses it everywhere). Its cousin **layer normalisation**, which averages over a unit's features rather than over the batch, is used in every transformer.

</details>

---

### Stage 03 · Seeing: networks for images

**The story.** Images are huge (millions of numbers) and the same object can appear anywhere in a picture. Convolutional networks solve this by sliding small pattern detectors across the image. This stage follows them from reading handwritten cheques in the 1990s to the 2012 breakthrough on ImageNet that started the modern deep-learning boom, and on to networks hundreds of layers deep.

#### 013 · [Gradient-Based Learning Applied to Document Recognition (LeNet-5)](03-CNNs-and-Vision/013-LeCun-et-al-1998-LeNet5-Gradient-Based-Learning/) — LeCun et al., 1998
- **The idea:** **LeNet-5**, a convolutional network that read handwritten digits on bank cheques. It introduced the building blocks still used today: convolution, pooling and shared weights.
- **What we built:** LeNet-5 exactly as described, down to its unusual connection pattern and exactly 60,000 adjustable numbers.
- **What we saw:** the details match the paper, including why one of its loss choices makes all outputs collapse to the same answer.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before this paper.** Recognising handwriting with an ordinary fully connected network is wasteful. A 32×32 image has 1,024 pixels, and every hidden unit would need 1,024 weights. Worse, the network would have to learn "a stroke" separately for every position in the image. Yann LeCun and colleagues built networks that understand images are made of **local patterns that can appear anywhere**.

**Topic 1: convolution.** A small filter (say 5×5 numbers) slides across the image. At each position, multiply the filter by the patch underneath and add everything up. The result is a new image (a **feature map**) showing *where* the filter's pattern appears.

Tiny example (3×3 image, 2×2 filter that detects a vertical edge):
```
image          filter      output (2×2)
1 0 0          1 -1        1·1+0·(-1)+1·1+0·(-1) = 2    ...
1 0 0          1 -1
1 0 0
```
The output is large where the left side is bright and the right side dark.

**Topic 2: weight sharing.** The **same** filter is used at every position. That gives far fewer weights (25 per filter instead of thousands), and a pattern learned in one place is automatically recognised everywhere.

**Topic 3: subsampling (pooling).** After convolution, average each 2×2 block, halving the map's width and height. The network becomes less sensitive to exact positions: a stroke shifted by one pixel gives nearly the same result.

**Topic 4: the LeNet-5 architecture (input 32×32).**

| Layer | What it does | Output |
|---|---|---|
| C1 | 6 filters of 5×5 (output size 32 − 5 + 1 = 28) | 6 maps of 28×28 |
| S2 | subsample | 6 maps of 14×14 |
| C3 | 16 maps from 5×5 filters; each map looks at only *some* of the 6 maps below (a fixed connection table, to break symmetry and save computation) | 16 maps of 10×10 |
| S4 | subsample | 16 maps of 5×5 |
| C5 | 120 units, each seeing all of S4 | 120 |
| F6 | 84 units | 84 |
| Output | 10 **RBF** units, one per digit; each measures the distance between F6's output and a fixed 7×12 "drawing" of its digit; the smallest distance wins | 10 |

The total is about 60,000 trainable weights.

**Topic 5: why the loss needed care.** If the RBF targets were also learned with a plain squared-error loss, the network could cheat: make all targets equal and all outputs equal, giving zero loss but recognising nothing. The paper keeps the targets fixed and adds a term that pushes wrong classes away.

**Topic 6: whole systems.** The paper's second half shows how to train a full document-reading system end to end (segmenting characters and reading them together), using "graph transformer networks". This was deployed to read millions of bank cheques.

**Why it matters.** Convolution, weight sharing and pooling are the foundations of computer vision, and the "conv → pool → conv → pool → fully connected" layout lasted for decades.

</details>

#### 014 · [ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)](03-CNNs-and-Vision/014-Krizhevsky-et-al-2012-AlexNet/) — Krizhevsky, Sutskever & Hinton, 2012
- **The idea:** **AlexNet** won the 2012 ImageNet competition by a huge margin, using a deep convolutional network, a simple activation called ReLU, dropout, and graphics cards. This is often called the start of the deep-learning revolution.
- **What we built:** AlexNet with its original two-GPU design, its normalisation trick and its data-enlarging tricks (cropping, flipping, colour changes).
- **What we saw:** our model has exactly the expected number of parameters (about 61 million). The full ImageNet training is written but not run.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before this paper.** ImageNet is a competition with 1.2 million photos in 1,000 categories. Before 2012 the best systems used hand-designed features, with a top-5 error around 26% (top-5 error: the true label is not among the system's 5 best guesses). Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton entered a deep convolutional network and got **15.3%**, a huge leap.

**Topic 1: the network.** Five convolutional layers, then three fully connected layers, ending in a 1,000-way softmax (a function turning scores into probabilities). In total, about **60 million parameters** and 650,000 neurons.

**Topic 2: ReLU.** Instead of tanh or sigmoid, AlexNet used the **Rectified Linear Unit**: f(x) = max(0, x). It never saturates for positive inputs, so gradients don't vanish. On a small test, ReLU networks reached the same error **six times faster** than tanh networks.

**Topic 3: two GPUs.** The network didn't fit in one graphics card's 3 GB of memory. So it was split in half across two GPUs, which communicated only at certain layers. This is an early example of **model parallelism** (Stage 13).

**Topic 4: local response normalisation.** Each neuron's output is divided by a sum over neighbouring channels at the same position, so strongly active channels suppress their neighbours (like "lateral inhibition" in biology). Later networks dropped this in favour of batch norm.

**Topic 5: overlapping pooling.** Max-pooling windows of 3×3 that move 2 pixels at a time, so neighbouring windows overlap. This slightly reduced overfitting.

**Topic 6: fighting overfitting with more data and dropout.**
- **Random crops:** train on random 224×224 crops of 256×256 images, plus mirror images. That gives 2,048 times more variations.
- **Colour jitter (PCA lighting):** add small random amounts of the image set's main colour directions, so the network ignores lighting changes.
- **Dropout** (paper 009) with p = 0.5 in the first two fully connected layers.
- At test time, average the predictions over 10 crops (corners, centre and their mirrors).

**What we built and saw.** The full architecture with both GPU halves, and every augmentation. Our model has the expected ~61 million parameters. Full ImageNet training (days on GPUs) is written but not run.

**Why it matters.** AlexNet's win convinced the field that deep learning works. GPUs, ReLU, dropout and data augmentation all became standard, and the modern AI boom started here.

</details>

#### 015 · [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)](03-CNNs-and-Vision/015-Simonyan-Zisserman-2015-VGG/) — Simonyan & Zisserman, 2015
- **The idea:** **VGG** showed that simply stacking many small 3×3 filters, making the network deeper, gives better results.
- **What we built:** all six network designs from the paper, with their exact sizes.
- **What we saw:** the sizes match the paper exactly (138 million parameters for the most famous version), and we showed how badly very deep networks train without a careful start.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The question.** After AlexNet, people asked: what matters more, clever layer designs or simply **depth**? Karen Simonyan and Andrew Zisserman (Oxford's Visual Geometry Group, hence "VGG") built very regular networks made only of 3×3 convolutions and made them deeper and deeper.

**Topic 1: stacks of small filters.** Two 3×3 convolutions in a row "see" a 5×5 area of the input; three see 7×7. Why not use one 7×7 filter?
- **Fewer weights:** with C channels, three 3×3 layers use 3 × 9 × C² = 27C² weights, while one 7×7 layer uses 49C². That is 45% fewer.
- **More non-linearity:** three ReLUs instead of one, so the network can express more complex functions.

**Topic 2: a simple, regular design.**
- Blocks of 3×3 convolutions, each followed by 2×2 max-pooling.
- The number of channels doubles after each pool (64 → 128 → 256 → 512).
- The network ends with three fully connected layers.

The paper tests configurations **A to E**, from 11 to 19 weight layers. VGG-16 (configuration D) has **138 million parameters**, most of them in the first fully connected layer.

**Topic 3: training deep nets in 2014.** Before good initialisation methods were common, the 19-layer network was hard to train from random weights. The authors first trained the shallow network A, then used its weights to start the deeper ones. (They later noted that Glorot initialisation, paper 007, also works.)

**Topic 4: multi-scale training and testing.** Images were rescaled to different sizes during training, so the network learned objects at many scales. At test time, predictions from several scales were averaged.

**Topic 5: results.** Deeper was better, up to 19 layers. VGG came 2nd in ImageNet classification in 2014 (about 7% top-5 error) and 1st in localisation.

**What we built and saw.** All six configurations with the exact layer sizes; the parameter counts match the paper's table. We also showed that very deep VGG-style networks barely train without careful initialisation, which explains the "train shallow first" trick.

**Why it matters.** VGG's "small filters, more depth" became a standard design rule. Its features were widely reused for other tasks (for example in "style transfer"). Its training troubles at 19 layers motivated ResNet.

</details>

#### 016 · [Deep Residual Learning for Image Recognition (ResNet)](03-CNNs-and-Vision/016-He-et-al-2016-ResNet/) — He et al., 2016
- **The idea:** **skip connections**: let each block add its output to its input, so the signal has a shortcut path. This made it possible to train networks with over 100 layers, and the idea is now in almost every modern network, including transformers.
- **What we built:** ResNets from 18 to 152 layers, and the very deep small versions (up to 1,202 layers) from the paper.
- **What we saw:** the sizes and amounts of computation match the paper exactly, and we measured how plain networks without shortcuts struggle as they get deeper.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The puzzle.** If deeper is better, why not 50 or 100 layers? Kaiming He and colleagues found something surprising: a 56-layer plain network had **higher training error** than a 20-layer one. That is not overfitting (that would show up only on test data). The deeper network was simply harder to **optimise**. Yet in principle the deeper network could copy the shallower one and set its extra layers to "do nothing" (the identity). The optimiser just couldn't find that solution.

**Topic 1: residual learning.** Instead of asking a block of layers to learn the desired function H(x) directly, let it learn the **difference** F(x) = H(x) − x and add the input back:
```
output = F(x) + x
```
If the best thing a block can do is nothing, it only has to push F to zero, which is easy (weights near 0). The "+ x" is a **skip (shortcut) connection** that costs no parameters.

**Topic 2: why gradients flow.** The slope of x + F(x) with respect to x is 1 + F′(x). Even if F′ is tiny, the gradient still passes back through the "1". A signal from the loss can reach the first layer directly through the chain of shortcuts, so very deep networks still get a learning signal.

**Topic 3: bottleneck blocks.** For the deeper models (50, 101 and 152 layers), each block is 1×1 conv (shrink channels, for example 256 → 64), then 3×3 conv (on the small number of channels), then 1×1 conv (expand back to 256). This makes depth affordable.

**Topic 4: changing size.** When a block halves the image size and doubles the channels, the shortcut can't add directly. Options:
- **A:** pad the extra channels with zeros;
- **B:** use a 1×1 convolution only where sizes change;
- **C:** use 1×1 convolutions everywhere.

B is the usual choice.

**Topic 5: results.**
- ResNet-152 won ImageNet 2015 with **3.57% top-5 error** (an ensemble), better than the human estimate of about 5%.
- On CIFAR-10, a 110-layer network trained easily.
- A **1,202-layer** network also trained, but it overfit and was a bit worse than the 110-layer one.

**What we built and saw.** ResNets from 18 to 152 layers and the very deep CIFAR versions; parameter counts and FLOPs match the paper. Plain (no-shortcut) networks got worse as they got deeper, while residual ones did not, reproducing the degradation problem.

**Why it matters.** Skip connections are everywhere: in every transformer block (paper 034), in U-Nets for image generation, and in large language models. ResNet is one of the most cited papers in all of science.

</details>

---

### Stage 04 · What networks really learn: surprises

**The story.** Once deep networks worked, researchers found two strange things. They can be fooled by changes to an image so small a person can't see them, and they can perfectly memorise completely random labels. Both papers make you rethink what "learning" means for these models.

#### 017 · [Intriguing Properties of Neural Networks](04-Robustness-and-Generalization/017-Szegedy-et-al-2013-Intriguing-Properties/) — Szegedy et al., 2013
- **The idea:** discovered **adversarial examples**: tiny, invisible changes to an image that make a network confidently give the wrong answer.
- **What we built:** the paper's method for finding the smallest such change, and its way of measuring how sensitive each layer is.
- **What we saw:** on a simple case where the answer is known, the method finds exactly the smallest change.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The discovery.** Christian Szegedy and colleagues (including Ilya Sutskever and Ian Goodfellow) studied trained image networks and found two "intriguing properties".

**Topic 1: single units are not special.** People liked to look at which images most strongly activate a single neuron and say "this neuron detects dogs". The paper compared this with **random directions** (random mixtures of neurons). The images that maximise a random mixture look just as meaningful. So the information is spread across the whole layer, not stored in individual units.

**Topic 2: adversarial examples.** For almost any correctly classified image, there is a **tiny change**, invisible to people, that makes the network classify it wrongly, often with high confidence. A school bus becomes an "ostrich".

**Topic 3: how to find them.** Look for the smallest change r such that the network outputs a chosen wrong label l:
```
minimise   c·|r|  +  loss(x + r, l)       while keeping x + r a valid image (pixels in [0,1])
```
They used **box-constrained L-BFGS** (an optimiser for smooth problems with bounds), and searched over c to find the smallest change that still fools the network.

**Topic 4: why tiny changes add up.** (This intuition was made precise by Goodfellow in 2014.) A linear unit computes w·x. If each pixel changes by a tiny ε in the direction of its weight's sign, the output changes by ε × (sum of |w|). With 784 pixels, ε = 0.01 and an average |w| of 0.5, that is 784 × 0.01 × 0.5 ≈ 3.9: a big change from invisible pixel changes. High-dimensional inputs make small per-pixel changes powerful.

**Topic 5: they transfer.** Adversarial examples made for one network often fool **other** networks, even ones with different architectures or trained on different data. So they reflect something systematic, not a quirk of one model.

**Topic 6: measuring instability.** The paper bounds how much each layer can amplify a change, using each layer's largest "stretching factor" (its operator norm). Large factors mean an unstable network.

**What we built and saw.** The L-BFGS search and the layer-by-layer stability bounds. On a simple model where the smallest possible change can be computed exactly, our search found exactly that change.

**Why it matters.** This paper started the field of **adversarial robustness**. It matters for safety (self-driving cars, spam filters, malware detection), and it reappears in paper 102 on malicious uses of AI.

</details>

#### 018 · [Understanding Deep Learning Requires Rethinking Generalization](04-Robustness-and-Generalization/018-Zhang-et-al-2017-Rethinking-Generalization/) — Zhang et al., 2017
- **The idea:** showed that networks can **memorise random labels** perfectly. This means traditional explanations of why they work on new data can't be the whole story.
- **What we built:** all the paper's randomisation tests (shuffled labels, random pixels), and its proof that a fairly small network can fit any labels at all.
- **What we saw:** the proof's construction works as stated.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The puzzle.** Modern networks have far more parameters than training examples. Classical learning theory says such models should overfit badly, yet they generalise well. Chiyuan Zhang and colleagues ran simple experiments that showed the classical explanations can't be the whole story.

**Topic 1: the randomisation test.** Take CIFAR-10 (60,000 images in 10 classes) and **replace every label with a random one**. There is now nothing real to learn. Yet standard networks reach **zero training error**: they memorise all 50,000 random labels perfectly. Training takes only somewhat longer than with true labels. Test accuracy is of course at chance (10%).

**Topic 2: other randomisations.**
- **Random pixels:** replace images with pure noise. The networks still memorise them.
- **Shuffled pixels:** the same permutation for all images, or a different one per image. Still memorised.
- **Partially corrupted labels:** as more labels are randomised, test error rises smoothly.

**Topic 3: what this means.** A model's **capacity** (how many different labelings it can fit) is often used to explain generalisation: low capacity → can't memorise → must generalise. These networks have enough capacity to memorise anything, yet with real labels they generalise. So capacity measures like Rademacher complexity (which literally asks "can you fit random labels?") cannot explain their success.

**Topic 4: regularisation isn't the answer either.** Turning off weight decay, dropout and data augmentation reduces test accuracy only modestly. Networks still generalise reasonably without explicit regularisation. Regularisers help, but they are not the main reason.

**Topic 5: the expressivity theorem.** A two-layer ReLU network with 2n + d weights can fit **any** labels on n data points in d dimensions. The construction is simple:
1. Pick a random direction a and project every point onto it: z_i = a·x_i. With probability 1 these numbers are all different. Sort them.
2. Put the ReLU "kinks" (biases) between consecutive sorted values.
3. Then the network's outputs form a **triangular** system of equations, which can always be solved for the output weights.

**Topic 6: implicit regularisation.** For linear models, gradient descent started from zero finds the **smallest-norm** solution among all solutions that fit the data. So the training algorithm itself prefers "simple" solutions. The paper suggests SGD plays a similar role in deep networks.

**What we built and saw.** All the randomisation experiments (written for real data, not run) and the theorem's construction, which fits arbitrary labels exactly as stated.

**Why it matters.** It started a whole research area on why overparameterised networks generalise (including "double descent" and the study of implicit bias).

</details>

---

### Stage 05 · Words and sequences: from word vectors to attention

**The story.** Language comes as sequences, and the meaning of a word depends on its neighbours. This stage builds up, one step at a time, to the most important idea in modern AI. First, words become lists of numbers (embeddings). Then networks with memory read sentences word by word (RNNs and LSTMs). Then two networks work together to translate whole sentences (sequence-to-sequence). Finally, **attention** lets the translator look back at the right source words at each step.

#### 019 · [Distributed Representations of Words and Phrases (Word2Vec)](05-Words-and-Sequences/019-Mikolov-et-al-2013-Word2Vec-Negative-Sampling/) — Mikolov et al., 2013
- **The idea:** **Word2Vec** turns each word into a list of numbers so that similar words get similar numbers, learned simply by predicting nearby words. It's famous for arithmetic like *king − man + woman ≈ queen*.
- **What we built:** the model with every training shortcut the paper describes, each checked for correctness.
- **What we saw:** all the training methods compute correct gradients. The full experiment on a large text collection is written but not run.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before this paper.** Computers treated words as unrelated symbols. To a program, "cat" and "kitten" were as different as "cat" and "carburettor". Tomas Mikolov and colleagues at Google showed how to learn a list of numbers (a **vector**) for every word from plain text, so that words used in similar contexts get similar vectors.

**Topic 1: the distributional idea.** "You shall know a word by the company it keeps." Words that appear near the same neighbours tend to mean similar things.

**Topic 2: skip-gram.** Slide over the text. For each word, try to predict the words within a small window around it. In "the cat sat on the mat" with window 2, the word "sat" should predict "the", "cat", "on" and "the". Each word has an **input vector** v and an **output vector** v′. The model scores a pair by their dot product v_sat · v′_cat. Training adjusts the vectors so that real neighbours score high.

**Topic 3: the expensive softmax.** Turning scores into probabilities over all words requires summing over the whole vocabulary (maybe a million words) for every training pair. That is far too slow. The paper gives two fixes.
- **Hierarchical softmax:** arrange the words as leaves of a binary tree (a Huffman tree, so frequent words are near the top). The probability of a word is the product of left/right decisions along its path: about log₂(V) ≈ 20 steps instead of 1,000,000.
- **Negative sampling:** turn the problem into "real pair or fake pair?". For each real (word, context) pair, draw k random "noise" words (k = 5–20 for small data, 2–5 for large data) and train:
```
maximise   log σ(v′_context · v_word)  +  Σ over k noise words  log σ(−v′_noise · v_word)
```

**Topic 4: the 3/4 power.** Noise words are drawn with probability proportional to frequency^0.75. This boosts rare words. Example: two words with frequencies 0.9 and 0.1. Raised to 0.75 they become 0.924 and 0.178, so after normalising their probabilities are 0.84 and 0.16. The rare word's chance rose from 10% to 16%.

**Topic 5: subsampling frequent words.** Words like "the" appear constantly and carry little information. Each occurrence is thrown away with probability 1 − √(t / frequency), with t = 10⁻⁵. This speeds training and improves rare-word vectors.

**Topic 6: phrases.** "New York" is not "new" + "York". Word pairs that occur together far more often than chance (scored by (count(ab) − δ) / (count(a)·count(b))) are merged into single tokens like "New_York".

**Topic 7: vector arithmetic.** The famous result: vec("king") − vec("man") + vec("woman") is closest to vec("queen"). Relationships (gender, capital-of, verb tense) become consistent **directions** in the vector space.

**What we built and saw.** Skip-gram with both softmax alternatives, subsampling and phrase detection. Every gradient was checked numerically; the large-corpus training is written but not run.

**Why it matters.** Word vectors (**embeddings**) are the first layer of every language model, including GPT. The idea of learning embeddings by predicting context leads straight to BERT and GPT.

</details>

#### 020 · [Exploiting Similarities among Languages for Machine Translation](05-Words-and-Sequences/020-Mikolov-et-al-2013-Similarities-Among-Languages/) — Mikolov, Le & Sutskever, 2013
- **The idea:** word vectors from two languages have similar shapes, so a simple rotation-like mapping can translate words between them.
- **What we built:** the mapping between languages, the paper's accuracy measures and its baselines.
- **What we saw:** the pieces work as described; the full experiment on real language data is written but not run.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The idea.** If word vectors capture meaning, then the vectors for English and for Spanish should have a **similar shape**. Numbers, animals and family words should sit in similar arrangements in both languages, maybe rotated or stretched. If so, a simple mapping could translate words.

**Topic 1: learning the mapping.** Take a small dictionary of known pairs (the paper used about 5,000 frequent words), for example (one, uno), (cat, gato). Find a matrix W that maps each English vector x close to its Spanish vector z:
```
minimise   Σ  || W·x_i − z_i ||²
```
This is ordinary least-squares regression, solvable by gradient descent or directly.

**Topic 2: translating a new word.** To translate an English word not in the dictionary, compute W·x and find the **nearest** Spanish word vector (using cosine similarity, which measures the angle between vectors).

**Topic 3: measuring quality.**
- **Precision@1:** the correct translation is the single nearest word.
- **Precision@5:** it is among the 5 nearest.

**Topic 4: why it works (the picture).** The paper shows a 2-D plot (made with PCA) of number words and animal words in English and Spanish. The two clouds have nearly the same arrangement, just rotated. Concepts have similar relationships in every language, because languages describe the same world.

**Topic 5: confidence and dictionary repair.** The distance between W·x and the nearest word acts as a confidence score. High-confidence translations are much more accurate. The method can even spot mistakes in an existing dictionary: entries where the mapping strongly disagrees.

**Topic 6: baselines.** The method was compared with edit distance (similar spelling, useful for related languages) and with older co-occurrence-based methods, and beat them.

**What we built.** The mapping, nearest-neighbour translation, precision@k, confidence, and the baselines. Real-data experiments are written but not run.

**Why it matters.** This showed that meaning has a **geometry** shared across languages. Later work built on it to align languages with *no* dictionary at all, and modern multilingual models share one embedding space across languages.

</details>

#### 021 · [Long Short-Term Memory (LSTM)](05-Words-and-Sequences/021-Hochreiter-Schmidhuber-1997-LSTM/) — Hochreiter & Schmidhuber, 1997
- **The idea:** the **LSTM**, a memory cell for sequence networks that can keep information for a long time without it fading. It powered speech recognition and translation for two decades.
- **What we built:** the original LSTM (without the "forget gate" added later), with the paper's own learning rule.
- **What we saw:** the memory cell keeps its information perfectly as designed, and we measured a known weakness (its stored value slowly drifts) together with the paper's fix.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before this paper.** Recurrent neural networks (RNNs) read a sequence one step at a time, keeping a hidden state as memory. In practice they could not learn to remember anything for more than about 10 steps. Sepp Hochreiter and Jürgen Schmidhuber analysed why, and designed a cell that could remember for **1,000** steps.

**Topic 1: vanishing (and exploding) gradients.** To learn that something at step 1 caused an error at step 100, the error signal must travel back 99 steps. At each step it is multiplied by (recurrent weight × slope of the activation). If that product is 0.9, after 100 steps the signal is 0.9¹⁰⁰ ≈ 0.00003, which is effectively gone. If it is 1.1, it becomes 1.1¹⁰⁰ ≈ 13,780, which explodes.

**Topic 2: the constant error carousel.** The fix: a memory cell whose value is carried forward by a self-connection with weight **exactly 1.0** and **no squashing** activation:
```
c_t = c_{t−1} + (new input)
```
Now the multiplier per step is exactly 1, so the error flows back unchanged across hundreds of steps. That is the "carousel".

**Topic 3: gates protect the cell.** A cell that adds every input would fill up with junk. So each cell has two **gates** (units with a 0–1 sigmoid output that multiply a signal):
- the **input gate** decides when new information may be written into the cell;
- the **output gate** decides when the cell's content may influence the rest of the network.

This lets the cell hold a value undisturbed until it is needed. The famous **forget gate** (which lets the cell erase itself) was added later, in 2000, by Gers and colleagues. The original LSTM had no forget gate.

**Topic 4: the training rule.** The original paper used a **truncated** gradient: errors flow back only inside the memory cells and are cut off elsewhere. This keeps computation cheap and local.

**Topic 5: experiments.** Artificial tasks designed to need long memory, for example remembering the first symbol of a long noisy sequence, or adding two marked numbers far apart. LSTM solved tasks with time lags over 1,000 steps that other RNNs could not.

**What we built and saw.** The original LSTM (no forget gate) with its truncated learning rule. The carousel preserved error signals exactly. We also measured a known weakness: without a forget gate the cell's value can drift and grow over long sequences. The paper's countermeasures (such as starting gates biased toward "closed") helped.

**Why it matters.** LSTMs powered speech recognition, translation and handwriting recognition from about 2007 to 2017 (Google Translate, Siri, Alexa). The gating idea also inspired the GRU (paper 026) and survives in many modern architectures.

</details>

#### 022 · [Generating Text with Recurrent Neural Networks](05-Words-and-Sequences/022-Sutskever-et-al-2011-Generating-Text-with-RNNs/) — Sutskever, Martens & Hinton, 2011
- **The idea:** a network that writes text **one character at a time**, using a special design where each input character changes how the memory updates.
- **What we built:** three network designs and a special second-order training method.
- **What we saw:** the training method's maths checks out. The large training run is written but not run.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The goal.** Train a recurrent network to model text **one character at a time**, then let it write new text. Ilya Sutskever, James Martens and Geoffrey Hinton wanted to show that RNNs could do this well when trained with a powerful optimiser.

**Topic 1: character-level language modelling.** At each step the network sees one character and outputs a probability for every possible next character. Training maximises the probability of the real next character. To **generate**, sample a character from the output, feed it back in, and repeat.

**Topic 2: measuring quality, bits per character.** If the model gives the true next character probability p, it "costs" −log₂ p bits. The average over the text is **bits per character (bpc)**. Example: always giving the right character probability 0.5 means 1 bit per character. Lower is better; good English models get around 1.5 bpc or below.

**Topic 3: the multiplicative RNN (MRNN).** In a normal RNN, the input character is *added* to the hidden state's update. In the MRNN, the input character **chooses which transition matrix** is used:
```
h_t = tanh( W_hx·x_t + W_hh^(x_t) · h_{t−1} )
```
"After seeing 'q', move the memory this way; after 'u', that way." A separate full matrix per character would be huge, so it is **factorised**:
```
W_hh^(x) = W_hf · diag(W_fx · x) · W_fh
```
That is: project the hidden state onto some factors, scale each factor by an amount that depends on the character, and project back.

**Topic 4: Hessian-free optimisation.** RNNs were hard to train with plain gradient descent. Hessian-free (HF) optimisation uses **curvature**: it uses not just the slope but how the slope changes. Inside each step it solves for a good update with the **conjugate gradient** method, using only products of the curvature matrix with vectors (never building the full matrix). They used the **Gauss–Newton** approximation of curvature and "structural damping", which keeps the hidden states from changing too wildly.

**Topic 5: what it wrote.** Trained on Wikipedia, the model generated text with real words, plausible sentence structure, correctly balanced brackets and quotes, and invented but plausible names and links, all learned from raw characters.

**What we built.** The plain RNN, the MRNN and a gated variant, plus HF optimisation. The curvature-vector products were checked against exact computations. The large training runs are written but not run.

**Why it matters.** This was a striking early demonstration that a neural network can learn the structure of language from raw characters. Andrej Karpathy's famous 2015 "char-rnn" blog post popularised the same idea with LSTMs.

</details>

#### 023 · [Training Recurrent Neural Networks (PhD thesis)](05-Words-and-Sequences/023-Sutskever-2013-PhD-Thesis-Training-RNNs/) — Sutskever, 2013
- **The idea:** a long study of how to make sequence networks learn hard, long-range patterns, including a model that learns videos of bouncing balls.
- **What we built:** the thesis's video model and its improved training method, plus all eight "hard problems" it uses as tests.
- **What we saw:** the methods compute what they should, checked against exact calculations.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**What it is.** Ilya Sutskever's PhD thesis (University of Toronto, 2013). It gathers his work on making recurrent networks learn hard problems, and much of it reappears in other papers in this repository (008, 022).

**Topic 1: why RNN training is hard.** The thesis gives an accessible explanation of vanishing and exploding gradients (see paper 021) and of why the error surface of an RNN has sharp cliffs and long flat plateaus.

**Topic 2: Hessian-free training for RNNs.** A detailed account of using second-order (curvature-aware) optimisation, including the "structural damping" trick that stabilises it (see paper 022).

**Topic 3: momentum and initialisation as an alternative.** The thesis shows that carefully tuned momentum and good initialisation reach similar results with plain first-order methods (this became paper 008).

**Topic 4: the "pathological" problems.** A test suite of synthetic tasks designed so that the network **must** remember information over long time spans. Examples:
- **Addition:** a long stream of random numbers, two of which are marked; output their sum at the end.
- **Multiplication:** the same, but output the product.
- **Temporal order:** two special symbols appear at random times among noise; report which came first.
- **Memorisation:** remember a short sequence and repeat it after a long delay.

These problems are easy to state but impossible without long memory, so they are a clean test of an optimiser.

**Topic 5: modelling video, the RTRBM.** For sequences of images (videos of bouncing balls), the thesis uses a **Recurrent Temporal Restricted Boltzmann Machine**.
- A **Restricted Boltzmann Machine** (RBM) is a probabilistic model with visible units (the image) and hidden units (features). Each joint state has an "energy", and low-energy states are more likely.
- It is trained with **contrastive divergence**: make the real data more likely and the model's own fantasies less likely.
- The **recurrent** version lets the hidden units at each frame depend on the previous frame, so it can predict motion.

**What we built and saw.** The RTRBM for bouncing-ball video, the improved training, and all eight pathological problems. Every gradient and probability computation matches exact calculations on small cases.

**Why it matters.** Many ideas that made deep learning practical (momentum schedules, initialisation, curvature-aware training, sequence models) are developed here in one place. Its author went on to co-create sequence-to-sequence learning (paper 027) and co-found OpenAI.

</details>

#### 024 · [Recurrent Neural Network Regularization](05-Words-and-Sequences/024-Zaremba-et-al-2014-RNN-Regularization/) — Zaremba, Sutskever & Vinyals, 2014
- **The idea:** dropout (from Stage 02) didn't work for LSTMs until this paper showed **where** to apply it: between layers, not on the memory connections.
- **What we built:** a deep LSTM language model with the paper's dropout placement, plus the two wrong placements for comparison.
- **What we saw:** the dropout lands exactly where intended. The full training runs are written but not run.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** Dropout (papers 009–010) worked well for feedforward networks, but when people applied it to LSTMs, results got **worse**. Wojciech Zaremba, Ilya Sutskever and Oriol Vinyals found out why and how to do it right.

**Topic 1: where can dropout go in an RNN?** In a multi-layer LSTM, information flows in two directions:
- **vertically:** from the input up through the layers to the output at the same time step;
- **horizontally:** from one time step to the next through the recurrent connections, which carry the memory.

**Topic 2: the rule.** Apply dropout **only to the vertical (non-recurrent) connections**. If you drop parts of the memory at every time step, then after 50 steps the memory has been hit by 50 rounds of noise, and the network can no longer remember long-range information. Dropping only between layers means any piece of information is corrupted a limited number of times (about the number of layers + 1), no matter how long ago it was stored.

**Topic 3: the model.** A 2-layer LSTM language model that reads Penn Treebank text (about 1 million words, 10,000-word vocabulary) word by word. It has a "medium" version (650 units per layer, 50% dropout) and a "large" version (1,500 units, 65% dropout).

**Topic 4: measuring with perplexity.** **Perplexity** = e^(average loss per word). It can be read as "the model is as confused as if it were choosing uniformly among this many words". With correct dropout the large model reached a test perplexity of about **78**, against about 114 for a non-regularised model. That was a major improvement.

**Topic 5: beyond language modelling.** The same trick improved speech recognition, machine translation and image captioning.

**What we built and saw.** A deep LSTM language model with dropout placed per the paper, plus the two wrong placements (on recurrent connections, and everywhere) for comparison. We checked that the masks land exactly where intended. The full training runs are written but not run.

**Why it matters.** It became the standard way to regularise RNNs, and a good lesson: *where* you apply a technique matters as much as the technique itself.

</details>

#### 025 · [An Empirical Exploration of Recurrent Network Architectures](05-Words-and-Sequences/025-Jozefowicz-et-al-2015-Empirical-Exploration-RNN-Architectures/) — Jozefowicz, Zaremba & Sutskever, 2015
- **The idea:** searched through thousands of variations of the LSTM to see which parts matter. A key finding: starting the "forget gate" open makes a big difference.
- **What we built:** all ten memory-cell designs from the paper and the search procedure itself.
- **What we saw:** our general framework reproduces the standard LSTM and GRU exactly.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The question.** The LSTM's design (paper 021, plus the forget gate) was chosen by hand. Is it the best possible design? Rafal Jozefowicz, Wojciech Zaremba and Ilya Sutskever ran a large **architecture search** to find out.

**Topic 1: the search.** They represented an RNN cell's update equations as a computation graph and **mutated** them randomly: swapping operations (add, multiply, tanh, sigmoid…), rewiring inputs, and so on. Each candidate was trained on several tasks and the best ones survived to be mutated further. They evaluated over **10,000** architectures.

**Topic 2: the tasks.**
- Arithmetic (reading digits and computing sums and differences, character by character).
- XML modelling (predicting characters of nested tags, which needs a stack-like memory).
- Penn Treebank language modelling.
- Music modelling.

**Topic 3: the GRU vs the LSTM.** The **GRU** (paper 026) outperformed the standard LSTM on most tasks except language modelling. But a single change, initialising the LSTM's **forget gate bias to 1**, closed most of the gap. With bias 1, the forget gate starts "open" (σ(1) ≈ 0.73), so the cell remembers by default from the very start of training, and gradients can flow through time immediately.

**Topic 4: which LSTM parts matter (an ablation).** Removing each gate in turn showed:
- the **forget gate** is the most important;
- the **input gate** is next;
- the **output gate** matters least.

**Topic 5: the discovered cells.** The search found three new architectures (MUT1, MUT2, MUT3) that rivalled or beat the GRU on some tasks, but none dramatically better than a well-initialised LSTM or a GRU. Conclusion: the LSTM and GRU are close to a sweet spot.

**What we built and saw.** A general cell framework that expresses all the paper's designs (LSTM, its ablated variants, GRU, MUT1–3) and the mutation-based search. Our framework reproduces the standard LSTM and GRU **exactly** when configured to.

**Why it matters.** "Initialise the forget-gate bias to 1" became standard practice. The paper is also an early example of automated architecture search.

</details>

#### 026 · [Learning Phrase Representations using an RNN Encoder–Decoder](05-Words-and-Sequences/026-Cho-et-al-2014-RNN-Encoder-Decoder/) — Cho et al., 2014
- **The idea:** introduced the **GRU** (a simpler memory cell than the LSTM) and the **encoder–decoder**: one network reads a sentence into a summary, another writes a translation from that summary.
- **What we built:** the GRU and the full encoder–decoder from the paper's appendix.
- **What we saw:** the parts behave as specified; the translation experiments are written but not run.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Two contributions.** Kyunghyun Cho and colleagues introduced the **encoder–decoder** RNN for translation and a simpler gated cell, now called the **GRU** (Gated Recurrent Unit).

**Topic 1: the encoder–decoder.**
- The **encoder** RNN reads the source phrase (for example English) word by word. Its final hidden state is a fixed-size **summary vector** c.
- The **decoder** RNN generates the target phrase (for example French) word by word. At every step it is given c, the previous word and its own hidden state.

The whole system is trained to maximise the probability of the correct translation.

**Topic 2: the GRU.** It has two gates instead of the LSTM's three, and no separate memory cell:
- the **reset gate** r decides how much of the previous state to use when proposing a new state;
- the **update gate** z decides how much to keep the old state vs replace it with the proposal.
```
r  = σ(W_r x + U_r h_prev)
z  = σ(W_z x + U_z h_prev)
h̃ = tanh(W x + U (r ⊙ h_prev))           ← the proposal
h  = z ⊙ h_prev + (1 − z) ⊙ h̃           ← the paper's convention
```
(⊙ means multiply element by element.) If z ≈ 1, the unit just copies its old value forward, which works like the LSTM's constant error carousel, so gradients survive across time. If r ≈ 0, the unit ignores the past and "resets".

**Topic 3: how they used it.** In 2014 the best translation systems were **phrase-based statistical machine translation** (SMT): huge tables of phrase pairs with probabilities. The authors did not replace SMT. They used their network to **score** each phrase pair in the table and added that score as an extra feature. Translation quality, measured with BLEU (paper 027 explains it), improved.

**Topic 4: phrase representations.** The summary vectors c of different phrases formed meaningful clusters. Phrases about time ("for the past few months") clustered together, separately from phrases about countries, and so on. The network learned a semantic and syntactic space of phrases.

**What we built.** The GRU and the full encoder–decoder exactly as in the paper's appendix (including its "maxout" output layer). Translation experiments are written but not run.

**Why it matters.** The encoder–decoder framework (with paper 027) became the standard design for translation, summarisation and speech recognition. Its weakness, squeezing a whole sentence into one vector, led directly to **attention** (paper 028). The GRU is still widely used.

</details>

#### 027 · [Sequence to Sequence Learning with Neural Networks](05-Words-and-Sequences/027-Sutskever-et-al-2014-Seq2Seq/) — Sutskever, Vinyals & Le, 2014
- **The idea:** **sequence-to-sequence**: a deep LSTM reads a whole sentence and another writes the translation. A surprising trick: **reversing the input sentence** makes learning much easier.
- **What we built:** the model, input reversal, and "beam search" (keeping several candidate translations at once).
- **What we saw:** the parts work and beam search finds the best answer on a test case where we know it; the full translation runs are written but not run.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The breakthrough.** Ilya Sutskever, Oriol Vinyals and Quoc Le showed that a large deep LSTM encoder–decoder can translate whole sentences **end to end**, competing with mature statistical systems.

**Topic 1: the model.**
- Two separate **4-layer LSTMs** with 1,000 units per layer: one encodes the English sentence, one decodes the French.
- 1,000-dimensional word embeddings, 160,000-word input vocabulary, 80,000-word output vocabulary.
- About **384 million parameters**, trained on 12 million sentence pairs (WMT'14 English→French).

**Topic 2: the reversal trick.** Feed the source sentence **backwards**. To translate "a b c" into "α β γ", feed "c b a". The first source word "a" is now right next to the first target word "α". These short-range connections make it much easier for training to "get started": the gradients have short paths to follow, and once early words are aligned, later ones follow. Reversing improved BLEU from 25.9 to 30.6 in their experiments.

**Topic 3: decoding with beam search.** Choosing the single most likely word at each step (**greedy** decoding) can lead to a bad sentence overall. **Beam search** keeps the B most probable partial sentences at every step:
1. Start with B = 2 empty hypotheses.
2. Extend each one with every possible next word, scoring each by total log-probability.
3. Keep only the best 2 overall.
4. Repeat until the hypotheses end.

Even a beam of 2 gave most of the benefit.

**Topic 4: BLEU, the translation score.** BLEU compares the machine translation with a human reference by counting matching word sequences (1 to 4 words long), with a penalty for translations that are too short. 100 would mean identical to the reference. In 2014, scores in the mid-30s were state of the art for English→French.

**Topic 5: results.**
- An ensemble of 5 reversed LSTMs scored **34.8 BLEU** on its own, against 33.3 for a strong phrase-based system.
- Using the LSTM to rescore the phrase-based system's top 1,000 candidates reached 36.5.
- Surprisingly, it handled **long sentences** well.

**Topic 6: what the encoder learned.** Plotting the summary vectors showed that sentences with similar meanings are close together, even when word order differs, and that the vectors are sensitive to who did what to whom (active vs passive voice).

**What we built and saw.** Encoder, decoder, input reversal and beam search. Beam search found the true best output on a test case where it can be checked by brute force. Full translation training is written but not run.

**Why it matters.** "Sequence to sequence" (seq2seq) became the template for translation, summarisation, speech recognition and chatbots, and Google Translate switched to neural translation within two years.

</details>

#### 028 · [Neural Machine Translation by Jointly Learning to Align and Translate (Attention)](05-Words-and-Sequences/028-Bahdanau-et-al-2015-Attention-NMT/) — Bahdanau, Cho & Bengio, 2015
- **The idea:** **attention**. Instead of squeezing a whole sentence into one summary, the translator looks back at the most relevant source words at each step. This is the key ingredient of transformers.
- **What we built:** the paper's attention model exactly, and the older model without attention.
- **What we saw:** on reversing 16-symbol sequences, the old model managed 59% while the attention model got 100%, and the attention pattern clearly showed it "looking" at the right positions.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The bottleneck.** In seq2seq (papers 026–027), the whole source sentence is squeezed into **one** fixed-size vector. For long sentences that is like summarising a whole page on a sticky note. Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio let the decoder **look back** at every source word whenever it needs to. This is **attention**.

**Topic 1: keep all the encoder states.** A **bidirectional** RNN reads the sentence both forwards and backwards. Each source word j gets an annotation h_j that summarises the word *and* its surrounding context on both sides.

**Topic 2: score every source word.** At decoder step i, with previous decoder state s_{i−1}, compute a relevance score for each source word j with a small neural network:
```
e_ij = vᵀ · tanh( W·s_{i−1} + U·h_j )
```

**Topic 3: turn scores into weights (softmax).**
```
α_ij = exp(e_ij) / Σ_k exp(e_ik)
```
All weights are positive and add up to 1.

**Topic 4: build a context vector.** Take a weighted average of the annotations: c_i = Σ_j α_ij · h_j. The decoder uses c_i to choose the next word. A **different** context vector is computed for every output word.

**Worked example.** Three source words with scores e = [2, 0, 1].
- exp gives [7.39, 1, 2.72], which sums to 11.11.
- The weights are α = [0.665, 0.090, 0.245].
- The context is 66.5% word 1, 9% word 2 and 24.5% word 3.

**Topic 5: soft alignment.** The weights α_ij form an **alignment** between output and input words, learned without ever being told which words correspond. In English→French, the alignment plots are mostly diagonal. Where French reverses adjective–noun order ("European Economic Area" → "zone économique européenne"), the plot shows the correct reversed pattern.

**Topic 6: results.** Without attention, quality dropped sharply for sentences longer than about 20 words. With attention, quality stayed high even for long sentences.

**What we built and saw.** The attention model exactly as described, plus the no-attention baseline. On reversing 16-symbol sequences, the baseline got 59% and the attention model **100%**, and the attention weights formed a clean anti-diagonal: the model looked at exactly the right position.

**Why it matters.** Attention is the single most important idea leading to transformers (paper 034), where it replaces recurrence entirely. Every modern language model is built on it.

</details>

#### 029 · [Show and Tell: A Neural Image Caption Generator](05-Words-and-Sequences/029-Vinyals-et-al-2015-Show-and-Tell/) — Vinyals, Toshev, Bengio & Erhan, 2015
- **The idea:** describe a picture in a sentence by connecting an image network to a sentence-writing LSTM.
- **What we built:** the captioning model and the standard caption-scoring measures.
- **What we saw:** in a small made-up picture world, it wrote correct captions for combinations it had never seen. It also "beat humans" on the automatic score, showing why that score can be misleading.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The task.** Given a photo, write a sentence describing it, for example "A group of people shopping at an outdoor market." Oriol Vinyals and colleagues treated this like translation: translate an image into English.

**Topic 1: the architecture.**
- A **convolutional network** (GoogLeNet with batch norm), pre-trained on ImageNet, turns the image into a vector.
- This vector is fed into an **LSTM** as if it were the very first "word".
- The LSTM then generates the caption word by word, each word fed back in as the next input.
- Words are represented by learned **embeddings**.

**Topic 2: training.** Maximise the probability of the human-written captions:
```
log p(caption | image) = Σ_t log p(word_t | image, word_1 … word_{t−1})
```

**Topic 3: generation.** At test time, use **beam search** (paper 027) to find a high-probability caption.

**Topic 4: avoiding overfitting.** Caption datasets are small compared with ImageNet. Key choices:
- start the image network from ImageNet-trained weights (**transfer learning**);
- use dropout;
- average several models (an ensemble).

**Topic 5: evaluating captions.** The authors used automatic scores (BLEU, METEOR, CIDEr), which count overlapping words with human captions, and also **human ratings**. A major lesson: on some datasets the model's BLEU score came close to, or matched, the score of human captions, yet human judges still clearly preferred the human captions. Automatic metrics can be fooled.

**Topic 6: novel descriptions.** The model often generated captions that did not appear in the training set, so it was not just retrieving stored sentences.

**What we built and saw.** The captioning model and the scoring metrics, in a small synthetic "picture world" of coloured shapes in positions. The model wrote correct captions for **combinations it had never seen** during training. It also "beat humans" on the automatic score, which reproduces the paper's warning that the metric can be misleading.

**Why it matters.** It established the "vision encoder + language decoder" recipe. Modern models that talk about images (GPT-4o, paper 058) are descendants of this idea.

</details>

#### 030 · [Grammar as a Foreign Language](05-Words-and-Sequences/030-Vinyals-et-al-2015-Grammar-as-Foreign-Language/) — Vinyals et al., 2015
- **The idea:** treat working out a sentence's grammatical structure as "translating" it into a bracketed tree.
- **What we built:** converting trees to and from text, the attention model, and the standard scoring tool.
- **What we saw:** on a toy grammar, attention clearly helped (89.8 vs 80.5 on the standard score, with a bigger gap on long sentences).

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The idea.** **Parsing** means finding the grammatical structure of a sentence: which words form a noun phrase, which form a verb phrase, and so on. It was traditionally done with specialised algorithms. Oriol Vinyals and colleagues asked: what if we just treat the parse tree as **another language**, and translate the sentence into it with seq2seq + attention?

**Topic 1: turning a tree into text (linearisation).** A tree can be written as a bracketed string by walking it depth-first. "John has a dog ." becomes:
```
(S (NP NNP )NP (VP VBZ (NP DT NN )NP )VP . )S
```
Words are replaced by their part-of-speech tags (NNP = proper noun, VBZ = verb, DT = determiner, NN = noun). The parser outputs this string token by token, and the tree can be rebuilt from it.

**Topic 2: the model.** A 3-layer LSTM encoder–decoder with **attention** (paper 028). The input is the sentence (reversed, as in paper 027); the output is the bracketed tree.

**Topic 3: attention is essential.** Trained only on the standard Penn Treebank (about 40,000 sentences), the model without attention was poor. With attention it matched good traditional parsers. Attention lets the decoder keep track of which input word it is "at" while writing brackets.

**Topic 4: more data with synthetic labels.** Even better results came from training on millions of sentences parsed automatically by existing parsers (a form of **self-training** or distillation). With that extra data, the model reached state-of-the-art accuracy.

**Topic 5: measuring parsers with F1.** Each bracket is a "constituent" (a label covering a span of words).
- **Precision:** the fraction of predicted constituents that are correct.
- **Recall:** the fraction of true constituents that were found.
- **F1** = 2 × precision × recall / (precision + recall). 100 is perfect.

**Topic 6: output validity.** The model can, in principle, output an unbalanced tree. In practice this happened very rarely, and simple fixes (adding missing brackets) handle it.

**What we built and saw.** Linearisation and de-linearisation of trees, the attention parser, and an F1 scorer. On a toy grammar, attention gave F1 89.8 vs 80.5 without it, with the gap largest on long sentences.

**Why it matters.** It showed that one general sequence model with attention can replace specialised algorithms. This "everything is sequence-to-sequence" view is how today's language models handle almost any task.

</details>

#### 031 · [Pointer Networks](05-Words-and-Sequences/031-Vinyals-et-al-2015-Pointer-Networks/) — Vinyals, Fortunato & Jaitly, 2015
- **The idea:** use attention to **point** at items in the input, so the output can only be input positions. Useful for sorting, or for drawing the outline around a set of points.
- **What we built:** the model, its baselines, and exact solvers for the geometry problems to compare against.
- **What we saw:** one model could sort lists of different lengths, including a length it had never seen.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** Some tasks need outputs that are **positions in the input**, for example "sort these numbers" (output the order of positions) or "which points form the outer boundary of this set?". The number of possible outputs changes with the input length, so a fixed output vocabulary doesn't work. Oriol Vinyals, Meire Fortunato and Navdeep Jaitly proposed **Pointer Networks**.

**Topic 1: attention as a pointer.** In normal attention (paper 028), the weights α are used to *blend* the input states. In a pointer network, the attention weights **are the output**: a probability distribution over input positions.
```
u_j  = vᵀ · tanh( W1·e_j + W2·d_i )        for each input position j
p(output = j) = softmax(u)_j
```
(e_j is the encoder state for input j; d_i is the decoder state at step i.)

**Worked example (sorting).** For the input [0.7, 0.2, 0.5], the correct output is the pointer sequence 2, 3, 1 (the smallest is at position 2, then position 3, then position 1). At step 1 the network should put most probability on position 2.

**Topic 2: the problems tested.**
- **Convex hull:** given points, list the ones on the outer boundary (imagine a rubber band around nails).
- **Delaunay triangulation:** connect points into "nice" triangles.
- **Travelling salesman (TSP):** the shortest tour through all cities.

**Topic 3: training.** Supervised: the correct answers come from exact algorithms (for small TSP) or approximate solvers. The network learns to imitate them.

**Topic 4: generalisation to other sizes.** Because the output space grows naturally with the input, one network can handle inputs of different lengths, and to some extent lengths **larger** than any seen in training (for convex hull it was trained on up to 50 points and tested on up to 500).

**Topic 5: limitations.** For TSP, the network's tours were close to the training solver's but not better. Performance degraded on much larger inputs than seen in training.

**What we built and saw.** The pointer network, a seq2seq baseline and exact solvers for the geometry tasks. One network learned to sort lists of several lengths, including an **unseen** length.

**Why it matters.** Pointing is used in summarisation (copying rare words from the source, as in "pointer-generator" networks), in combinatorial optimisation with neural networks, and in code models that copy variable names.

</details>

#### 032 · [Order Matters: Sequence to Sequence for Sets](05-Words-and-Sequences/032-Vinyals-et-al-2015-Order-Matters/) — Vinyals, Bengio & Kudlur, 2016
- **The idea:** when the input or output is really a *set*, the order you present it in still affects how well the model learns, and you can design for that.
- **What we built:** the paper's order-independent reader and its search over output orders.
- **What we saw:** on sorting, the paper's design (84%) beat the pointer network (75%), as the paper reports.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The question.** Seq2seq models read and write **sequences**. But many problems involve **sets**, where order is meaningless: a bag of numbers to sort, a set of points, a set of labels for an image. Oriol Vinyals, Samy Bengio and Manjunath Kudlur showed that **the order you choose still matters** for how well the model learns, and how to handle sets properly.

**Topic 1: input order matters.** If you feed a set to an RNN in different orders, the RNN produces different results, and some orders train better than others. Even for seq2seq, reversing the input helped (paper 027), which shows order affects learning.

**Topic 2: an order-independent reader (Read–Process–Write).**
- **Read:** embed each element of the set independently. No order is involved.
- **Process:** an LSTM with **no input** runs for T steps. At each step it uses attention over all the set's elements (content-based attention, which doesn't care about position) and updates its state. Because attention is a weighted sum, shuffling the elements doesn't change the result. The process block "thinks" about the whole set several times.
- **Write:** a pointer-network decoder (paper 031) outputs the answer.

**Topic 3: output order matters too.** When the output is a set (for example, all objects in an image), the training targets must be written in some order. Experiments on language modelling and parsing showed that some orders (such as depth-first vs breadth-first for parse trees, or natural vs reversed for sentences) are much easier to learn.

**Topic 4: searching over output orders.** Instead of fixing an order, during training let the model **pick** the order: for each example, consider several orderings of the target set and train on the one the model currently finds most likely (with some sampling early on to explore). The model settles on an order that is easy for it.

**Topic 5: sorting results.** The Read–Process–Write model sorted numbers better than a pointer network that reads the input sequentially, and more "process" steps helped.

**What we built and saw.** The Read–Process–Write model and the order search. On sorting, the paper's design reached 84% against 75% for a pointer network, matching the paper's direction.

**Why it matters.** It was an early step toward **permutation-invariant** models. Later "set transformers" and the treatment of tokens in transformers (which, without positional encoding, ignore order) follow the same idea.

</details>

#### 033 · [Neural Machine Translation in Linear Time (ByteNet)](05-Words-and-Sequences/033-Kalchbrenner-et-al-2016-ByteNet/) — Kalchbrenner et al., 2016
- **The idea:** process sequences with **convolutions** that have growing gaps ("dilations"), instead of step-by-step networks, so the work grows only in proportion to the length.
- **What we built:** the full model with its dilated convolutions.
- **What we saw:** dilation let it learn a pattern 10 steps back that a plain stack could not. **Surprise:** a number in the paper (a "receptive field" of 315) doesn't match its own formula; we get 373.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The goal.** RNN translators process words one after another, which is slow and makes long-range information travel through many steps. Nal Kalchbrenner and colleagues at DeepMind built **ByteNet**, a translator made of **convolutions** that works on raw characters (bytes) with computation that grows only **linearly** with length.

**Topic 1: dilated convolutions.** A normal convolution with kernel size 3 sees 3 neighbouring positions. Stacking layers grows the view slowly: L layers see about 2L + 1 positions. A **dilated** convolution skips positions: with dilation d, it looks at positions t, t−d and t−2d. Doubling the dilation each layer (1, 2, 4, 8, 16) grows the view **exponentially**. With kernel size 3, five layers see 1 + 2 × (1+2+4+8+16) = 63 positions.

**Topic 2: masked (causal) convolutions in the decoder.** When generating, the model must not peek at future characters. The decoder's convolutions look only at the current and past positions.

**Topic 3: stacking the decoder on the encoder.** The encoder processes the source characters. The decoder sits **on top of** the encoder's output, position by position. Because the target can be longer than the source, the paper uses **dynamic unfolding**: the target length is estimated as a·|source| + b, and the encoder representation is padded to that length.

**Topic 4: why linear time and no bottleneck.**
- Each layer does a fixed amount of work per position, so total work grows linearly with length (the attention models of the time cost source length × target length).
- No fixed-size summary vector: the decoder has direct access to the source representation at each position (**resolution preserving**).
- All positions of the encoder can be computed in **parallel** (unlike an RNN).

**Topic 5: results.** ByteNet achieved strong results on character-level language modelling (Hutter Prize Wikipedia) and character-level English→German translation, competitive with attention-based RNNs.

**What we built and saw.** The full model with dilated, masked residual blocks and dynamic unfolding. Dilation let the model learn a dependency 10 steps back that a plain (non-dilated) stack of the same depth could not. **A discrepancy:** the paper states a receptive field of 315 tokens, but its own formula for its architecture gives 373.

**Why it matters.** Dilated convolutions are also the core of **WaveNet** (speech synthesis). ByteNet's goals (parallel, linear-time, no bottleneck) were shared by the transformer, which arrived a year later and won out.

</details>

---

### Stage 06 · Transformers: attention is all you need

**The story.** In 2017 a team showed you could drop the step-by-step memory entirely and build a network out of attention alone. The **transformer** trains faster and scales further than anything before it, and it is the engine inside ChatGPT, BERT and image models alike. This stage covers the original design, ways to make it faster and handle longer inputs, and its use for understanding text (BERT) and for images (ViT).

#### 034 · [Attention Is All You Need (the Transformer)](06-Transformers/034-Vaswani-et-al-2017-Attention-Is-All-You-Need/) — Vaswani et al., 2017
- **The idea:** the **Transformer**. Every word looks at every other word through attention (many "heads" at once), and a trick called positional encoding tells it the word order.
- **What we built:** the complete model from scratch, with its training tricks and beam search.
- **What we saw:** our sizes are close to the paper's (63 vs 65 million parameters for the base model). On a reversing task it got 50 out of 50 right, with one attention head clearly learning the reversal.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The leap.** RNNs process words one at a time, which is slow to train and makes distant words hard to connect. Attention (paper 028) helped, but was attached to an RNN. Ashish Vaswani and colleagues at Google removed the RNN entirely: the **Transformer** uses only attention (plus simple per-word layers). It trains in parallel and connects any two words in one step.

**Topic 1: queries, keys and values.** Each word's vector is turned into three vectors by three learned matrices:
- a **query** (q): "what am I looking for?";
- a **key** (k): "what do I contain?";
- a **value** (v): "what will I hand over if chosen?".

Each word compares its query with every word's key; good matches get more weight, and the result is a weighted average of the values.

**Topic 2: scaled dot-product attention.**
```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```
**Why divide by √d_k?** A dot product of two random d-dimensional vectors has variance about d. With d = 64, the scores would be large, and softmax would become extremely peaked (nearly all weight on one word, with tiny gradients). Dividing by √64 = 8 keeps the scores in a sensible range.

**Worked example (d_k = 2).** The query q = [1, 0] is compared with the keys k1 = [1, 0] and k2 = [0, 1].
- Scores: 1/√2 = 0.707 and 0.
- Softmax weights: e^0.707 = 2.03 and e^0 = 1, so the weights are 0.67 and 0.33.
- If the values are v1 = 10 and v2 = 0, the output is 6.7.

**Topic 3: multi-head attention.** Instead of one attention, run **h = 8** smaller ones in parallel, each with its own Q/K/V matrices (d_model = 512 split into 8 heads of 64). One head can track grammar, another which noun a pronoun refers to, another nearby words. Their outputs are concatenated and mixed.

**Topic 4: positional encoding.** Attention by itself ignores order: "dog bites man" and "man bites dog" contain the same words. So a **position signal** is added to each word's vector, made of sine and cosine waves of different frequencies:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d)),   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
Each position gets a unique pattern, and relative offsets correspond to simple rotations, which helps the model reason about relative positions.

**Topic 5: the block.** Each layer has:
1. multi-head self-attention;
2. a **feed-forward network**, applied to each word separately (512 → 2048 → 512 with ReLU);
3. around each of these, a **residual connection** (paper 016) and **layer normalisation**.

The encoder stacks 6 such layers. The decoder stacks 6 layers that also have **masked** self-attention (a word can't look at future words) and **cross-attention** to the encoder's outputs.

**Topic 6: training details.**
- Adam with β2 = 0.98.
- A learning rate that **warms up** linearly for 4,000 steps and then decays as 1/√step.
- Dropout 0.1.
- **Label smoothing** 0.1: the target gives 90% to the right word and spreads 10% over the others, so the model doesn't become overconfident.

**Topic 7: results.** 28.4 BLEU on English→German and 41.8 on English→French, new state of the art, at a fraction of the training cost of previous models.

**What we built and saw.** The complete model from scratch (63M parameters for our base model vs the paper's 65M, a small difference in counting embeddings), with warm-up, label smoothing and beam search. On a reversal task it got 50/50 correct, and one head's attention clearly formed the reversal pattern.

**Why it matters.** This is the architecture behind BERT, GPT, ChatGPT, LLaMA, Vision Transformers, Whisper and almost every frontier AI model since 2018.

</details>

#### 035 · [Fast Transformer Decoding: One Write-Head is All You Need](06-Transformers/035-Shazeer-2019-Multi-Query-Attention/) — Shazeer, 2019
- **The idea:** **multi-query attention**: share one set of "keys" and "values" across all heads, which makes text generation much faster with little loss in quality.
- **What we built:** standard, multi-query and the in-between "grouped-query" attention, with the memory cache used during generation.
- **What we saw:** the cache became 8 times smaller and generation on a laptop processor got faster (6.0 → 2.2 milliseconds per step).

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** Transformers **generate** text one token at a time. To avoid recomputing everything, the keys and values of all previous tokens are stored: the **KV cache**. At every new token, the model must read the whole cache from memory. Noam Shazeer noticed that for generation, the bottleneck is **memory bandwidth** (moving the cache), not arithmetic.

**Topic 1: the size of the KV cache.** For every layer, every head and every previous token, store one key and one value vector:
```
cache size = 2 × layers × heads × d_head × tokens × bytes per number
```
Example: 32 layers, 32 heads, d_head = 128, 4,096 tokens and 2 bytes per number. That gives about 2 GB **per sequence**.

**Topic 2: multi-query attention (MQA).** Keep **many query heads** (each head still asks its own question), but share **one** key and **one** value projection across all heads. The cache shrinks by a factor equal to the number of heads (32× in the example). Less memory to read means much faster decoding, and larger batches fit in memory.

**Topic 3: the quality cost.** Sharing keys and values loses a bit of flexibility. In the paper, quality dropped only slightly (a small rise in perplexity and a small BLEU drop on translation), while decoder speed improved greatly.

**Topic 4: incremental decoding and arithmetic intensity.** During generation, each step does little arithmetic per byte loaded (low **arithmetic intensity**), so the chip waits on memory. MQA raises the ratio of arithmetic to memory traffic.

**Topic 5: the middle ground, grouped-query attention (GQA).** A later paper (Ainslie et al., 2023) groups heads: for example 32 query heads share 8 key/value heads. This gives most of MQA's speed with almost all of the quality of full multi-head attention. LLaMA-2 70B and many current models use GQA.

**What we built and saw.** Multi-head, multi-query and grouped-query attention with a KV cache for incremental decoding. With 8 heads, the cache became **8× smaller**, and generation on a laptop CPU went from 6.0 to 2.2 milliseconds per step.

**Why it matters.** MQA/GQA is one of the main reasons modern chatbots can serve millions of users affordably. PaLM used MQA, and LLaMA-2/3 and Mistral use GQA.

</details>

#### 036 · [Generating Long Sequences with Sparse Transformers](06-Transformers/036-Child-et-al-2019-Sparse-Transformers/) — Child, Gray, Radford & Sutskever, 2019
- **The idea:** let each position attend to only a structured subset of others, so very long sequences become affordable.
- **What we built:** both of the paper's sparse attention patterns.
- **What we saw:** 11 to 32 times fewer pairs to compute; one pattern copied repeating structure as well as full attention, while the other struggled.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** Full attention compares every token with every other. With n tokens that is **n²** pairs: 1,000 tokens give 1 million pairs, and 16,000 tokens give 256 million. Long documents, images (as sequences of pixels) or audio quickly become unaffordable. Rewon Child, Scott Gray, Alec Radford and Ilya Sutskever proposed **sparse** attention patterns.

**Topic 1: factorised attention.** Each token attends to only a structured subset of earlier tokens, chosen so that **any token can reach any other within two steps**. Information can still flow everywhere, through an intermediate token.

**Topic 2: the strided pattern.** With stride l ≈ √n:
- one head attends to the **previous l tokens** (local neighbourhood);
- another head attends to **every l-th token** (one per "row").

For images written row by row, this is like looking along your row and down your column.

**Topic 3: the fixed pattern.** Split the sequence into blocks of length l:
- one head attends **within your block**;
- another attends to a few designated **"summary" positions** at the end of each block.

Information flows block → summary → anywhere. This suits text, where the "rows" of the strided pattern don't mean anything.

**Topic 4: the savings.** Each token attends to about 2√n others instead of n, so the total work is about **n·√n** instead of n². For n = 4,096: n² = 16.8 million pairs, while n√n ≈ 262,000, about **64× fewer**.

**Topic 5: making very deep, long models train.**
- **Recompute** attention during the backward pass instead of storing it, to save memory.
- Place layer norm before each sub-layer (**pre-LN**), and scale the initial weights by 1/√(2N) for N layers.
- Write custom GPU kernels that compute only the needed blocks.

**Topic 6: results.** State-of-the-art density modelling for images (CIFAR-10, ImageNet 64×64), text (enwik8) and raw audio, with sequences up to tens of thousands of steps.

**What we built and saw.** Both patterns (with a check that every pair is reachable in two hops), blocked attention without building the n×n matrix, pre-LN blocks and recomputation. Our patterns used 11–32× fewer pairs. On a task where rows repeat, the strided pattern copied the structure as well as full attention, while the fixed pattern struggled. Patterns must match the data's structure.

**Why it matters.** Sparse and local attention appear in many long-context models (for example sliding-window attention in Mistral, and the patterns in GPT-3's alternating layers).

</details>

#### 037 · [BERT](06-Transformers/037-Devlin-et-al-2019-BERT/) — Devlin, Chang, Lee & Toutanova, 2019
- **The idea:** **BERT** learns language by filling in hidden words using context from **both** sides, then is fine-tuned for many tasks.
- **What we built:** the full recipe: word pieces, hiding 15% of words, the "is this the next sentence?" task, and heads for classification and question answering.
- **What we saw:** when the missing word depends on what comes after it, BERT-style training recovered it 100% of the time, against 9.5% for a left-to-right model.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before BERT.** GPT (paper 048) learned language by predicting the **next** word, so it only used **left** context. But understanding a word often needs both sides: in "The **bank** of the river", "river" (on the right) tells you which "bank" is meant. Jacob Devlin and colleagues at Google created **BERT** (Bidirectional Encoder Representations from Transformers).

**Topic 1: masked language modelling (MLM).** You can't simply "predict the next word using both sides", because the model would see the answer. So BERT hides words and predicts them:
- choose **15%** of the tokens;
- of those, replace 80% with a special [MASK] token, replace 10% with a random word, and leave 10% unchanged;
- predict the original word at each chosen position.

**Why the 80/10/10 split?** [MASK] never appears when BERT is later used on real tasks. If it were the only signal, the model might only learn useful representations at [MASK] positions. The random and unchanged cases force it to build a good representation of **every** word.

**Topic 2: next sentence prediction (NSP).** Given two sentences A and B, predict whether B really followed A in the text (50% of the time) or is a random sentence. This was meant to teach relationships between sentences, though later work (RoBERTa) found NSP unnecessary.

**Topic 3: input format.**
```
[CLS]  my dog is cute  [SEP]  he likes play ##ing  [SEP]
```
- **WordPiece** tokens (about 30,000 sub-word pieces; "playing" becomes "play" + "##ing", so rare words are handled).
- Each token's input is the sum of a **token embedding**, a **segment embedding** (sentence A or B) and a **position embedding**.
- The final vector of [CLS] serves as a summary of the whole input.

**Topic 4: sizes.**
- BERT-Base: 12 layers, hidden size 768, 12 heads, 110M parameters.
- BERT-Large: 24 layers, 1,024 hidden, 16 heads, 340M parameters.

Both were pre-trained on BooksCorpus and English Wikipedia (about 3.3 billion words).

**Topic 5: fine-tuning.** Add one small layer on top and train everything briefly on the target task.
- **Classification** (sentiment, entailment): use the [CLS] vector.
- **Question answering** (SQuAD): for each token, predict the probability it is the **start** or the **end** of the answer span.
- **Tagging** (named entities): classify each token's vector.

**Topic 6: results.** BERT set new records on 11 NLP tasks, including the GLUE benchmark and SQuAD, often by large margins.

**What we built and saw.** The whole recipe: WordPiece, masking, NSP, and heads for classification and question answering. On a task where the hidden word can only be determined from the words **after** it, BERT-style training recovered it 100% of the time, against 9.5% for a left-to-right model.

**Why it matters.** BERT dominated language understanding from 2018 to about 2021, and its descendants are still widely used for search, classification and embeddings (for example sentence-embedding models used in retrieval, paper 070).

</details>

#### 038 · [An Image is Worth 16x16 Words (Vision Transformer)](06-Transformers/038-Dosovitskiy-et-al-2021-ViT/) — Dosovitskiy et al., 2021
- **The idea:** cut an image into small patches, treat each patch like a word, and use a plain transformer.
- **What we built:** the Vision Transformer and the paper's tools for looking inside it.
- **What we saw:** with very little data, a convolutional network won easily (100% vs 40.8%). That matches the paper's point that transformers need lots of data because they have fewer built-in assumptions about images.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The question.** Convolutional networks (Stage 03) were the only serious choice for images. Alexey Dosovitskiy and colleagues asked: does a **plain transformer**, with almost no image-specific design, work for images?

**Topic 1: images as sequences of patches.** Cut a 224×224 image into **16×16-pixel patches**: 14 × 14 = **196 patches**. Each patch has 16 × 16 × 3 colour values = 768 numbers. Flatten each patch and multiply it by a learned matrix to get a token vector. An image is now a "sentence" of 196 "words", hence the title "An Image is Worth 16x16 Words".

**Topic 2: class token and positions.**
- Add a learned **[class]** token at the front (as in BERT's [CLS]). Its final vector is used for classification.
- Add learned **position embeddings** so the model knows where each patch came from.

**Topic 3: the encoder.** A standard transformer encoder (paper 034), with layer norm before each sub-layer. Sizes: ViT-Base (86M parameters), ViT-Large (307M) and ViT-Huge (632M).

**Topic 4: inductive bias and data.** A CNN has built-in assumptions (**inductive biases**):
- **locality:** nearby pixels matter together;
- **translation equivariance:** a cat is a cat anywhere in the image.

ViT has almost none, so it must **learn** these from data.
- Trained only on ImageNet (1.3M images), ViT was a bit **worse** than ResNets of similar size.
- Pre-trained on huge datasets (ImageNet-21k with 14M images, or JFT-300M with 300M), it **beat** the best CNNs, reaching 88.55% top-1 on ImageNet while using less compute to pre-train.

**Topic 5: what ViT learns.**
- The learned position embeddings rediscover the 2-D grid: nearby patches have similar embeddings.
- **Attention distance:** some heads attend globally even in the first layers (a CNN can't do this early), while others stay local, like convolutions.
- The first-layer filters look like the edge and colour detectors CNNs learn.

**Topic 6: fine-tuning at higher resolution.** Images can be fine-tuned at larger sizes (more patches). The position embeddings are interpolated in 2-D to the new grid.

**What we built and saw.** ViT, plus tools to plot position-embedding similarity and attention distance. With very little data, a small CNN won easily (100% vs 40.8%). That is the paper's point: without lots of data, the CNN's built-in assumptions win.

**Why it matters.** ViT unified vision and language architectures. CLIP (053), DALL·E-style models and image-understanding chatbots use transformer image encoders descended from ViT.

</details>

---

### Stage 07 · Generative models: networks that create

**The story.** Instead of labelling data, these models *create* it: new faces, digits, pictures from a sentence. This stage covers the main families: VAEs (compress and rebuild), GANs (a forger against a detective), pixel-by-pixel models, invertible "flows", and finally DALL·E, which draws images from text descriptions.

#### 039 · [Auto-Encoding Variational Bayes (VAE)](07-Generative-Models/039-Kingma-Welling-2014-VAE/) — Kingma & Welling, 2014
- **The idea:** the **VAE** squeezes data into a small "code" and learns to rebuild it. A clever "reparameterisation trick" makes this trainable.
- **What we built:** the VAE, both of the paper's estimators, and the older methods it beat.
- **What we saw:** the trick made learning signals 8 to 12 times less noisy, and the VAE beat the older "wake-sleep" method on digits.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** We want a model that can **generate** new data (for example new digit images) and learn a compact **code** (a few numbers) that describes each data point. Models with hidden "latent" variables z do this: first pick a code z, then generate x from z. But training them requires the **posterior** p(z|x) ("which codes could have produced this image?"), which is intractable to compute. Diederik Kingma and Max Welling made this practical with the **Variational Autoencoder** (VAE).

**Topic 1: encoder and decoder.**
- The **encoder** network q(z|x) takes an image and outputs a **mean μ** and **standard deviation σ** for each code dimension. It is a guess at the posterior.
- The **decoder** network p(x|z) takes a code and outputs a distribution over images.
- The prior p(z) is a standard bell curve N(0, 1).

**Topic 2: the ELBO (evidence lower bound).** We can't maximise log p(x) directly, but we can maximise a lower bound:
```
ELBO = E_q[ log p(x|z) ]  −  KL( q(z|x) || p(z) )
```
- The first term is **reconstruction**: codes from the encoder should let the decoder rebuild x.
- The second term is a **regulariser**: the encoder's distribution should stay close to the prior. This keeps the code space smooth and filled in, so random codes decode into sensible images.

**Topic 3: the KL term in closed form.** For Gaussians, the KL term has an exact formula:
```
KL = −½ Σ_j ( 1 + log σ_j² − μ_j² − σ_j² )
```
Example for one dimension with μ = 1, σ = 1: KL = −½(1 + 0 − 1 − 1) = 0.5. With μ = 0, σ = 1 it is 0, meaning identical to the prior.

**Topic 4: the reparameterisation trick.** To train the encoder we need gradients through a **random sample** z ~ N(μ, σ²). Sampling isn't differentiable. The trick is to write the sample as:
```
z = μ + σ · ε,     ε ~ N(0, 1)
```
The randomness is now in ε, which doesn't depend on the parameters. z is a smooth function of μ and σ, so gradients flow through ordinary backpropagation. The older alternative (the "score function" or REINFORCE estimator) works but has very high variance.

**Topic 5: two estimators.**
- **Estimator A** samples everything, including the KL term.
- **Estimator B** uses the exact KL formula, so it is less noisy.

The paper calls the method **SGVB/AEVB** (stochastic gradient variational Bayes / auto-encoding variational Bayes).

**Topic 6: what you get.** A 2-D code space where moving smoothly changes the digit's style and identity (the famous "manifold" plots), and the ability to sample new images by decoding random z.

**What we built and saw.**
- Both estimators, and the older baselines (wake-sleep and Monte Carlo EM).
- The reparameterised gradients were **8–12× less noisy** than score-function gradients.
- Estimator B was 4× less noisy than A.
- The VAE beat wake-sleep on a slice of MNIST.

**Why it matters.** VAEs are a core generative model. The reparameterisation trick is used throughout machine learning, and the VAE's image compressor is a key part of Stable Diffusion (latent diffusion).

</details>

#### 040 · [Improved Variational Inference with Inverse Autoregressive Flow](07-Generative-Models/040-Kingma-et-al-2016-Inverse-Autoregressive-Flow/) — Kingma et al., 2016
- **The idea:** make a VAE's guesses about the hidden code more flexible by passing them through a chain of invertible steps (a "flow").
- **What we built:** the flow and its building blocks.
- **What we saw:** the simple version was stuck with a fixed error, while the flow brought it close to zero.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** In a standard VAE (paper 039), the encoder's guess q(z|x) is a simple Gaussian with independent dimensions, like a round or axis-aligned blob. But the true posterior may be curved, skewed or correlated (shaped like a banana). The mismatch makes the ELBO looser and the model worse. Diederik Kingma and colleagues proposed **Inverse Autoregressive Flow** (IAF) to make q flexible.

**Topic 1: normalising flows.** Start with a simple random sample z₀ (a Gaussian). Pass it through a chain of **invertible** transformations to get z_T. If a transformation stretches space, the probability density thins out there; if it squeezes, the density increases. The exact change in density is given by the **determinant of the Jacobian** (how much the transformation scales volume):
```
log q(z_T) = log q(z_0) − Σ_t log | det( ∂z_t / ∂z_{t−1} ) |
```
The chain can be built from simple pieces, but the determinant must be **cheap** to compute.

**Topic 2: autoregressive transformations.** Make each new dimension depend only on the **earlier** dimensions of the input:
```
z_t[i] = μ_i(z_{t−1}[1..i−1]) + σ_i(z_{t−1}[1..i−1]) · z_{t−1}[i]
```
The Jacobian is then **triangular**, and the determinant of a triangular matrix is the product of its diagonal: here, the product of the σ_i. So log det = Σ log σ_i. That is cheap.

**Topic 3: why "inverse" matters (speed).** With this IAF direction, all μ_i and σ_i come from the **input** z_{t−1}, which is fully known. So all dimensions can be computed **in parallel** in one pass. That makes sampling, which the VAE needs at every training step, fast.

**Topic 4: MADE, a network with masks.** To compute all μ_i and σ_i in one network while guaranteeing that output i depends only on inputs 1..i−1, IAF uses **masked** layers (MADE): some weights are forced to zero so information can't flow "forward" in the ordering.

**Topic 5: results.** IAF made VAEs much better at modelling images: on CIFAR-10 it was competitive with the best pixel-by-pixel models, while generating images far faster.

**What we built and saw.** IAF steps with masked networks, the log-determinant bookkeeping and a VAE using them. In our toy, a plain Gaussian encoder was stuck with a fixed gap (it simply cannot represent the true posterior's shape), while adding IAF steps drove the gap close to zero.

**Why it matters.** Normalising flows are a whole family of generative models (Glow, paper 046), and flexible posteriors are used in many probabilistic models. The same autoregressive idea underlies "masked" designs used elsewhere.

</details>

#### 041 · [Variational Lossy Autoencoder](07-Generative-Models/041-Chen-et-al-2017-Variational-Lossy-Autoencoder/) — Chen et al., 2017
- **The idea:** control **what** information the code stores (for example, the overall shape but not fine texture) by limiting what the decoder can see.
- **What we built:** decoders that see only a small window, and the paper's information accounting.
- **What we saw:** in a toy with one important global fact, the limited decoder kept exactly that fact in the code, as the paper predicts.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** If you give a VAE a very powerful decoder, such as a PixelCNN that predicts each pixel from the previous ones (paper 045), the decoder can model images **all by itself**. The VAE then learns to **ignore** the code z: the KL term pushes z toward carrying nothing. This is called **posterior collapse**. Xi Chen and colleagues explained why and how to control it.

**Topic 1: the "bits-back" view.** Think of the ELBO as the cost of sending data in bits: the cost of the code z (the KL term) plus the cost of x given z. Putting information in z costs bits. The model only does that if it **saves more bits** in describing x than it costs. If the decoder can already predict x well without z, then z isn't worth it, and the model leaves it empty.

**Topic 2: deciding what goes in the code.** This explanation gives a design tool. Anything the decoder **can** model by itself stays out of z; anything it **cannot** goes into z. So restrict the decoder:
- give the PixelCNN decoder only a **small local window** of previous pixels;
- it can then model fine local texture by itself;
- but it **cannot** see the global shape, so global information (which digit, overall layout) must go into z.

The result is a **lossy** code: it keeps the global content and discards local detail the decoder will fill in. Hence the name "Variational Lossy Autoencoder".

**Topic 3: a better prior with autoregressive flows.** Instead of a simple Gaussian prior on z, use an **autoregressive flow prior** (related to IAF, paper 040). The paper shows that a flow on the prior is equivalent to a flow on the posterior, but is more efficient and gives better results.

**Topic 4: results.** State-of-the-art likelihoods at the time on MNIST, OMNIGLOT and Caltech-101 silhouettes, and competitive results on CIFAR-10. Samples decoded from the same code kept the same global structure with different local details.

**What we built and saw.** Decoders with limited windows and the bits-back accounting. In a toy where each sequence has one hidden global fact plus local noise, the limited decoder caused the code to store **exactly** that global fact, as the paper predicts. An unlimited decoder led to posterior collapse.

**Why it matters.** Controlling what a latent code captures is central to representation learning and compression, and modern latent-diffusion systems have a similar split: a code for content, with a decoder for detail.

</details>

#### 042 · [Generative Adversarial Nets (GAN)](07-Generative-Models/042-Goodfellow-et-al-2014-GAN/) — Goodfellow et al., 2014
- **The idea:** a **forger** network makes fake samples and a **detective** network tries to spot them. Each improves by competing with the other.
- **What we built:** the original GAN and the paper's way of measuring it.
- **What we saw:** the theory holds exactly, and a one-dimensional GAN learned its target. We also reproduced two famous problems: training swings back and forth, and the forger can collapse to producing only a few kinds of output.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The idea.** Ian Goodfellow and colleagues proposed training a generator through a **game**:
- the **generator** G is a forger that turns random noise z into fake samples;
- the **discriminator** D is a detective that outputs the probability that a sample is real.

D learns to tell real from fake; G learns to fool D. If both become perfect, G's fakes are indistinguishable from real data.

**Topic 1: the objective.**
```
min_G  max_D   V(D, G) = E_real[ log D(x) ]  +  E_noise[ log(1 − D(G(z))) ]
```
D wants D(real) close to 1 and D(fake) close to 0. G wants D(fake) close to 1.

**Topic 2: the best possible detective.** For a fixed G, the optimal D at any point x is:
```
D*(x) = p_data(x) / ( p_data(x) + p_G(x) )
```
Example: where real data has density 0.3 and the forger's density is 0.1, D* = 0.3/0.4 = 0.75, so "probably real". When the forger is perfect (p_G = p_data), D* = 0.5 everywhere: a coin flip.

**Topic 3: what the game minimises.** Plugging D* in, the generator's objective becomes:
```
C(G) = −log 4 + 2 · JSD( p_data || p_G )
```
JSD is the **Jensen–Shannon divergence**, a measure of how different two distributions are (0 when identical). So the game's global minimum is exactly when the generator matches the data.

**Topic 4: practical training.**
- Alternate k steps of D (often k = 1) with one step of G.
- **Non-saturating trick:** early on, D easily rejects G's bad fakes, so log(1 − D(G(z))) is flat and gives G almost no gradient. So G instead **maximises log D(G(z))**, which has the same goal but strong gradients when G is bad.

**Topic 5: evaluation.** GANs don't give likelihoods directly. The paper estimated them with **Parzen windows** (place a small Gaussian blob on each generated sample and measure test data under the resulting density). This is a rough method, later replaced by FID and other metrics.

**Topic 6: known problems.**
- **Mode collapse:** G produces only a few kinds of output that fool D (for example only one digit).
- **Oscillation:** G and D chase each other in circles instead of settling.

**What we built and saw.** The original GAN, the optimal-D formula (checked exactly), and the Parzen evaluation. A 1-D GAN learned its target distribution. We also reproduced oscillation and mode collapse on a mixture of several bumps.

**Why it matters.** GANs dominated realistic image generation from about 2014 to 2021 (StyleGAN faces, "deepfakes"). The adversarial idea appears in many areas. Diffusion models have since largely replaced them for image generation.

</details>

#### 043 · [InfoGAN](07-Generative-Models/043-Chen-et-al-2016-InfoGAN/) — Chen et al., 2016
- **The idea:** add "control knobs" to a GAN that end up meaning something (like digit type or slant), without any labels.
- **What we built:** InfoGAN with its information-based training.
- **What we saw:** the knobs grouped the data correctly 100% of the time without labels, against 26% for a plain GAN.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** A GAN's noise input z is unstructured: changing one number of z changes the image in some arbitrary, tangled way. We would like **meaningful knobs**, such as "digit identity", "slant" and "stroke width", learned **without any labels**. Xi Chen and colleagues proposed **InfoGAN**.

**Topic 1: split the input.** The generator's input becomes (z, c):
- z is ordinary noise;
- c is a **latent code**, for example one categorical variable with 10 values and two continuous variables between −1 and 1.

**Topic 2: the problem with just adding c.** A normal GAN could simply **ignore** c: nothing forces it to use it.

**Topic 3: mutual information.** InfoGAN adds a goal: the code c should be **recoverable** from the generated image. In information theory, the **mutual information** I(c; G(z,c)) measures how much knowing the image tells you about c. It is large if you can read c off the image, and zero if the image says nothing about c. InfoGAN maximises it:
```
min_G max_D   V(D, G)  −  λ · I( c ; G(z, c) )
```

**Topic 4: a practical lower bound.** Mutual information is hard to compute directly. The paper adds an auxiliary network Q(c|x) (sharing most layers with D) that tries to **predict c from the image**. How well Q predicts gives a lower bound on I, which can be maximised by gradient descent. In practice:
- for the categorical code, this is a classification loss on Q;
- for the continuous codes, a Gaussian log-likelihood.

**Topic 5: what emerges.** On MNIST, with no labels:
- the 10-way categorical code captured **digit identity** (it matched the true class with about 95% accuracy);
- one continuous code captured **rotation/slant**;
- another captured **stroke width**.

On faces, codes captured pose, lighting and so on.

**Topic 6: why it works.** The easiest way to make c recoverable is to tie it to a **major source of variation** in the data, such as the digit class. Using the codes for small, hard-to-see details would make them hard to recover.

**What we built and saw.** InfoGAN with categorical and continuous codes and the Q network. In our toy, the categorical code grouped the data into its true clusters **100%** correctly without labels, against 26% for a plain GAN's input.

**Why it matters.** InfoGAN was an influential early example of **disentangled representation learning**: finding interpretable factors of variation without supervision.

</details>

#### 044 · [Wasserstein GAN](07-Generative-Models/044-Arjovsky-et-al-2017-Wasserstein-GAN/) — Arjovsky, Chintala & Bottou, 2017
- **The idea:** use a different way of measuring how far apart two distributions are (the "earth mover's distance"), which gives smoother learning signals and steadier GAN training.
- **What we built:** all the distance measures, the new critic network and its training.
- **What we saw:** the old measure gave a flat, useless signal while the new one gave a steady slope, and the new GAN smoothly moved to the right answer.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** GANs were notoriously unstable. Martin Arjovsky, Soumith Chintala and Léon Bottou traced this to the distance the GAN implicitly minimises (Jensen–Shannon, paper 042), and proposed a better distance.

**Topic 1: when distributions don't overlap.** Real images lie on a thin "surface" in a huge space, and so do generated images. Two thin surfaces usually **don't overlap** at all. Then:
- the **JS divergence** is constant (log 2) no matter how close the surfaces are;
- the **KL divergence** is infinite.

Neither tells the generator which way to move. So the gradient is zero or useless.

**Worked example.** Real data is a point at 0; fake data is a point at θ.
- JS = log 2 for any θ ≠ 0. It is flat, so it carries no information.
- KL = ∞ for any θ ≠ 0.
- The **Wasserstein distance** = |θ|. It is smooth and its slope points straight toward 0.

**Topic 2: the Earth Mover's (Wasserstein-1) distance.** Picture one distribution as piles of earth and the other as holes. The EM distance is the **minimum total work** (amount × distance moved) to fill the holes with the earth. It stays meaningful even when the two don't overlap: farther apart means more work.

**Topic 3: computing it with a critic.** Computing the EM distance directly is hard. The **Kantorovich–Rubinstein duality** gives:
```
W(p_real, p_fake) = max over 1-Lipschitz functions f   of   E_real[ f(x) ] − E_fake[ f(x) ]
```
A function is **1-Lipschitz** if its slope is never steeper than 1. So train a network f (now called the **critic**, since it outputs a score, not a probability) to maximise this difference while keeping it "not too steep".

**Topic 4: enforcing the slope limit.**
- WGAN **clips** every critic weight to [−0.01, 0.01] after each update. This is crude but works.
- A later paper (WGAN-GP) replaced clipping with a **gradient penalty**.

**Topic 5: training recipe.**
- Train the critic **5 times** per generator step (a well-trained critic gives better gradients, unlike a too-good GAN discriminator).
- Use RMSProp with a small learning rate.
- No log or sigmoid at the critic's output.

**Topic 6: a meaningful loss curve.** The critic's estimate of W decreases as the samples get better. For the first time, the GAN loss **correlated with image quality**, which makes debugging much easier. The paper also reports less mode collapse.

**What we built and saw.** The JS, KL and EM distances on toy examples, the critic with clipping and the training loop. JS gave a flat signal while EM gave a smooth slope, and the WGAN generator moved smoothly to the right place.

**Why it matters.** WGAN (and WGAN-GP) made GAN training far more reliable, and optimal transport distances became an important tool across machine learning.

</details>

#### 045 · [PixelCNN++](07-Generative-Models/045-Salimans-et-al-2017-PixelCNN-plus-plus/) — Salimans et al., 2017
- **The idea:** generate an image one pixel at a time, each pixel predicted from the ones before it, with several improvements over earlier versions.
- **What we built:** the paper's pixel probability model and its network design.
- **What we saw:** the simpler model memorised its training images, while the improved one generalised better.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The idea.** An image can be generated **one pixel at a time**, with each pixel predicted from all the pixels before it (left to right, top to bottom):
```
p(image) = p(x_1) · p(x_2 | x_1) · p(x_3 | x_1, x_2) · …
```
This gives an **exact** likelihood, unlike GANs. **PixelCNN** computes these predictions with **masked convolutions** (a filter that can't see the current or future pixels). Tim Salimans and colleagues at OpenAI improved it into **PixelCNN++**.

**Topic 1: a better output distribution.** The original PixelCNN treated each colour value (0–255) as one of **256 unrelated classes**. That ignores the fact that 127 is close to 128, and it needs lots of data to learn every value separately. PixelCNN++ uses a **discretised mixture of logistic distributions**: a few smooth bumps over the number line, with the probability of value v being the area of the bumps between v − 0.5 and v + 0.5. The edge values 0 and 255 get all the probability beyond them. Benefits: fewer parameters, smoother predictions and faster learning.

**Topic 2: conditioning on whole pixels.** Rather than modelling red, green and blue as three separate steps with complicated masks, the model predicts a pixel's three colours together. The green prediction depends linearly on red, and blue on red and green. This is simpler and faster.

**Topic 3: downsampling instead of dilation.** To see far across the image, the network uses strided convolutions to shrink the image and then expands it again, like a U-Net. This is cheaper than many dilated layers.

**Topic 4: shortcut connections.** Skip connections from the shrinking path to the expanding path recover detail lost during downsampling.

**Topic 5: dropout.** Large PixelCNNs **overfit** CIFAR-10 (the training likelihood keeps improving while the test likelihood gets worse). Dropout in the residual blocks fixes this.

**Topic 6: measuring with bits per dimension.** The average −log₂ p per colour value. Lower is better; 8 bits would be "no better than random". PixelCNN++ reached **2.92 bits/dim** on CIFAR-10, the state of the art at the time.

**What we built and saw.** The discretised logistic mixture likelihood (checked to sum to 1 over all 256 values) and the network design. In our toy, the 256-way softmax version memorised the training images, while the improved model generalised better to test images.

**Why it matters.** Autoregressive modelling, "predict the next piece from the previous ones", is exactly how GPT generates text. PixelCNN++'s logistic mixture output was also used in WaveNet-style audio models.

</details>

#### 046 · [Glow](07-Generative-Models/046-Kingma-Dhariwal-2018-Glow/) — Kingma & Dhariwal, 2018
- **The idea:** a fully **invertible** network: you can turn an image into a code and back again exactly. That makes exact probability calculations possible.
- **What we built:** all of Glow's building blocks.
- **What we saw:** the network was exactly invertible, and its fast probability calculation matched the slow exact one to six decimal places.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The idea.** A **flow** model is a network that is exactly **invertible**: image → code and code → image, with no information lost. Because it is invertible, the exact probability of an image can be computed with the change-of-variables formula (paper 040):
```
log p(x) = log p(z) + Σ over layers  log | det(Jacobian of the layer) |
```
Diederik Kingma and Prafulla Dhariwal built **Glow**, which made flows generate realistic high-resolution faces.

**Topic 1: actnorm.** Like batch norm (paper 012), but with a **per-channel scale and bias** that are initialised from the first batch of data (so the outputs start with mean 0 and variance 1) and then trained as normal parameters. It works with batch size 1, which is important for big images. Its log-determinant is simply height × width × Σ log|scale|.

**Topic 2: the invertible 1×1 convolution.** Earlier flows **shuffled** or **reversed** the channels between layers so that every channel eventually gets transformed. Glow **learns** this mixing: a 1×1 convolution is just a C×C matrix W applied at every pixel. Its log-determinant is height × width × log|det W|. To make det W cheap, W is stored as an **LU decomposition** (permutation × lower triangular × upper triangular). The determinant is then the product of the diagonal entries.

**Topic 3: the affine coupling layer.** Split the channels into two halves, x_a and x_b:
```
y_a = x_a
y_b = x_b ⊙ s(x_a) + t(x_a)
```
s and t can be **any** neural network: they don't need to be invertible. Inverting is easy because y_a = x_a lets you recompute s and t:
```
x_b = (y_b − t(y_a)) / s(y_a)
```
The log-determinant is Σ log|s|.

**Topic 4: multi-scale architecture.** After some steps, half of the dimensions are **factored out** directly into the code, and the rest continue through more steps. Coarse features get processed more.

**Topic 5: what you can do with it.**
- **Exact likelihood:** compare models in bits/dim.
- **Sampling with temperature:** sample z with a smaller standard deviation (for example 0.7) for cleaner, more typical images.
- **Interpolation:** blend two faces' codes for a smooth morph.
- **Attribute manipulation:** find the average code difference between "smiling" and "not smiling" faces and add it to any face.

**What we built and saw.** Actnorm, the invertible 1×1 convolution with LU, affine coupling and a multi-scale flow. The network inverted **exactly** (to rounding error), and the fast log-determinant matched a slow exact computation to six decimal places.

**Why it matters.** Glow showed flows could produce realistic images with exact likelihoods. Coupling layers and invertible networks are used in audio synthesis (WaveGlow), scientific modelling and probabilistic inference.

</details>

#### 047 · [Zero-Shot Text-to-Image Generation (DALL·E)](07-Generative-Models/047-Ramesh-et-al-2021-DALL-E/) — Ramesh et al., 2021
- **The idea:** **DALL·E** turns images into "visual words" and trains one transformer on text followed by image, so it can draw a picture from a caption.
- **What we built:** the image-to-token compressor, the text-plus-image transformer, and the trick of generating many images and keeping the best.
- **What we saw:** it composed colours and positions it had never seen together about half the time ("inconsistent", as the paper honestly says), and picking the best of several samples raised accuracy from 83% to 100%.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The idea.** Generate an image from a text description, even for unusual combinations ("an armchair in the shape of an avocado"). Aditya Ramesh and colleagues at OpenAI treated text and image as **one long sequence of tokens** and trained a single transformer to predict it.

**Topic 1: stage 1, turning images into tokens (dVAE).** A 256×256 image has 196,608 numbers, far too many tokens. A **discrete VAE** compresses each image into a **32×32 grid of tokens**, each chosen from a **codebook of 8,192** "visual words". A decoder can rebuild a slightly blurry version of the image from these 1,024 tokens. Choosing discrete tokens isn't differentiable, so training uses the **Gumbel-softmax relaxation**: a smooth approximation of picking one option, made gradually sharper during training.

**Topic 2: stage 2, a transformer over text + image.**
- Up to **256 text tokens** (BPE pieces of the caption) are followed by **1,024 image tokens**, giving one sequence of 1,280 tokens.
- A **12-billion-parameter** decoder transformer is trained, like GPT, to predict the next token.
- It was trained on 250 million text–image pairs from the internet.
- It uses sparse attention patterns (paper 036) for the image part.

**Topic 3: generating.** Feed in the caption's tokens, sample the 1,024 image tokens one by one, then decode them into pixels with the dVAE decoder.

**Topic 4: reranking with CLIP.** Sampling is random, so some images are better than others. DALL·E generates **many** images per caption (up to 512) and uses **CLIP** (paper 053) to score how well each image matches the caption, keeping the best. This greatly improves quality.

**Topic 5: zero-shot results.** Without training on the MS-COCO benchmark, human judges preferred DALL·E's images over a previous model trained on COCO for realism and caption match in most comparisons. It could **combine concepts** (objects, colours, styles, text in images) in new ways, but inconsistently. The paper honestly notes that results vary and often need reranking.

**What we built and saw.** A small image tokeniser, a text+image transformer and best-of-n reranking. In our small coloured-shape world it composed colour/position combinations it had never seen about half the time (inconsistent, as the paper says), and picking the best of several samples raised accuracy from 83% to 100%.

**Why it matters.** DALL·E launched the text-to-image era. Later systems (DALL·E 2/3, Stable Diffusion, Midjourney) moved to diffusion models, but "images as tokens", reranking and text conditioning remain core ideas, and multimodal LLMs still use image tokenisers.

</details>

---

### Stage 08 · Pretraining and scaling: how large language models are made

**The story.** This is where ChatGPT-style models come from. Train a transformer to predict the next word on huge amounts of text, and it picks up grammar, facts and even skills. Make it bigger and feed it more data, and it gets predictably better. This stage covers the GPT series, the "scaling laws" that predict how good a model will be, open models like LLaMA, models for images and text (CLIP), code (Codex) and speech (Whisper), and how GPT-4 was evaluated.

#### 048 · [Improving Language Understanding by Generative Pre-Training (GPT)](08-Pretraining-and-Scaling-LLMs/048-Radford-et-al-2018-GPT/) — Radford et al., 2018
- **The idea:** **GPT**: first learn language by predicting the next word on lots of text, then fine-tune on a small labelled task.
- **What we built:** the model, the pretraining, and the paper's ways of turning tasks into text.
- **What we saw:** with only 40 labelled examples, the pretrained model reached 80–82% against 73% when trained from scratch. It could also classify reviews with **no labels at all** (89%), just by comparing "very good" with "very bad".

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**Before GPT.** Each language task (sentiment, question answering, entailment) needed its own model, trained on its own labelled data. Labelled data is expensive; unlabelled text is nearly free. Alec Radford and colleagues at OpenAI showed how to use unlabelled text first.

**Topic 1: stage 1, generative pre-training.** Train a transformer **decoder** (paper 034, masked so each word sees only earlier words) to **predict the next word** on a large text corpus:
```
maximise   Σ_i log p( word_i | word_{i−k}, …, word_{i−1} )
```
It was trained on **BooksCorpus** (about 7,000 unpublished books). Books have long stretches of connected text, which teaches long-range structure. The model had 12 layers, 768 dimensions and 12 heads.

**Topic 2: why next-word prediction teaches so much.** To predict the next word well, the model must learn grammar ("a" vs "an"), facts ("Paris is the capital of …"), sentiment, and some reasoning about the story. All of this comes "for free" from raw text.

**Topic 3: stage 2, fine-tuning.** Add one linear layer on top and train on the labelled task. The paper also keeps the language-modelling loss as an **auxiliary objective** during fine-tuning, which helps generalisation and speeds up learning.

**Topic 4: turning tasks into text.** A transformer reads one sequence, so structured inputs are flattened with special tokens:
- **Classification:** [start] text [extract].
- **Entailment:** [start] premise [delim] hypothesis [extract].
- **Similarity:** process both orders (A then B, and B then A) and add the results.
- **Multiple choice:** one sequence per (context, answer) pair, then a softmax over the answers.

**Topic 5: results.** It improved the state of the art on 9 of 12 tasks studied. The analysis showed that more pre-trained layers transferred to the task means better results, and that the pre-trained model could already do some tasks **zero-shot** (with no fine-tuning) using simple heuristics, a hint of what GPT-2 and GPT-3 would do.

**What we built and saw.** Pre-training, fine-tuning with the auxiliary loss, and the input transformations.
- With only 40 labelled examples, the pre-trained model reached 80–82% against 73% training from scratch.
- Zero-shot sentiment by comparing the model's probability of "very good" vs "very bad" after a review reached 89% with no labels at all.

**Why it matters.** "Pre-train on lots of text, then adapt" is the foundation of all modern language models. This is the "G-P-T": **Generative Pre-trained Transformer**.

</details>

#### 049 · [Language Models are Unsupervised Multitask Learners (GPT-2)](08-Pretraining-and-Scaling-LLMs/049-Radford-et-al-2019-GPT-2/) — Radford et al., 2019
- **The idea:** **GPT-2** showed that a big enough language model can do tasks like answering questions **without any fine-tuning**, just from how the question is phrased.
- **What we built:** GPT-2's way of splitting text into tokens, its exact model sizes, and its method for detecting test data leaked into training.
- **What we saw:** in a toy "web", it answered questions about facts it had only seen written as normal prose.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The step forward.** GPT-2 (Alec Radford and colleagues, 2019) asked: if we make the model much bigger and train it on much more varied text, can it do tasks **without any fine-tuning**, just from how the input is written?

**Topic 1: WebText.** A new dataset of about **8 million web pages (40 GB of text)**, collected from links shared on Reddit that received at least 3 "karma" (upvotes). This is a cheap human quality filter. Wikipedia was removed to reduce overlap with test sets.

**Topic 2: byte-level BPE tokenisation.** **Byte Pair Encoding** starts with single characters and repeatedly merges the most frequent adjacent pair into a new token: "t" + "h" becomes "th", then "th" + "e" becomes "the", and so on. GPT-2 does this on raw **bytes**, so any text (any language, emoji, code) can be encoded with no "unknown" tokens. It uses a vocabulary of 50,257 tokens, with a rule preventing merges across character categories (so "dog." and "dog!" don't become separate tokens).

**Topic 3: model sizes.** Four sizes: 117M, 345M, 762M and **1.5B** parameters (48 layers, 1,600 dimensions). Layer norm was moved to the input of each block, and residual weights were scaled at initialisation by 1/√(number of layers).

**Topic 4: zero-shot task transfer.** Tasks are **written as text** the model can continue:
- **Summarisation:** article + "TL;DR:" makes the model write a summary.
- **Translation:** "english sentence = french sentence" examples, then "new english sentence =".
- **Question answering:** a document, then "Q: … A:".
- **Reading comprehension** and **commonsense** (LAMBADA, Winograd schemas).

The model was never trained on these tasks; it picked them up from patterns in web text.

**Topic 5: results.** State of the art on 7 of 8 language-modelling benchmarks zero-shot, and large improvements on LAMBADA (predicting the last word of a passage). Translation and summarisation were rudimentary but non-trivial. Performance improved steadily with size.

**Topic 6: contamination checks.** Did test sets leak into WebText? The paper checked for overlapping **8-word sequences** using Bloom filters and found small overlaps, which explained only small parts of the gains.

**Topic 7: staged release.** Concerned about misuse (fake news, spam), OpenAI released the models gradually, which started a lasting debate on openness.

**What we built and saw.** Byte-level BPE, the exact model configurations and the 8-gram overlap detector. In a toy "web" where facts appear only in prose, the model could answer questions about them when prompted as Q/A.

**Why it matters.** GPT-2 showed that language modelling at scale produces general skills ("unsupervised multitask learners") and set up the scaling of GPT-3.

</details>

#### 050 · [Language Models are Few-Shot Learners (GPT-3)](08-Pretraining-and-Scaling-LLMs/050-Brown-et-al-2020-GPT3/) — Brown et al., 2020
- **The idea:** **GPT-3** (175 billion parameters) can learn a new task from a few examples written in the prompt (**in-context learning**), with no retraining.
- **What we built:** GPT-3's size and compute formulas, its data filtering and de-duplication, and the prompt formats.
- **What we saw:** a small model clearly learned from the examples in its context, coming within 0.02 of the best possible score on a task designed to measure this.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The leap.** GPT-3 (Tom Brown and colleagues, 2020) scaled GPT-2's approach about 100×, to **175 billion parameters**, and showed something new: the model could learn a task **from a few examples in its prompt**, with no training at all.

**Topic 1: in-context learning.** Three settings, with no weight updates in any of them:
- **Zero-shot:** a task description only: "Translate English to French: cheese =>".
- **One-shot:** a description plus one example.
- **Few-shot:** a description plus typically 10–100 examples (as many as fit in the 2,048-token window).

The model works out the pattern from the examples and continues it. The examples work like "training data" read at inference time.

**Topic 2: the model.**
- 96 layers, 12,288 dimensions and 96 heads.
- Alternating dense and locally banded sparse attention (paper 036).
- Eight sizes from 125M to 175B, to study scaling.

**Topic 3: compute.** A useful rule of thumb: training costs about **6 × N × D** floating-point operations (N = parameters, D = training tokens). For GPT-3: 6 × 175×10⁹ × 300×10⁹ ≈ **3.1 × 10²³ FLOPs**. That is thousands of GPU-years in 2020 terms.

**Topic 4: the data.** About 300 billion training tokens from:
- **filtered Common Crawl** (a quality classifier trained to resemble WebText, plus fuzzy de-duplication);
- WebText2, two book corpora and Wikipedia.

Higher-quality sources were **sampled more often** than their size alone would suggest.

**Topic 5: results.**
- Strong few-shot results on question answering (TriviaQA), translation into English, and cloze tasks.
- Arithmetic: near-perfect 2-digit addition few-shot, weaker on larger numbers.
- Generated news articles that humans could barely distinguish from real ones (about 52% accuracy, close to chance).
- Weak spots: some reasoning and comparison tasks (such as ANLI and WiC).
- Few-shot ability **grows with model size** faster than zero-shot.

**Topic 6: contamination.** A bug meant some test-set overlaps were not removed before training. The paper analyses each benchmark's "clean" subset to estimate the effect, which was usually small.

**Topic 7: limitations and risks.** The paper discusses repetition, lack of grounding, bias in gender, race and religion, energy use and misuse.

**What we built and saw.** The size and compute formulas, data filtering, fuzzy de-duplication (MinHash) and the prompt formats. A small model learned clearly from examples in its context, coming within 0.02 of the best possible score on a task designed to measure in-context learning.

**Why it matters.** In-context learning changed how people use AI: instead of training models, they **write prompts**. GPT-3 led directly to InstructGPT (paper 067) and ChatGPT.

</details>

#### 051 · [Scaling Laws for Neural Language Models](08-Pretraining-and-Scaling-LLMs/051-Kaplan-et-al-2020-Scaling-Laws/) — Kaplan et al., 2020
- **The idea:** a model's error falls smoothly and predictably as you add parameters, data and computing power, following simple "power laws".
- **What we built:** every formula in the paper and a calculator for the best model size for a given budget.
- **What we saw:** the calculator agrees with the paper's own table to within about a factor of 2, and the paper's numbers fit together consistently.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The question.** How does a language model's quality change as you make it bigger, give it more data, or train it longer? Jared Kaplan and colleagues at OpenAI trained hundreds of models and found surprisingly simple rules.

**Topic 1: power laws.** The test loss L (cross-entropy, lower is better) follows a **power law** in each resource when the others aren't limiting:
```
L(N) ≈ (N_c / N)^0.076        N = parameters (excluding embeddings)
L(D) ≈ (D_c / D)^0.095        D = training tokens
L(C) ≈ (C_c / C)^0.050        C = compute
```
On a log-log plot, these are straight lines, over **more than seven orders of magnitude**.

**Worked example.** Doubling N multiplies the loss by 2^−0.076 ≈ 0.949, about a 5% reduction. Ten times more parameters multiply it by 10^−0.076 ≈ 0.84. Progress is steady and **predictable**.

**Topic 2: shape matters little.** For a fixed number of parameters, the exact depth, width and number of heads hardly matter (within a wide range). Total size is what counts.

**Topic 3: overfitting.** When both N and D are limited, the loss depends on them jointly. To avoid overfitting, data should grow roughly as N^0.74: a model 8× bigger needs only about 5× more data.

**Topic 4: bigger models learn faster.** Larger models reach a given loss with **fewer** tokens (they are more sample-efficient).

**Topic 5: compute-optimal training (Kaplan's version).** With a fixed compute budget, most of the extra compute should go into **model size**: N grows about as C^0.73, while data grows much more slowly. Train large models and **stop well before convergence**. This advice shaped GPT-3, which was trained on relatively few tokens for its size. Paper 052 later revised it.

**Topic 6: batch size.** There is a **critical batch size** that grows as the loss decreases. Beyond it, bigger batches waste compute.

**What we built and saw.** Every formula in the paper and a calculator for the best model size and data size at a given compute budget. The calculator agreed with the paper's own tables to within about a factor of 2, and the paper's fitted constants were mutually consistent.

**Why it matters.** Scaling laws turned AI development into partly an **engineering forecast**: labs could predict the benefit of a bigger training run before paying for it. This drove the race to scale up.

</details>

#### 052 · [Training Compute-Optimal Large Language Models (Chinchilla)](08-Pretraining-and-Scaling-LLMs/052-Hoffmann-et-al-2022-Chinchilla/) — Hoffmann et al., 2022
- **The idea:** a correction to the scaling laws: for the best results, train on **far more data**, about 20 words (tokens) per parameter. Models before this were too big for the data they saw.
- **What we built:** all three of the paper's methods for finding the best balance.
- **What we saw:** all three recovered the correct answer from data we generated, and at equal computing cost the "Chinchilla" balance scored better than the older approach.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The correction.** DeepMind's Jordan Hoffmann and colleagues re-examined scaling (paper 051) and concluded that large models of the time (GPT-3, Gopher) were **undertrained**: too many parameters for too little data.

**Topic 1: the question.** For a fixed compute budget C ≈ 6·N·D (paper 050), what model size N and number of tokens D give the lowest loss?

**Topic 2: three independent methods.**
1. **Envelope of training curves:** train models of many sizes, each for different numbers of tokens. For each compute level, find which model got the lowest loss.
2. **IsoFLOP profiles:** fix the compute (an "isoFLOP" budget), vary the model size, and observe a U-shaped curve of loss. The bottom of the U is the best size for that budget.
3. **A fitted formula:**
```
L(N, D) = E + A / N^α + B / D^β        (fitted: E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28)
```
   - E is the irreducible loss (the entropy of natural text);
   - A/N^α is the penalty for a finite model;
   - B/D^β is the penalty for finite data.

**Topic 3: the answer.** All three methods agree: N and D should grow **equally** with compute (each about C^0.5). That works out to roughly **20 training tokens per parameter**. Kaplan's laws had suggested putting most of the compute into size.

**Topic 4: Chinchilla vs Gopher.** Gopher had 280B parameters and was trained on 300B tokens (about 1 token per parameter). Chinchilla had **70B** parameters and 1.4 trillion tokens (20 per parameter), using **about the same compute**: 6 × 70×10⁹ × 1.4×10¹² ≈ 5.9×10²³ FLOPs. Chinchilla beat Gopher on almost every benchmark (for example 67.5% on MMLU vs 60%). Being 4× smaller, it is also much **cheaper to run**.

**Topic 5: why the earlier result differed.** Kaplan et al. used a fixed learning-rate schedule length for all runs, regardless of how many tokens each run used. That made short runs look worse than they should, and biased the conclusion toward bigger models.

**What we built and saw.** All three fitting methods. On data generated from a known law, all three recovered the correct optimum. At equal compute, the "Chinchilla" allocation reached a lower loss than the Kaplan allocation.

**Why it matters.** After Chinchilla, models were trained on far more data (LLaMA, paper 056, uses even more than 20 tokens per parameter, because smaller models are cheaper to serve). "Tokens per parameter" became a standard planning number.

</details>

#### 053 · [Learning Transferable Visual Models from Natural Language Supervision (CLIP)](08-Pretraining-and-Scaling-LLMs/053-Radford-et-al-2021-CLIP/) — Radford et al., 2021
- **The idea:** **CLIP** learns to match pictures with their captions, which lets it recognise new kinds of images just from a text description, with no labelled examples.
- **What we built:** CLIP's training method and its "zero-shot" classifier built from text prompts.
- **What we saw:** trained only on captions, it recognised digits 88% of the time with no digit labels, and matching captions worked far better than predicting the exact caption words (88% vs 49%).

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The idea.** Image classifiers were trained on fixed label sets (for example ImageNet's 1,000 categories) and couldn't recognise anything else. The internet has hundreds of millions of images **with captions**. Alec Radford and colleagues trained **CLIP** (Contrastive Language–Image Pre-training) on **400 million** image–caption pairs.

**Topic 1: two encoders.**
- An **image encoder** (ResNet or ViT, paper 038) turns an image into a vector.
- A **text encoder** (a transformer) turns a caption into a vector.
- Both are projected to the same size and normalised to length 1, so their dot product is the **cosine similarity**.

**Topic 2: contrastive training.** Take a batch of N image–caption pairs and compute an **N×N matrix** of similarities between every image and every caption. The N correct pairs lie on the diagonal. Train so that each image's correct caption scores highest in its row, and each caption's correct image scores highest in its column. Each row and each column is a classification problem solved with cross-entropy, so the loss is **symmetric**. A learned **temperature** scales the similarities (how sharp the softmax is).

**Topic 3: why contrastive?** Predicting the **exact caption words** is very hard: there are many valid ways to describe a picture. The paper found that predicting a bag of words was about 3× more efficient than predicting the exact caption, and the contrastive approach was about **4× more efficient** again. Matching is easier than generating.

**Topic 4: zero-shot classification.** To classify an image into new categories:
1. write each class as a sentence: "a photo of a dog.", "a photo of a cat.", …;
2. encode all these sentences;
3. pick the class whose sentence is most similar to the image.

No training images for those classes are needed. **Prompt ensembling** averages several templates ("a photo of a big {}", "a drawing of a {}"), which improves accuracy.

**Topic 5: results.** Zero-shot CLIP matched the accuracy of an original ResNet-50 on ImageNet (76.2%) **without using any of ImageNet's 1.28 million training labels**. It was also much more **robust**: on ImageNet variants (sketches, renditions, adversarially filtered photos) it lost far less accuracy than standard ImageNet models.

**What we built and saw.**
- The contrastive loss, the zero-shot classifier with prompt ensembling, and a linear-probe baseline.
- Trained only on captions of coloured digit images, CLIP recognised digits **88%** of the time with no digit labels (85% with a single prompt), and colours 100%.
- Contrastive training reached 88% vs 49% for bag-of-words prediction at the same training steps.

**Why it matters.** CLIP connected vision and language. It guides and ranks image generation (DALL·E, paper 047; Stable Diffusion uses CLIP's text encoder), powers image search, and its image encoders feed many multimodal chatbots.

</details>

#### 054 · [Evaluating Large Language Models Trained on Code (Codex)](08-Pretraining-and-Scaling-LLMs/054-Chen-et-al-2021-Codex/) — Chen et al., 2021
- **The idea:** **Codex** writes programs. The paper's key contribution is judging code by **running it against tests** ("pass@k"), not by how similar it looks to a reference.
- **What we built:** the fair way of computing pass@k, a safe harness for running generated code, and sampling settings.
- **What we saw:** a buggy program scored high on text similarity while a correct one scored low, which shows why running tests matters.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The goal.** Mark Chen and colleagues at OpenAI fine-tuned GPT on public code to create **Codex**, which writes Python functions from their descriptions (docstrings). It powered the first GitHub Copilot.

**Topic 1: training data.** **159 GB** of unique Python files from 54 million public GitHub repositories, after filtering out auto-generated files, files with very long lines, and so on.

**Topic 2: judging code by running it (HumanEval).** Text-similarity scores like BLEU are bad for code: two programs can look alike yet one is broken, or look different yet both be correct. The paper created **HumanEval**: **164 hand-written programming problems**, each with a function signature, a docstring and **unit tests**. A solution counts only if it **passes the tests**. Problems were written by hand to avoid being in the training data.

**Topic 3: pass@k.** Generate k samples per problem; the problem is solved if **any** sample passes. Estimating this naively from exactly k samples is noisy. The paper's unbiased estimator draws n ≥ k samples, counts c correct, and computes:
```
pass@k = 1 − C(n − c, k) / C(n, k)          (C = "choose", the number of combinations)
```
**Worked example.** With n = 10 samples of which c = 2 are correct:
- pass@1 = 1 − C(8,1)/C(10,1) = 1 − 8/10 = **0.2**;
- pass@5 = 1 − C(8,5)/C(10,5) = 1 − 56/252 = **0.78**.

**Topic 4: sampling temperature.** **Temperature** controls randomness. For pass@1, low temperature (near-greedy) is best. For pass@100, higher temperature (around 0.8) is best, because diverse attempts make it more likely that one is right.

**Topic 5: results.**
- Codex-12B solved **28.8%** of problems with one sample.
- With 100 samples, at least one correct for **72.3%**.
- GPT-3 solved essentially 0%.
- Choosing among samples by the model's own average log-probability helped when tests aren't available.

**Topic 6: running code safely.** Generated code is untrusted. Evaluations run in a **sandbox** (an isolated environment without network access) so that bad code can't do harm.

**Topic 7: limitations and risks.** Codex struggles with long chains of operations and variable binding, and can produce insecure code or code that looks right but isn't. The paper also discusses over-reliance and economic effects.

**What we built and saw.** The unbiased pass@k estimator (checked against brute force), a sandboxed test runner, temperature sampling, and BLEU for comparison. A buggy program scored high on BLEU while a correct but differently written one scored low, which is exactly why functional testing matters.

**Why it matters.** Running tests (pass@k) is now the standard way to evaluate code models, and code assistants are among the most widely used AI products.

</details>

#### 055 · [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](08-Pretraining-and-Scaling-LLMs/055-Radford-et-al-2023-Whisper/) — Radford et al., 2022
- **The idea:** **Whisper** learns speech recognition from 680,000 hours of imperfectly labelled audio, and becomes robust to noise and accents.
- **What we built:** Whisper's audio processing, model design and output format.
- **What we saw:** a tiny model trained only on clean sound fell apart with any noise, while one trained on varied noise stayed accurate, which is the paper's main lesson.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The idea.** Speech recognisers trained on clean, carefully transcribed datasets (like LibriSpeech audiobooks) performed brilliantly on that dataset but poorly on real-world audio. Alec Radford and colleagues trained **Whisper** on **680,000 hours** of audio with transcripts from the internet. The transcripts are imperfect ("weak supervision"), but there is a huge and varied amount of them.

**Topic 1: from sound to a spectrogram.** Audio is resampled to 16 kHz and cut into **30-second chunks**. Each chunk becomes a **log-Mel spectrogram**: the frequency content of each 25 ms window (moving 10 ms at a time), measured in 80 frequency bands spaced like human hearing (the Mel scale), on a logarithmic loudness scale. It works like a picture of sound.

**Topic 2: the model.** A standard **encoder–decoder transformer** (paper 034). The encoder reads the spectrogram (after two small convolution layers); the decoder writes the text tokens.

**Topic 3: one model, many tasks, via special tokens.** The decoder's first tokens say what to do:
```
<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>  …text…  <|endoftext|>
```
- a **language** tag (99 languages);
- **transcribe** (same language) or **translate** (into English);
- timestamps on or off;
- a special token when there is no speech.

**Topic 4: data cleaning.** Internet transcripts include machine-generated captions from other recognisers (which would teach Whisper their mistakes). The authors used heuristics to remove them (for example, text that is all lower-case or has no punctuation), along with mismatched languages and bad alignments.

**Topic 5: measuring with word error rate (WER).** WER = (substitutions + deletions + insertions) / number of words in the reference. Example: the reference "the cat sat down" vs the output "the bat sat" has 1 substitution and 1 deletion, so WER = 2/4 = 50%.

**Topic 6: robustness.** Whisper was tested **zero-shot** (without fine-tuning) on many datasets. A model fine-tuned on LibriSpeech and Whisper can have the same LibriSpeech score, yet on other datasets Whisper made **about 55% fewer errors** on average. It was also much more robust to background noise. Its errors approached those of professional human transcribers on some tests.

**What we built and saw.** The audio front end, the encoder–decoder and the special-token format. A tiny model trained only on clean synthetic sounds fell apart when noise was added, while one trained on varied noisy data stayed accurate. That is the paper's main lesson about diverse data.

**Why it matters.** Whisper is open source and widely used for transcription, subtitles and voice interfaces. It showed that **lots of diverse, imperfect data** beats small, perfect data for robustness.

</details>

#### 056 · [LLaMA: Open and Efficient Foundation Language Models](08-Pretraining-and-Scaling-LLMs/056-Touvron-et-al-2023-LLaMA/) — Touvron et al., 2023
- **The idea:** **LLaMA**, a family of openly released models trained on public data, with several small design improvements.
- **What we built:** a tiny LLaMA with all its improvements. Later papers in this repository reuse it.
- **What we saw:** the paper's model sizes, training time and carbon figures match our calculations, and our efficient attention gives the same answer as the slow version.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The goal.** Hugo Touvron and colleagues at Meta trained a family of strong language models (**7B, 13B, 33B and 65B** parameters) using **only publicly available data**, and released them to researchers.

**Topic 1: train smaller models longer.** Following Chinchilla (paper 052), but with a twist: Chinchilla optimises **training** compute, while LLaMA optimises **inference** cost. A smaller model trained on more tokens is cheaper to use forever after. LLaMA-7B was trained on 1 trillion tokens and the 33B/65B on 1.4 trillion, which is well past "20 tokens per parameter" for the smaller models, and they kept improving.

**Topic 2: the data.** Public sources only:
- CommonCrawl (67%), filtered and de-duplicated;
- C4;
- GitHub code;
- Wikipedia in 20 languages;
- public-domain books;
- arXiv papers;
- StackExchange.

**Topic 3: RMSNorm (pre-normalisation).** Each sub-layer's **input** is normalised (as in GPT-2), using **RMSNorm**, which divides by the root-mean-square without subtracting the mean:
```
RMSNorm(x) = x / √(mean(x²) + ε) · g
```
**Example.** x = [3, 4]: the mean of the squares is (9 + 16)/2 = 12.5, whose square root is 3.536, so the output is [0.849, 1.131]·g. It is simpler and faster than layer norm and works as well.

**Topic 4: SwiGLU.** The feed-forward layer uses a **gated** activation:
```
FFN(x) = W2 · ( SiLU(W1·x) ⊙ (W3·x) )        SiLU(u) = u · σ(u)
```
One branch acts as a learned gate on the other. The hidden size is set to (2/3)·4d so the parameter count matches a standard FFN.

**Topic 5: rotary position embeddings (RoPE).** Instead of adding position vectors, RoPE **rotates** the query and key vectors by an angle proportional to their position (each pair of dimensions at a different speed). The dot product between a query at position m and a key at position n then depends only on their **relative offset** m − n, which is a neat way to encode relative position.

**Topic 6: efficient training.**
- AdamW with a cosine learning-rate schedule.
- An efficient causal attention implementation (no storing of masked scores).
- Activation checkpointing (saving memory by recomputing).

The 65B model trained in about 21 days on 2,048 A100 GPUs.

**Topic 7: results.** LLaMA-13B beat GPT-3 (175B) on most benchmarks despite being 10× smaller, and LLaMA-65B was competitive with Chinchilla-70B and PaLM-540B.

**What we built and saw.** A tiny LLaMA with RMSNorm, SwiGLU and RoPE; later papers in this repo (073–078) reuse it. The paper's size, compute and carbon numbers match our calculations, and our efficient attention gives the same output as the straightforward version.

**Why it matters.** LLaMA kicked off the open-weights model ecosystem (Alpaca, Vicuna, LLaMA-2/3, Mistral and thousands of fine-tunes). Its architecture choices (pre-RMSNorm, SwiGLU, RoPE) became the standard recipe for open LLMs.

</details>

#### 057 · [GPT-4 Technical Report](08-Pretraining-and-Scaling-LLMs/057-OpenAI-2023-GPT-4-Technical-Report/) — OpenAI, 2023
- **The idea:** the report keeps the model's design secret, but explains how OpenAI **predicted** GPT-4's performance from much smaller models, and how it checked calibration and test-data leaks.
- **What we built:** those prediction and checking methods.
- **What we saw:** predicting a real tiny model's result from smaller ones came within 4.8%, and the leak checker caught copied text but missed paraphrases.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**What this is.** OpenAI's report on GPT-4. It **does not** reveal the architecture, size, data or training method, citing competition and safety. It does explain **how** they predicted and evaluated the model, which is the part we implement.

**Topic 1: predictable scaling.** Training GPT-4 costs a fortune, so you want to know the outcome in advance. OpenAI trained small models with **1,000× to 10,000× less compute**, fitted a scaling law to their final loss:
```
L(C) = a · C^b + c          (c = irreducible loss)
```
and **extrapolated**. The prediction of GPT-4's final loss was very accurate. Doing this required training infrastructure that behaves **predictably** across scales.

**Topic 2: predicting capabilities, not just loss.** They also predicted the pass rate on a subset of HumanEval (paper 054) from small models, using a power law on the mean log pass rate. Some abilities are **hard to predict**: the "hindsight neglect" task got **worse** with scale in smaller models (inverse scaling), and then GPT-4 reversed that trend.

**Topic 3: exams and benchmarks.** GPT-4 was tested on human exams (a simulated bar exam around the top 10% of test takers, SAT, GRE, AP exams) and on standard benchmarks like MMLU, where it reached 86.4% (5-shot).

**Topic 4: calibration.** A model is **calibrated** if, when it says it is 70% confident, it is right about 70% of the time. The pre-trained GPT-4 was well calibrated; **after RLHF** (Stage 10), calibration got noticeably worse. Fine-tuning for helpfulness can distort confidence.

**Topic 5: contamination checks.** Did test questions appear in training data? Their method: take **three random 50-character substrings** of each test question; if any appears in the training data, mark the question as contaminated. Then compare scores on contaminated vs clean questions.

**Topic 6: safety.** The report also describes red-teaming by domain experts, a reward model that penalises disallowed content, and reductions in harmful responses compared with GPT-3.5, along with remaining limitations (hallucinations, reasoning errors).

**What we built and saw.**
- The scaling-law fit and extrapolation: predicting a real tiny model's final loss from even smaller ones came within 4.8%.
- Calibration plots and the expected calibration error.
- The substring contamination check: it caught copied text but **missed paraphrased** questions, a real limitation of this method.

**Why it matters.** "Predict the big run from small runs" is how frontier labs plan training. Calibration and contamination checks are essential to trusting benchmark numbers.

</details>

#### 058 · [GPT-4o System Card](08-Pretraining-and-Scaling-LLMs/058-OpenAI-2024-GPT-4o-System-Card/) — OpenAI, 2024
- **The idea:** how a model is evaluated for safety before release: risk categories, scoring rules, and checks such as making sure the voice model only speaks in approved voices.
- **What we built:** the scoring rules and a toy voice-checking system.
- **What we saw:** the voice checker shows a real trade-off: to catch a very similar-sounding voice, it must also wrongly flag many approved clips.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**What this is.** A **system card** is a public document describing how a model was evaluated for risks before release, and what mitigations were applied. GPT-4o ("o" for omni) handles text, images and **audio** in one model, and talks in real time.

**Topic 1: the Preparedness Framework.** OpenAI rates models in four risk categories:
- **cybersecurity:** can it help carry out cyberattacks?
- **CBRN:** can it help create chemical, biological, radiological or nuclear threats?
- **persuasion:** can it change people's opinions more than humans can?
- **model autonomy:** can it act on its own to gain resources or self-improve?

Each is scored **low, medium, high or critical**. The overall score is the **highest** category score. Only models with a post-mitigation score of **medium or below** can be deployed. GPT-4o was rated medium overall (persuasion borderline medium; the others low).

**Topic 2: new risks from voice.**
- **Unauthorised voice generation:** the model might imitate someone's voice (for example the user's). Mitigation: the model may only speak in a few **preset voices**, and an **output classifier** checks every response; if the voice deviates, the output is blocked.
- **Speaker identification:** refusing to identify people from their voice.
- **Ungrounded inferences:** refusing to guess sensitive traits (like race or religion) from audio.
- **Violent or erotic speech** and copyrighted music.

**Topic 3: red teaming.** Over 100 external red-teamers, speaking 45 languages, probed the model in phases as it developed.

**Topic 4: classifier trade-offs.** A voice-checking classifier gives a score; you choose a **threshold** for blocking.
- **Precision:** of the blocked outputs, how many were really wrong voices.
- **Recall:** of the wrong voices, how many were blocked.

A stricter threshold catches more imitations but blocks more legitimate outputs. The card reports high precision and recall for its voice checker.

**Topic 5: societal impacts.** The card also discusses anthropomorphisation and emotional reliance (people treating a voice model like a person), health applications, and scientific capabilities.

**What we built and saw.** The scoring rules (max over categories, the deployment threshold) and a toy voice-checking system using voice "embeddings" and a similarity threshold. It shows the real trade-off: to catch a very similar-sounding imitation, the threshold must be strict enough that it also wrongly flags many genuine clips.

**Why it matters.** System cards have become standard for frontier models. They show that releasing AI involves **measured risk decisions**, not just accuracy numbers.

</details>

---

### Stage 09 · Reasoning and agents: getting more out of language models

**The story.** Once you have a language model, how do you make it think more carefully, check its own work, and act in the world? This stage covers "think step by step" prompting, a second model that checks answers, and **agents**: models that reason, use tools like search, remember, and plan.

#### 059 · [Chain-of-Thought Prompting Elicits Reasoning](09-Reasoning-and-Agents/059-Wei-et-al-2022-Chain-of-Thought/) — Wei et al., 2022
- **The idea:** show the model examples that **write out their reasoning step by step**, and it reasons better on new problems.
- **What we built:** the paper's prompts and a multi-step toy problem tested in several formats.
- **What we saw:** step-by-step answers stayed at 100% as problems got longer, while direct answers collapsed to around 11%. Like the paper, padding with meaningless dots did not help.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The discovery.** Large language models were poor at multi-step problems like maths word problems, even very large ones. Jason Wei and colleagues at Google found that if the **examples in the prompt show the reasoning steps**, the model also writes out reasoning, and gets far more answers right.

**Topic 1: standard vs chain-of-thought prompting.**

Standard few-shot example:
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls.
   How many tennis balls does he have now?
A: The answer is 11.
```
Chain-of-thought example:
```
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11.
   The answer is 11.
```
The paper used 8 such hand-written examples, then the new question.

**Topic 2: why it helps.**
- **More computation:** each written step is extra processing; the model doesn't have to jump to the answer in one go.
- **Decomposition:** a hard problem becomes a series of easy ones.
- **Interpretability:** you can see where reasoning went wrong.

**Topic 3: it is an emergent ability of scale.** Chain-of-thought **hurt or didn't help** small models (they write fluent but wrong reasoning). It helped a lot only at about **100 billion parameters** and above. With PaLM 540B, accuracy on GSM8K (grade-school maths) rose from **17.9% to 56.9%**, beating the previous best fine-tuned system.

**Topic 4: ablations, what is actually needed.** The authors tested alternatives:
- **Equation only** (just "5 + 2 × 3 = 11"): helped on simple problems, not on harder ones that need natural-language reasoning.
- **Variable compute only** (outputting dots "…" of the same length as a reasoning chain): did **not** help. So it isn't just extra tokens; the content matters.
- **Reasoning after the answer:** did not help. The reasoning must come **before** the answer to influence it.

**Topic 5: beyond maths.** It also improved commonsense reasoning (StrategyQA, date understanding, sports understanding) and symbolic tasks (concatenating the last letters of words, coin-flip tracking), including generalising to longer problems than the examples.

**What we built and saw.** The paper's prompt formats and a multi-step toy problem for a small model trained in each format. As problems got longer, step-by-step answers stayed at 100% while direct answers collapsed to about 11%. Meaningless "dots" padding did not help, matching the paper's ablation.

**Why it matters.** "Thinking out loud" became central to how LLMs are used and trained. Today's "reasoning models" are trained to produce long chains of thought before answering.

</details>

#### 060 · [Large Language Models are Zero-Shot Reasoners](09-Reasoning-and-Agents/060-Kojima-et-al-2022-Zero-Shot-Reasoners/) — Kojima et al., 2022
- **The idea:** simply adding **"Let's think step by step"** to a question makes models reason better, with no examples needed.
- **What we built:** the two-step prompting method and all 16 trigger phrases the paper compares.
- **What we saw:** the reasoning phrase kept accuracy high (83% on the hardest level, against 13% without it), while irrelevant or misleading phrases did not help, matching the paper's pattern.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The surprise.** Paper 059 needed hand-written reasoning examples. Takeshi Kojima and colleagues showed that a **single sentence**, "**Let's think step by step**", placed before the answer makes large models reason, with **no examples at all** (zero-shot).

**Topic 1: two-stage prompting.**
1. **Reasoning extraction:** prompt with "Q: [question] A: Let's think step by step." The model writes out its reasoning.
2. **Answer extraction:** feed everything back with a phrase like "Therefore, the answer (arabic numerals) is", so the final answer comes out in a format that can be checked.

**Topic 2: results.** With InstructGPT (text-davinci-002):
- MultiArith accuracy rose from **17.7% to 78.7%**;
- GSM8K rose from **10.4% to 40.7%**.

Gains also appeared on symbolic reasoning (last letters, coin flips) and some logical tasks. Like few-shot chain-of-thought, the benefit grows with model size.

**Topic 3: which trigger phrases work?** The paper compared 16 templates, grouped into types:
- **Instructive** ("Let's think step by step", "Let's think about this logically", "First,"): big improvements.
- **Misleading** ("Don't think. Just feel."): no help.
- **Irrelevant** ("By the way, I found a good restaurant nearby."): no help.

So it is not just any extra text; the phrase must **invite reasoning**.

**Topic 4: zero-shot vs few-shot.** Zero-shot chain-of-thought was below few-shot chain-of-thought (which has good examples), but far above ordinary zero-shot. Its big advantage is **zero effort**: no task-specific examples to write.

**Topic 5: the lesson about LLMs.** Large models contain reasoning abilities that are not shown by default. Simple prompting can **unlock** them, which is why "prompt engineering" became a skill.

**What we built and saw.** The two-stage method and all 16 trigger phrases, applied to a toy model trained on a mix of direct and step-by-step text. The reasoning phrase kept accuracy high on the hardest level (83% vs 13% without it), while irrelevant and misleading phrases did not help, matching the paper's pattern.

**Why it matters.** "Let's think step by step" is one of the most famous sentences in AI. It showed that how you ask changes what a model can do, and it inspired automatic prompt search and reasoning training.

</details>

#### 061 · [Training Verifiers to Solve Math Word Problems](09-Reasoning-and-Agents/061-Cobbe-et-al-2021-Training-Verifiers-Math/) — Cobbe et al., 2021
- **The idea:** generate many candidate solutions, then use a second model (a **verifier**) to pick the one most likely to be right.
- **What we built:** the generator, the verifier and the voting methods.
- **What we saw:** with little training data the verifier made things worse; with more data it helped clearly (75%, then 81% with voting). The paper predicts exactly this.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** Language models make reasoning mistakes, and a single error ruins a maths answer. Karl Cobbe and colleagues at OpenAI asked: rather than trying to always generate the right answer, generate **many** answers and train a second model to **recognise** which one is right.

**Topic 1: GSM8K.** The paper introduced **GSM8K**: about 8,500 grade-school maths word problems (7,500 for training, 1,000 for test), each requiring 2–8 steps of basic arithmetic, with full written solutions. It became one of the most used reasoning benchmarks.

**Topic 2: the generator.** Fine-tune a language model on the training solutions, then sample **many solutions** (for example 100) per problem at a moderate temperature for variety. Solutions include **calculator annotations** like `<<48/2=24>>`; at test time a real calculator fills in the result, which removes arithmetic slips.

**Topic 3: the verifier.** Label each sampled solution as correct or incorrect by checking **only the final answer**. Train a model to predict, from the problem plus solution, the probability that the solution is correct. At test time, generate 100 solutions and pick the one the verifier rates highest.

**Topic 4: token-level verifiers.** Instead of one score per solution, the verifier predicts correctness **after every token**, like a value function. This was better than solution-level scoring and gives a more detailed signal. Training the verifier jointly with a language-modelling objective also helped.

**Topic 5: voting.** Take the verifier's top-k solutions and **vote** on the final answer. This beats picking the single top one when the verifier is uncertain.

**Topic 6: the key results.**
- Verification gave about the same boost as a **30× larger model**: a 6B model with a verifier slightly beat a fine-tuned 175B model.
- With **little training data**, the verifier **overfits** and can be worse than simply sampling. It needs enough data to learn what correctness looks like.

**What we built and saw.** The generator, the token-level verifier and top-k voting, on a toy search problem.
- With only 100 training problems, the verifier made things **worse** (44%).
- With 2,000 it helped clearly: 75%, then 81% with voting.

That is exactly the data-dependence the paper reports.

**Why it matters.** "Generate many, then verify" is a core strategy in modern reasoning systems (best-of-n, reward models, process reward models), and it led to research on checking reasoning step by step.

</details>

#### 062 · [ReAct: Synergizing Reasoning and Acting](09-Reasoning-and-Agents/062-Yao-et-al-2023-ReAct/) — Yao et al., 2023
- **The idea:** an agent alternates **thinking** and **acting** (searching, reading) in a loop. This is the basis of most AI agents today.
- **What we built:** a Wikipedia-like environment, the think/act/observe loop, and the paper's fallback strategies.
- **What we saw:** a model answering from memory gave outdated answers when facts changed (3% right), while the acting agent read the pages and got 100%.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The idea.** Chain-of-thought (paper 059) reasons but can't look anything up, so it can hallucinate facts. Tool-using agents could act but didn't reason. Shunyu Yao and colleagues combined the two: **ReAct** (**Re**ason + **Act**) interleaves **thoughts** and **actions**.

**Topic 1: the loop.**
```
Thought 1: I need to find which magazine was started first, Arthur's Magazine or First for Women.
Action 1:  Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844–1846) was an American literary periodical…
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2:  Search[First for Women]
Observation 2: First for Women is a woman's magazine … started in 1989.
Thought 3: 1844 < 1989, so Arthur's Magazine was started first.
Action 3:  Finish[Arthur's Magazine]
```
- **Thoughts** plan, track progress, handle surprises and decide what to do next.
- **Actions** fetch real information.
- **Observations** are what the environment returns.

**Topic 2: the Wikipedia action space.** For question answering (HotpotQA) and fact checking (FEVER):
- `search[entity]` returns the first sentences of the page (or similar titles if not found);
- `lookup[string]` returns the next sentence containing the string (like Ctrl+F);
- `finish[answer]` ends the episode.

**Topic 3: prompting only.** ReAct used **few-shot prompting** with a handful of human-written trajectories, with no training at all. (Fine-tuning on generated trajectories was also explored.)

**Topic 4: combining with chain-of-thought.** ReAct is more **factual** (it checks sources) but sometimes gets stuck in loops. Chain-of-thought with self-consistency is better at pure reasoning but hallucinates. The best results came from **switching**: if ReAct fails to answer within a few steps, fall back to CoT-SC, and vice versa.

**Topic 5: interactive environments.**
- **ALFWorld** (a text household game): "go to countertop 1", "take apple".
- **WebShop:** navigate a shopping website to buy a product matching instructions.

With just one or two examples, ReAct beat imitation and reinforcement learning methods trained on thousands of examples (by +34% and +10% absolute success).

**What we built and saw.** A toy Wikipedia-like environment, the Thought/Action/Observation loop and the fallback strategy. When facts in the environment changed, a model answering from memory gave outdated answers (3% correct), while the ReAct agent read the current pages and got 100%.

**Why it matters.** ReAct is the template for most LLM agents today (tool use, browsing, coding agents): think, act, observe, repeat.

</details>

#### 063 · [WebGPT: Browser-Assisted Question Answering](09-Reasoning-and-Agents/063-Nakano-et-al-2021-WebGPT/) — Nakano et al., 2021
- **The idea:** a model that **browses the web** to answer questions, trained with human feedback on which answers are better.
- **What we built:** a text browser, a scoring model learned from comparisons, best-of-n selection and reinforcement learning.
- **What we saw:** a classic trap appeared. Pushing too hard on the scoring model made answers *worse* in reality while the score kept rising (the model learned that longer looked better).

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The goal.** Answer open-ended questions with long, **well-sourced** answers by letting GPT-3 use a web browser. Reiichiro Nakano and colleagues at OpenAI trained it using **human feedback**.

**Topic 1: a text-based browser.** The model sees a text summary of the browser state and issues commands:
- `Search <query>` (via the Bing API);
- `Clicked on link <id>`;
- `Find in page: <text>`;
- `Quote: <text>`, which saves a passage as a **reference**;
- `Scrolled down/up`, `Back`;
- `End: Answer`, which writes the final answer using the collected quotes.

**Topic 2: behaviour cloning.** People used the same interface to answer questions (from the ELI5, "Explain Like I'm Five", dataset). The model was fine-tuned to **imitate** their browsing and answering. This is called behaviour cloning.

**Topic 3: a reward model from comparisons.** Human raters compared **pairs of answers** to the same question (with references) and chose the better one. A **reward model** learned to predict these preferences. Comparisons are easier and more reliable for people than absolute scores.

**Topic 4: three ways to use the reward model.**
- **Reinforcement learning (PPO)**: fine-tune the policy to get higher rewards.
- **Rejection sampling (best-of-n)**: generate n answers and return the one with the highest reward (the best results used n = 64).
- Both together.

Surprisingly, **best-of-n** gave most of the benefit, and RL added little.

**Topic 5: results.** The best model's answers were preferred over the human demonstrators' answers **56%** of the time, and over the highest-voted Reddit answer 69% of the time. References made it easier for raters to judge factual accuracy.

**Topic 6: over-optimisation.** A reward model is only an approximation of human judgement. If you push too hard against it (very large n, or too much RL), the policy finds ways to get high scores that humans don't actually prefer. This is **Goodhart's law**: when a measure becomes a target, it stops being a good measure.

**What we built and saw.** A toy text browser, a learned reward model, best-of-n and RL. We reproduced the over-optimisation trap: the reward model had learned that longer answers tend to be better, so pushing hard on it made answers longer and **worse** in true quality while the reward score kept rising.

**Why it matters.** WebGPT was an early version of what is now standard: AI assistants that search the web and cite sources. It also contributed the reward-model and best-of-n methods used in RLHF (paper 067).

</details>

#### 064 · [Generative Agents: Interactive Simulacra of Human Behavior](09-Reasoning-and-Agents/064-Park-et-al-2023-Generative-Agents/) — Park et al., 2023
- **The idea:** a town of 25 AI characters with memories, reflections and daily plans, who spread news and organise a party on their own.
- **What we built:** the memory system with its scoring, reflection and planning, plus the town (with simple stand-ins in place of a real language model).
- **What we saw:** news of a party spread to about 9–12 of the 25 agents, and switching off the "importance" part of memory stopped it spreading entirely.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The idea.** Can language models simulate believable **people** who live their days, remember, form opinions and interact? Joon Sung Park and colleagues built **Smallville**, a little town with **25 AI agents** (like a game of The Sims), each powered by an LLM.

**Topic 1: the memory stream.** Each agent keeps a list of everything it observes, as natural-language records with timestamps: "Isabella Rodriguez is setting up tables at the café". The list quickly becomes too long to fit in a prompt, so the agent must **retrieve** the relevant memories.

**Topic 2: retrieval scoring.** Each memory gets three scores, each scaled between 0 and 1, and the total is their sum:
- **recency:** decays exponentially (factor 0.995 per game hour) since the memory was last accessed;
- **importance:** the LLM rates each memory from 1 (mundane, "brushing teeth") to 10 (poignant, "a breakup");
- **relevance:** cosine similarity between the memory's embedding and the current situation.

The top-scoring memories go into the prompt.

**Topic 3: reflection.** Raw observations aren't enough for higher-level thinking. When the summed importance of recent events passes a threshold (150), the agent **reflects**:
1. it asks itself the most salient questions about its recent experiences;
2. it retrieves memories to answer them;
3. it writes **insights** ("Klaus is dedicated to his research") back into the memory stream, citing the memories behind them.

Reflections can build on reflections, forming a tree of more abstract beliefs.

**Topic 4: planning.** Each morning an agent drafts a broad daily plan, then recursively breaks it into hour-long and then 5–15-minute chunks. Plans are stored in memory and **revised** when something unexpected happens (an agent decides whether to react to an observation).

**Topic 5: emergent social behaviour.**
- **Information diffusion:** news that Sam is running for mayor spread from 1 agent to 8 (32%); news of Isabella's Valentine's Day party spread from 1 to 13 (52%).
- **Coordination:** starting only from Isabella's intention to throw a party, agents invited each other, one asked another on a date to it, and five showed up at the right time and place.

**Topic 6: evaluation.** Agents were "interviewed" about their memories, plans and reflections, and humans rated how believable they were. Removing memory, reflection or planning (ablations) made them less believable. The full architecture was rated more believable than even human crowd-workers role-playing the agents.

**What we built and saw.** The memory stream with retrieval scoring, reflection, planning and a town simulation, with simple stand-ins for the LLM calls so it runs instantly. Party news spread to about 9–12 of the 25 agents. **Removing importance scoring** stopped it spreading, because the news got buried among mundane memories.

**Why it matters.** The memory/reflection/planning design became a blueprint for LLM agents with long-term memory, and for simulations used in social science, games and testing.

</details>

#### 065 · [A Survey on Large Language Model based Autonomous Agents](09-Reasoning-and-Agents/065-Wang-et-al-2023-LLM-Agents-Survey/) — Wang et al., 2023
- **The idea:** a map of the whole agent field: profiles, memory, planning and action.
- **What we built:** the survey's framework as plug-together parts, with five planning strategies.
- **What we saw:** on a six-step task, planning with feedback from the environment succeeded about 95% of the time, against about 25% for planning once and hoping.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**What it is.** A survey (Lei Wang and colleagues) organising the fast-growing field of agents built on large language models into a single framework. It is a map rather than a new method.

**Topic 1: the four-part architecture.**
- **Profile:** who the agent is (role, personality, background). It can be written by hand, generated by an LLM, or taken from real data (for example survey demographics).
- **Memory:** what the agent remembers.
  - *Short-term* is what fits in the prompt; *long-term* is stored outside it (for example in a vector database).
  - Formats include natural language, embeddings, databases and structured lists.
  - Operations: **reading** (retrieval by recency, relevance and importance, paper 064), **writing** (including handling duplicates and overflow) and **reflection** (summarising into insights).
- **Planning:** how the agent decides what to do.
  - *Without feedback*: plan once (a single chain-of-thought, or multiple paths like tree-of-thoughts).
  - *With feedback*: revise the plan using feedback from the **environment** (did the action work?), from **humans**, or from **another model** (self-critique, debate).
- **Action:** how plans become actions.
  - **Goals:** completing tasks, communicating, exploring.
  - **Ways of producing actions:** from memory or by following plans.
  - **Action space:** external tools (APIs, search, code execution) or the model's own knowledge.
  - **Effects:** changing the environment, changing internal state, triggering new actions.

**Topic 2: how agents gain capabilities.**
- **With fine-tuning:** training on human-annotated, LLM-generated or real-world task data.
- **Without fine-tuning:** prompt engineering (for example chain-of-thought) and **mechanism engineering**: trial and error, crowd-sourcing (many agents debating), accumulating experience, self-driven evolution.

**Topic 3: applications.** Social science (simulating people, psychology experiments), natural science (literature search, lab automation, chemistry tools) and engineering (software development with multiple agent roles, robotics).

**Topic 4: evaluation.**
- **Subjective:** human ratings, "Turing tests".
- **Objective:** task success, human-similarity metrics, efficiency, using benchmarks and simulated environments.

**Topic 5: open problems.** Role-playing fidelity, alignment with human values, prompt robustness, hallucination, knowledge boundaries (an agent simulating a 1900s person shouldn't know about smartphones), and efficiency.

**Why feedback matters, a simple calculation.** If each step of a 6-step task succeeds with probability 0.8 and errors are never corrected, the whole task succeeds with probability 0.8⁶ ≈ 0.26. If the agent can **notice and retry** failed steps using feedback, success approaches 100%.

**What we built and saw.** The framework as plug-together modules (profile, memory, planning, action) with five planning strategies. On a six-step task, planning with environment feedback succeeded about **95%** of the time, while plan-once-and-hope succeeded about **25%**, in line with the compounding-errors calculation above.

**Why it matters.** The survey's vocabulary (profile, memory, planning, action) is now standard when designing and discussing AI agents.

</details>

---

### Stage 10 · Alignment: teaching models what people want

**The story.** A model that predicts text is not automatically helpful or safe. This stage covers how assistants like ChatGPT are trained: people compare pairs of answers, a "reward model" learns their preferences, and the language model is then trained to score well on it. It also covers using AI feedback guided by written principles instead of human labels, and a simpler method (DPO) that skips the reinforcement-learning step.

#### 066 · [Learning to Summarize from Human Feedback](10-Alignment/066-Stiennon-et-al-2020-Summarize-from-Human-Feedback/) — Stiennon et al., 2020
- **The idea:** train a **reward model** from human comparisons of summaries, then improve the summariser with reinforcement learning.
- **What we built:** the whole pipeline with real (tiny) networks: supervised training, comparisons, reward model and reinforcement learning.
- **What we saw:** quality rose clearly. Without a penalty for drifting too far from the original model, it exploited the reward model by repeating the main topic: a higher score but worse real quality.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** Summarisation models were trained to **imitate** human summaries and evaluated with ROUGE (word overlap). But imitation copies the bad habits of the reference data, and ROUGE doesn't capture what people actually find good. Nisan Stiennon and colleagues at OpenAI trained directly on **human preferences**.

**Topic 1: the data.** Reddit posts from the TL;DR dataset (where authors write a short "too long; didn't read" summary), filtered for quality.

**Topic 2: step 1, a supervised baseline.** Fine-tune GPT-style models (1.3B and 6.7B) to write summaries by imitating the reference TL;DRs.

**Topic 3: step 2, collect comparisons.** Show human labellers a post and **two** summaries, and ask which is better. The authors worked closely with labellers to ensure high agreement with their own judgements.

**Topic 4: step 3, train a reward model.** A model r(post, summary) outputs a score. It is trained so that the preferred summary scores higher, using the **Bradley–Terry** model of preference:
```
P(summary A preferred over B) = σ( r(A) − r(B) )
loss = −log σ( r(preferred) − r(rejected) )
```
**Example.** If r(preferred) = 2 and r(rejected) = 1, then σ(1) = 0.731 and the loss is −log 0.731 = 0.31. If the scores were reversed, the loss would be −log σ(−1) = 1.31.

**Topic 5: step 4, optimise with reinforcement learning.** Fine-tune the summariser with PPO to maximise:
```
R = r(post, summary) − β · log( π_RL(summary|post) / π_SFT(summary|post) )
```
The second term is a **KL penalty**: it keeps the new policy close to the supervised one. Why? The reward model is imperfect. Without the penalty, the policy finds strange outputs that score high but are bad (**reward hacking**). The penalty also keeps outputs fluent and diverse.

**Topic 6: results.**
- Human-feedback models were preferred over the human reference summaries.
- A **1.3B** human-feedback model beat a **12.9B** supervised model.
- The models transferred to news articles (CNN/DailyMail) without news-specific training.
- **Over-optimisation:** as optimisation against the reward model increased, true human preference first rose, then **fell**.

**What we built and saw.** The full pipeline with tiny real networks: supervised training, comparisons, reward model, PPO with a KL penalty. Quality rose clearly. **Without** the KL penalty, the policy exploited the reward model by repeating the main topic: the reward went up while true quality went down.

**Why it matters.** This is the direct precursor of the RLHF recipe used for InstructGPT and ChatGPT (paper 067).

</details>

#### 067 · [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](10-Alignment/067-Ouyang-et-al-2022-InstructGPT/) — Ouyang et al., 2022
- **The idea:** the full **RLHF** recipe behind ChatGPT: learn from demonstrations, then rankings, then reinforcement learning, while mixing in pretraining so the model doesn't lose old skills.
- **What we built:** every stage with tiny real networks.
- **What we saw:** mixing in pretraining kept the new helpfulness while recovering lost skills. **Honest result:** reinforcement learning did not beat simple supervised training here, because our reward model was too weak.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** GPT-3 (paper 050) predicts internet text, but users want it to **follow instructions** helpfully, honestly and harmlessly. Often GPT-3 would continue a question with more questions, make things up, or produce toxic text. Long Ouyang and colleagues at OpenAI applied human-feedback training (paper 066) to general instructions, creating **InstructGPT**, the method behind ChatGPT.

**Topic 1: the three steps.**
1. **Supervised fine-tuning (SFT):** about 40 contractors wrote **demonstrations** of ideal responses to prompts (from the API and written by labellers), about 13,000 prompts. GPT-3 is fine-tuned on them.
2. **Reward model (RM):** for about 33,000 prompts, labellers **ranked** 4–9 model outputs from best to worst. Every pair from a ranking becomes a comparison (K outputs give K(K−1)/2 pairs). A 6B reward model was trained with the Bradley–Terry loss (paper 066).
3. **Reinforcement learning (PPO):** fine-tune the SFT model to maximise the reward model's score, with a KL penalty toward the SFT model, on about 31,000 prompts.

**Topic 2: the alignment tax and PPO-ptx.** RLHF made the model worse at some standard NLP benchmarks (it "forgot" some abilities): an **alignment tax**. The fix, **PPO-ptx**, mixes ordinary **pre-training gradients** (predicting next tokens on the original pre-training data) into the RL updates. This kept the benchmark performance with little loss in human preference.

**Topic 3: results.**
- Outputs from the **1.3B** InstructGPT were preferred over those of the **175B** GPT-3, despite being 100× smaller.
- InstructGPT was **more truthful** (on the TruthfulQA benchmark) and made up facts less often in closed-domain tasks (21% vs 41%).
- It was somewhat less toxic when asked to be respectful.
- It did **not** improve on bias benchmarks.
- It generalised to instructions in other languages and about code, which were rare in the fine-tuning data.

**Topic 4: whose preferences?** The paper is clear that the model is aligned to a specific group: the labellers, the researchers' instructions and the API customers. It is not aligned to "human values" in general.

**Topic 5: limitations.** It still makes simple mistakes, can be led into harmful output if the user asks, and sometimes hedges too much.

**What we built and saw.** Every stage with tiny real networks: SFT, a ranking-based reward model, PPO with a KL penalty, and PPO-ptx. Mixing in pre-training data kept the new helpful behaviour while recovering the skills RL had damaged. **Honest result:** in our toy, PPO did **not** beat plain SFT, because our small reward model was not accurate enough to improve on the demonstrations. This is a reminder that RLHF is only as good as its reward model.

**Why it matters.** This three-step recipe (SFT → reward model → RL) is how ChatGPT and most chat assistants were first made.

</details>

#### 068 · [Constitutional AI: Harmlessness from AI Feedback](10-Alignment/068-Bai-et-al-2022-Constitutional-AI/) — Bai et al., 2022
- **The idea:** instead of human labels, the AI critiques and revises its own answers using a written list of principles (a "constitution").
- **What we built:** the critique-and-revise loop and AI-generated preference labels, in a small simulated text world.
- **What we saw:** harmful answers fell from 100% to under 1% after a few revisions. Labels that rewarded dodging questions produced an evasive model, while the constitution produced one that explains instead.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** RLHF needs lots of human labels about harmful content. That is expensive and exposes workers to disturbing material. Also, RLHF-trained models often became **evasive**: they refused to engage with anything sensitive. Yuntao Bai and colleagues at Anthropic replaced most human harm labels with **AI feedback guided by written principles**, a "constitution".

**Topic 1: the constitution.** A short list of natural-language principles, for example: "Choose the response that is least harmful, unethical, racist, sexist, toxic, dangerous or illegal", or "…that a wise, ethical person would most likely say". At each step one principle is sampled at random (the paper used 16).

**Topic 2: phase 1, supervised learning from self-critique (SL-CAI).**
1. Give a **helpful-only** model a harmful "red-team" prompt. It produces a (possibly harmful) response.
2. Ask the model to **critique** its response according to a principle: "Identify specific ways in which the response is harmful…".
3. Ask it to **revise** the response to address the critique.
4. Repeat the critique and revision a few times.
5. Fine-tune a model on the final revised responses (plus helpful responses, to keep it helpful).

**Topic 3: phase 2, reinforcement learning from AI feedback (RLAIF).**
1. The phase-1 model generates **pairs** of responses to harmful prompts.
2. A model is asked which response is better according to a principle (as a multiple-choice question). Its probabilities become **soft preference labels**. Asking it to reason step by step first (chain-of-thought) improved its judgements.
3. Train a **preference model** on these AI labels (for harmlessness) plus **human** labels (for helpfulness).
4. Run RL against it, as in RLHF.

**Topic 4: harmless but not evasive.** The goal was a model that, when it declines, **explains why** and engages thoughtfully, instead of saying "I can't help with that". The CAI model was both more harmless **and** more helpful than an RLHF model trained on human harm labels: a Pareto improvement.

**Topic 5: transparency.** The training goals are written down in plain language, so they can be read, debated and changed, unlike thousands of individual labels.

**What we built and saw.** The critique → revise loop and AI-generated preference labels, in a small simulated text world with simple rule-based stand-ins for the model.
- The share of harmful answers fell from **100% to under 1%** after a few rounds of revision.
- Preference labels that rewarded refusing produced an **evasive** model.
- Constitution-style labels that rewarded engaging and explaining produced a model that **explains** instead.

**Why it matters.** Constitutional AI is how Anthropic's Claude models were trained. RLAIF (AI feedback) is now widely used across the industry to scale alignment.

</details>

#### 069 · [Direct Preference Optimization (DPO)](10-Alignment/069-Rafailov-et-al-2023-DPO/) — Rafailov et al., 2023
- **The idea:** **DPO** gets the same effect as RLHF with one simple training loss, no separate reward model and no reinforcement learning.
- **What we built:** DPO, a full RLHF baseline, and a task where the best possible result can be computed exactly.
- **What we saw:** DPO got close to the best possible (87–98%). **Honest result:** RLHF got even closer here (99–100%), so the paper's claim that DPO beats it was not reproduced on this toy.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** RLHF (papers 066–067) is complicated: train a reward model, then run reinforcement learning (PPO), which is unstable, needs many hyperparameters, and keeps several models in memory at once. Rafael Rafailov and colleagues found that the same goal can be reached with **one simple loss**, without a reward model and without RL: **Direct Preference Optimization** (DPO).

**Topic 1: the RLHF objective.** RLHF maximises reward while staying close to a reference model π_ref:
```
maximise  E[ r(x, y) ]  −  β · KL( π || π_ref )
```

**Topic 2: the best policy has a known form.** Mathematically, the solution of that objective is:
```
π*(y|x) = π_ref(y|x) · exp( r(x,y) / β ) / Z(x)
```
Start from the reference model and boost each answer by its exponentiated reward. Z(x) is a normalising constant that is hard to compute (it sums over all possible answers).

**Topic 3: flip it around.** Solve for the reward:
```
r(x, y) = β · log( π*(y|x) / π_ref(y|x) )  +  β · log Z(x)
```
Any policy implicitly defines a reward. The **policy is secretly a reward model**.

**Topic 4: plug it into the preference model.** The Bradley–Terry model (paper 066) uses only the **difference** r(y_w) − r(y_l) between the preferred (w) and rejected (l) answers. The troublesome β·log Z(x) term appears in both and **cancels**. That gives the DPO loss:
```
loss = −log σ( β · [ log π(y_w)/π_ref(y_w)  −  log π(y_l)/π_ref(y_l) ] )
```
This is just a classification loss on preference pairs: increase the probability of preferred answers and decrease rejected ones, each **relative to the reference model**.

**Worked example.** Take β = 0.1. Suppose the policy has raised the log-probability of the preferred answer by 2 (relative to the reference) and lowered the rejected one by 1.
- The inner value is 0.1 × (2 − (−1)) = 0.3.
- σ(0.3) = 0.574, so the loss is 0.55.

The gradient keeps pushing the two apart, and pushes harder on pairs the model currently gets **wrong**.

**Topic 5: experiments.**
- **Controlled sentiment generation:** DPO reached a better reward-vs-KL trade-off than PPO.
- **TL;DR summarisation:** DPO's win rate against reference summaries was about 61% vs about 57% for PPO (at their best sampling temperatures).
- **Single-turn dialogue** (the Anthropic HH data): DPO was the only efficient method that improved over the preferred completions in the data.

**What we built and saw.** DPO, a full RLHF baseline (reward model + policy-gradient RL), and a toy task where the best possible policy can be computed exactly. DPO got close to the optimum (87–98% of the best possible). **Honest difference:** on this small, clean toy, RLHF got even closer (99–100%), so the paper's claim that DPO beats PPO was not reproduced here.

**Why it matters.** DPO and its variants are now among the most popular ways to align open models, because they are much simpler and cheaper than PPO-based RLHF.

</details>

---

### Stage 11 · Retrieval: giving models knowledge they can look up

**The story.** Models forget, make things up, and go out of date. Retrieval lets a model search a collection of documents first and answer from what it finds. Search engines did this with keywords; these papers do it with meaning (embeddings), then connect search and answer-writing so both learn together. This is one of the most widely used patterns in AI applications today ("RAG").

#### 070 · [Dense Passage Retrieval for Open-Domain Question Answering](11-Retrieval-RAG/070-Karpukhin-et-al-2020-Dense-Passage-Retrieval/) — Karpukhin et al., 2020
- **The idea:** search by **meaning** using embeddings, instead of matching keywords.
- **What we built:** a toy Wikipedia, the classic keyword search (BM25), the embedding-based retriever, and a mix of the two.
- **What we saw:** for reworded questions, meaning-based search found the right page 82% of the time against 19% for keywords. But for names it had never seen, keywords won (83% vs 49%). Combining both was best everywhere.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** For open-domain question answering ("Who wrote Hamlet?"), a system must first **find** the relevant passages among millions. The classic method, **BM25**, matches keywords. But questions often use different words from the answer passage ("Who is the bad guy in Lord of the Rings?" vs "Sauron is the main antagonist…"). Vladimir Karpukhin and colleagues trained a **dense** retriever that matches by meaning.

**Topic 1: how BM25 works.** For each query word, a passage gets points that grow with:
- how often the word appears in the passage (**term frequency**, with diminishing returns: the 10th "Hamlet" adds little);
- how **rare** the word is across all passages (**inverse document frequency**: "Hamlet" is informative, "the" is not).

Longer passages are slightly penalised. BM25 is fast and strong for exact names, but blind to synonyms.

**Topic 2: dense passage retrieval (DPR).** Two BERT encoders (paper 037): one turns a question into a vector q, the other turns each passage into a vector p. Relevance is their dot product q·p. All 21 million Wikipedia passages (100-word chunks) are encoded **once in advance** and stored in an index (FAISS) that finds the highest dot products quickly.

**Topic 3: training with in-batch negatives.** Training data is questions paired with a passage containing the answer. For a batch of B questions and their B positive passages:
- compute all B×B dot products;
- each question's own passage is the positive, and the **other B − 1 passages in the batch are negatives**, for free.

**Example with B = 3.** Question 1 should score passage 1 above passages 2 and 3; question 2 should score passage 2 highest; and so on. This is a softmax classification over each row.

**Topic 4: hard negatives.** Adding one **BM25 hard negative** per question helped a lot: a passage that shares keywords with the question but doesn't contain the answer. It teaches the model to look past surface overlap.

**Topic 5: results.** On Natural Questions, the right passage was in the top 20 results **78.4%** of the time with DPR vs **59.1%** with BM25. Feeding better passages to a reader model improved end-to-end QA accuracy. Combining BM25 and DPR scores helped on some datasets.

**What we built and saw.** A toy Wikipedia, BM25, a dense dual encoder trained with in-batch and hard negatives, and a hybrid.
- For **reworded** questions, dense search found the right page 82% of the time vs 19% for BM25.
- For **names the model had never seen** in training, BM25 won (83% vs 49%): dense models struggle with rare exact terms.
- The **hybrid** was best everywhere.

**Why it matters.** Dense retrieval with embeddings is the "R" in most RAG systems today. Combining dense and keyword search ("hybrid search") is standard practice for exactly the reasons our experiment shows.

</details>

#### 071 · [Retrieval-Augmented Generation (RAG)](11-Retrieval-RAG/071-Lewis-et-al-2020-RAG/) — Lewis et al., 2020
- **The idea:** **RAG**: retrieve documents, then write the answer from them, training search and writing together.
- **What we built:** both versions of RAG from the paper, and swapping the document collection without retraining.
- **What we saw:** answering from memory got 9%, while RAG with a learned retriever got 58%. Swapping in an updated collection changed the answers correctly.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** Language models store knowledge in their weights. That knowledge is hard to update, hard to check, and they hallucinate. Patrick Lewis and colleagues combined a **retriever** (paper 070) with a **generator** (BART, a sequence-to-sequence transformer), and trained them together: **Retrieval-Augmented Generation** (RAG).

**Topic 1: the pipeline.**
1. Encode the question with DPR's question encoder.
2. Retrieve the top-k passages (k = 5 or 10) from a Wikipedia index of 21M passages.
3. For each passage z, the generator writes an answer **conditioned on the question plus that passage**.
4. Combine the results, weighting each passage by the retriever's probability p(z|x).

**Topic 2: two ways to combine passages.**
- **RAG-Sequence:** the whole answer is generated from **one** passage, and the probabilities are averaged over passages:
```
p(y|x) = Σ_z  p(z|x) · p(y | x, z)
```
- **RAG-Token:** each **token** can come from a different passage:
```
p(y|x) = Π_i  Σ_z  p(z|x) · p(y_i | x, z, y_<i)
```

**Worked example (RAG-Sequence).** Two passages with retrieval probabilities 0.7 and 0.3 give the answer "Paris" probability 0.9 and 0.2. The total is 0.7×0.9 + 0.3×0.2 = 0.69.

**Topic 3: training without retrieval labels.** Nobody says which passage is correct; training just maximises p(answer | question) through the sum above. Gradients flow into the **question encoder** too, so passages that help produce the answer become more likely to be retrieved. The **document encoder is frozen**, because re-encoding 21M passages after every update would be too expensive.

**Topic 4: hot-swapping knowledge.** Because knowledge lives in the index, you can **update it without retraining**. The paper swapped a 2016 Wikipedia index for a 2018 one and the model correctly answered "Who is the leader of X?" according to the new index.

**Topic 5: results.**
- New state of the art on open-domain QA benchmarks (Natural Questions, TriviaQA, WebQuestions, CuratedTREC).
- On generation tasks (Jeopardy question writing, MS MARCO), outputs were more factual and specific than BART alone.
- Retrieved passages also give a degree of **provenance**: you can see what the answer was based on.

**What we built and saw.** Both RAG variants, joint training of the question encoder and generator, and index swapping.
- Answering from memory alone (no retrieval) got 9%.
- RAG with a learned retriever got 58%.
- Swapping in an updated document collection changed the answers correctly with no retraining.

**Why it matters.** "RAG" is now one of the most common patterns in AI applications: chatbots over company documents, search assistants, and anything that must cite sources or stay up to date.

</details>

#### 072 · [Atlas: Few-shot Learning with Retrieval Augmented Language Models](11-Retrieval-RAG/072-Izacard-et-al-2022-Atlas/) — Izacard et al., 2022
- **The idea:** a retrieval model that learns new tasks from just a few examples, by pre-training the searcher and the reader together.
- **What we built:** the reader, four ways of training the retriever, joint pre-training and index compression.
- **What we saw:** with 64 examples, joint pre-training reached about 50% against 7% without it, and compressing the index 16 times lost almost nothing.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The goal.** Large models like GPT-3 can learn tasks from a few examples (paper 050) because they store a lot in their weights. Gautier Izacard and colleagues at Meta asked: can a **much smaller** model with **retrieval** do few-shot learning as well? Their model **Atlas** (11B parameters) did, beating a 540B model on some tasks.

**Topic 1: the reader, Fusion-in-Decoder (FiD).** A T5 encoder–decoder.
- Each retrieved passage is encoded **separately** together with the question (cheap, since each one is short).
- The decoder then attends over **all** the encoded passages at once, so it can **combine evidence** from many passages (for example 20–40).

**Topic 2: the retriever.** **Contriever**: a dense dual encoder (like DPR, paper 070) pre-trained without labels using contrastive learning on text.

**Topic 3: training the retriever from the reader's needs.** There are no labels saying which passages are useful. The idea is to let the **reader** tell the retriever. Four losses were compared:
- **Attention Distillation:** passages the reader pays more attention to should be ranked higher.
- **EMDR²:** treat the passage as a hidden variable and maximise the expected likelihood (similar to RAG, paper 071).
- **Perplexity Distillation:** passages that make the correct answer more likely (lower perplexity) should rank higher.
- **Leave-One-Out Perplexity Distillation (LOOP):** how much worse the answer becomes when a passage is **removed**.

The simpler perplexity-based losses worked as well as or better than the others.

**Topic 4: joint pre-training.** Before few-shot fine-tuning, Atlas pre-trains reader and retriever **together** on unlabelled text with pretext tasks:
- **prefix language modelling** (continue a text);
- **masked language modelling** (fill in masked spans, like T5);
- **title-to-section generation.**

Retrieval is trained during pre-training too. This joint pre-training was **critical** for few-shot performance.

**Topic 5: keeping the index fresh and small.**
- When the retriever changes, the passage embeddings go stale. Options: re-index periodically, re-rank a larger candidate set, or update only the query encoder.
- **Product quantisation** compresses the index (for example from 587 GB to 50 GB) with little loss.

**Topic 6: results.** With just **64 training examples**, Atlas reached over **42%** accuracy on Natural Questions, beating PaLM (540B) by about 3 points despite having 50× fewer parameters. It was also strong on MMLU, fact checking (FEVER) and KILT. Like RAG, its index can be updated or swapped.

**What we built and saw.** A FiD reader, four retriever-training losses, joint pre-training and index compression.
- With 64 examples, joint pre-training reached about **50%** vs **7%** without it.
- Compressing the index 16× lost almost nothing.

**Why it matters.** Atlas showed that **knowledge doesn't have to live in parameters**: retrieval lets smaller, cheaper models compete with giant ones and stay up to date.

</details>

---

### Stage 12 · Efficient fine-tuning and quantization: big models on small hardware

**The story.** Large models need huge amounts of memory. This stage covers two families of tricks used every day by AI engineers. **LoRA** adapts a model by training a tiny add-on instead of all its weights. **Quantization** stores the model's numbers in 8 or 4 bits instead of 16 or 32. Combined (QLoRA), they let you fine-tune a huge model on a single graphics card.

#### 073 · [LoRA: Low-Rank Adaptation of Large Language Models](12-Efficient-Finetuning-and-Quantization/073-Hu-et-al-2022-LoRA/) — Hu et al., 2022
- **The idea:** **LoRA**: freeze the big model and train a tiny "patch" made of two thin matrices. Far fewer numbers to train and store.
- **What we built:** LoRA on our tiny LLaMA, teaching it a new task.
- **What we saw:** with 4,096 trainable numbers LoRA got 97.4%, against 99.8% for full fine-tuning with 133,000. Switching the patch off brought back the original skill that full fine-tuning had destroyed.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** Fine-tuning a large model changes **all** its weights. For GPT-3 that is 175 billion numbers per task: you need huge memory to train, and you must store a full copy of the model for every task. Edward Hu and colleagues at Microsoft proposed **LoRA** (Low-Rank Adaptation).

**Topic 1: the low-rank idea.** Fine-tuning changes a weight matrix W into W + ΔW. The hypothesis is that the change ΔW has **low intrinsic rank**: it can be written as the product of two thin matrices:
```
ΔW = B · A          W: d × k,   B: d × r,   A: r × k,   with r much smaller than d, k
```
The output becomes h = W·x + B·A·x. **W is frozen**; only A and B are trained.

**Topic 2: the savings.** For d = k = 4,096 and rank r = 8:
- full ΔW: 4,096 × 4,096 = **16.8 million** numbers;
- LoRA: 4,096 × 8 + 8 × 4,096 = **65,536** numbers, which is **256× fewer**.

For GPT-3, LoRA reduced trainable parameters by **10,000×** and GPU memory by about 3×.

**Topic 3: initialisation and scaling.**
- A starts random (Gaussian) and **B starts at zero**, so BA = 0 and training begins **exactly** at the pre-trained model.
- The update is scaled by α/r, so changing r doesn't require re-tuning the learning rate.

**Topic 4: no extra cost at inference.** After training, **merge** W′ = W + BA. The model has exactly the same size and speed as before. To switch tasks, subtract BA and add another task's B′A′. Each task's adapter is tiny (a few MB).

**Topic 5: where to apply it.** In transformers, the paper applied LoRA to the attention matrices, especially the **query (W_q) and value (W_v)** projections. With a fixed budget, adapting several matrices with a small rank beat adapting one matrix with a large rank. Ranks as low as 1–4 were often enough.

**Topic 6: what ΔW learns.** Analysis showed that ΔW **amplifies directions that were already in W but not emphasised**: features useful for the task that pre-training had learned but given little weight.

**What we built and saw.** LoRA layers (with merge, unmerge and on/off) added to our tiny LLaMA (paper 056), taught to sort in descending order.
- LoRA with rank 8 (4,096 trainable numbers) reached 97.4% vs 99.8% for full fine-tuning (133,000 numbers).
- Rank 1 was too small (47%).
- **Switching the adapter off restored the original skill**, which full fine-tuning had destroyed.

**Why it matters.** LoRA is the most popular way to fine-tune large models. It is how most people customise open LLMs and image models (Stable Diffusion "LoRAs").

</details>

#### 074 · [SmoothQuant](12-Efficient-Finetuning-and-Quantization/074-Xiao-et-al-2023-SmoothQuant/) — Xiao et al., 2023
- **The idea:** a few huge values inside the model ruin 8-bit storage. **SmoothQuant** moves that difficulty from the activations to the weights, where it's easier to handle.
- **What we built:** 8-bit quantisation at several levels of detail, and the smoothing trick.
- **What we saw:** naive 8-bit collapsed to about 6–8% accuracy, while SmoothQuant kept the full-precision 98.9%.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** **Quantisation** stores numbers with fewer bits, for example 8-bit integers (INT8) instead of 16-bit floats. That halves memory and lets fast integer hardware do the maths. But large language models (above about 6.7B parameters) develop **outliers**: a few activation channels with values **100× larger** than the rest. Quantising activations then destroys accuracy. Guangxuan Xiao and colleagues proposed **SmoothQuant**.

**Topic 1: how INT8 quantisation works.** Choose a scale s = max|x| / 127, then store round(x / s) as an integer from −127 to 127.

**Example.** x = [0.1, −0.5, 2.0]. Then max = 2.0 and s = 0.01575, so the stored values are [6, −32, 127]. Converted back they give [0.095, −0.504, 2.0]: small errors.

**Topic 2: why outliers hurt.** Suppose one value is 100 and the others are around 0.1. Then s = 100/127 ≈ 0.79, and every small value rounds to **0**: all their information is lost.

**Topic 3: weights are easy, activations are hard.**
- **Weights** have a smooth, narrow range: easy to quantise.
- **Activations** have outlier channels, but the outliers are **consistent**: always the same channels.

Ideally we'd use a separate scale per activation channel, but fast INT8 matrix-multiply kernels can't do that efficiently.

**Topic 4: the smoothing trick.** Multiplication allows moving a factor from one side to the other without changing the result:
```
Y = X · W = ( X · diag(s)⁻¹ ) · ( diag(s) · W )
```
Divide each activation channel j by s_j (making the outliers smaller) and multiply the matching row of W by s_j (making those weights larger). Now **both** are moderately easy to quantise. The paper's choice:
```
s_j = max|X_j|^α / max|W_j|^(1−α)        with α = 0.5 (migration strength) usually
```
The division by s can be folded into the previous layer (for example LayerNorm), so it costs **nothing** at run time.

**Topic 5: results.**
- **W8A8** (8-bit weights **and** activations) for models up to OPT-175B and BLOOM-176B, with almost no accuracy loss.
- Up to **1.56× speed-up** and **2× memory reduction**.
- No retraining needed: only a small calibration set to measure the activation ranges.

**What we built and saw.** INT8 quantisation at different levels of detail (per-tensor, per-token, per-channel) and the smoothing transformation, on a tiny LLaMA with planted outlier channels. Naive W8A8 **collapsed** to about 6–8% accuracy. SmoothQuant kept the full-precision **98.9%**.

**Why it matters.** SmoothQuant's idea of migrating difficulty between activations and weights is used in production inference engines (TensorRT-LLM and others).

</details>

#### 075 · [GPTQ: Accurate Post-Training Quantization](12-Efficient-Finetuning-and-Quantization/075-Frantar-et-al-2023-GPTQ/) — Frantar et al., 2023
- **The idea:** **GPTQ** rounds weights to 4 (or fewer) bits one at a time, adjusting the remaining weights to make up for each rounding error.
- **What we built:** GPTQ, the simple rounding baseline, and the slow exact method it approximates.
- **What we saw:** at a harsh 2 bits, simple rounding fell to 75.5% while GPTQ kept 97.7% (full precision: 98.6%). The fast method gave exactly the same result as the slow one.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** To run a huge model on less hardware, quantise its **weights** to 3–4 bits. The simplest method, **round-to-nearest** (RTN), rounds every weight independently. At 3–4 bits, the rounding errors add up and quality drops badly. Elias Frantar and colleagues proposed **GPTQ**, which quantises a 175B model in a few GPU hours with very little loss.

**Topic 1: the per-layer objective.** Quantise one layer at a time. What matters is the layer's **output**, not the weights themselves. With X the layer's inputs on some calibration data:
```
minimise  || W·X − Ŵ·X ||²          Ŵ = the quantised weights
```

**Topic 2: compensating errors (the key idea).** When one weight is rounded, its error can be **compensated** by adjusting the not-yet-quantised weights, using how the inputs are correlated (captured by the "Hessian" H = 2·X·Xᵀ).

**Toy example.** A neuron computes w1·x1 + w2·x2, where in the data x1 and x2 are always **equal**. Weights w = [0.3, 0.3] give output 0.6·x. Allowed values are 0 or 1.
- **RTN** rounds both to 0, so the output is 0: error 0.6·x.
- **Compensation:** round w1 to 0 (error −0.3) and add 0.3 to w2, making it 0.6, which rounds to 1. The output is 1·x: error 0.4·x, which is smaller.

**Topic 3: the formula (from Optimal Brain Quantisation).** After quantising weight q, update the remaining weights:
```
δ = − ( w_q − quant(w_q) ) / [H⁻¹]_qq  ·  (H⁻¹)_{:, q}
```
Then remove q from the problem and continue. The original method (OBQ) picked the best weight order and was far too slow for big models.

**Topic 4: GPTQ's three speed-ups.**
1. **Fixed order:** quantise columns left to right, the same order for every row. The order barely matters for big layers, and every row can share the same H⁻¹ updates.
2. **Lazy batch updates:** process 128 columns at a time and apply the accumulated updates to the rest in one big matrix operation (efficient on GPUs).
3. **Cholesky reformulation:** compute the needed H⁻¹ rows once, with a numerically stable decomposition, avoiding error build-up.

**Topic 5: results.**
- OPT-175B and BLOOM-176B quantised in about **4 GPU hours**.
- At 4 bits, a negligible perplexity increase; at 3 bits, a small one (RTN collapses).
- The 175B model runs on a **single** 80 GB GPU at 3 bits, with 3–4.5× faster generation thanks to custom kernels.

**What we built and saw.** GPTQ, RTN and the slow exact OBQ, on our tiny LLaMA.
- At a harsh **2 bits**, RTN fell to 75.5%, while GPTQ kept **97.7%** (full precision: 98.6%).
- The fast GPTQ algorithm gave **exactly** the same result as the slow textbook version with the same order.

**Why it matters.** GPTQ is one of the most widely used methods to quantise open LLMs to 4 bits (you will see "GPTQ" model files everywhere), making big models usable on consumer GPUs.

</details>

#### 076 · [AWQ: Activation-aware Weight Quantization](12-Efficient-Finetuning-and-Quantization/076-Lin-et-al-2023-AWQ/) — Lin et al., 2023
- **The idea:** a small fraction of weights matter most, the ones connected to large activations. Protect them by scaling before rounding.
- **What we built:** AWQ's search for the best scaling, and the comparison with GPTQ.
- **What we saw:** protecting just 3 important channels rescued 4-bit accuracy from 27% to 99%, and AWQ kept 98%.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The insight.** Ji Lin and colleagues (MIT) observed that **not all weights matter equally**: about 0.1–1% of weight channels are **salient** (very important). Keeping just those in full precision preserves almost all the accuracy. The surprise is in how to find them: by looking at the **activations**, not the weights.

**Topic 1: weight-only quantisation with groups.** AWQ quantises **weights only** (to 4 bits, with activations kept at 16 bits, "W4A16"). This suits generation, where loading the weights from memory is the bottleneck. Weights are quantised in **groups** (for example 128 weights share one scale), which keeps errors smaller than one scale per whole row.

**Topic 2: salient channels are found by activation size.** An input channel that usually carries **large activations** multiplies its weights by large numbers, so errors in those weights are amplified. Choosing the 1% of channels with the largest **average activation** and keeping them in FP16 restored accuracy. Choosing by **weight** size was no better than random.

**Topic 3: avoid mixed precision with scaling.** Mixed precision (some weights at 16 bits, the rest at 4) is awkward for hardware. Instead, **scale up** the salient weights before quantising and scale down the matching activations by the same factor (the product is unchanged, as in SmoothQuant, paper 074):
```
Q(w · s) · (x / s)
```
The rounding error of a weight is at most half a quantisation step Δ. Scaling the weight by s doesn't change Δ much (other weights in the group set it), but the error is then divided by s when the activation is scaled back. So the **relative error on salient weights shrinks by about s**.

**Topic 4: searching the scale.** AWQ sets s = (average activation size)^α per channel, and grid-searches α in [0, 1] to minimise the layer's output error on a small calibration set. There is no backpropagation and no reconstruction of weights, so it doesn't overfit the calibration data, and it works across domains (including instruction-tuned and multimodal models). Optional **weight clipping** (shrinking the range slightly) helps further.

**Topic 5: results.** Better 4-bit (and 3-bit) accuracy than RTN, and often better than GPTQ, especially on instruction-following and multimodal models. A companion inference engine (TinyChat) gave about **3× speed-ups** over FP16, even on laptops and edge devices.

**What we built and saw.** Group quantisation, the "keep 1% in FP16" experiments, fixed scaling, the AWQ search and clipping, on a tiny LLaMA with planted outlier channels.
- Keeping just **3 activation-salient channels** in full precision rescued 4-bit accuracy from 27% to 99%; choosing by weight size did not help.
- AWQ reached 98% at 4 bits.
- **Honest notes:** the best fixed scale was 8 rather than the paper's 2 on our model, and GPTQ did poorly on our extreme-outlier model.

**Why it matters.** AWQ is one of the standard 4-bit formats for open LLMs ("AWQ" model files), supported by vLLM and other inference servers.

</details>

#### 077 · [QLoRA: Efficient Finetuning of Quantized LLMs](12-Efficient-Finetuning-and-Quantization/077-Dettmers-et-al-2023-QLoRA/) — Dettmers et al., 2023
- **The idea:** **QLoRA**: store the frozen model in 4 bits (with a new number format, "NF4") and train LoRA patches on top. A 65-billion-parameter model can then be fine-tuned on one GPU.
- **What we built:** the NF4 format, double quantisation and QLoRA on our tiny model.
- **What we saw:** QLoRA matched full 16-bit fine-tuning (99.3% vs 99.7%), and for a 65B model the memory need falls from about 786 GB to about 43 GB.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The goal.** Fine-tune a **65B** model on a **single 48 GB GPU** without losing quality. Normal 16-bit fine-tuning of a 65B model needs over 780 GB of GPU memory. Tim Dettmers and colleagues combined 4-bit quantisation with LoRA (paper 073): **QLoRA**.

**Topic 1: the basic recipe.**
- Store the frozen base model in **4 bits**.
- Add **LoRA adapters** in 16-bit and train only those.
- During the forward and backward passes, each 4-bit weight block is temporarily **de-quantised** to 16-bit for the computation.

Gradients flow **through** the frozen 4-bit weights into the adapters.

**Topic 2: NormalFloat-4 (NF4), a smarter 4-bit format.** 4 bits allow only 16 values. Where should they go? Trained weights are roughly **bell-shaped** (normally distributed). NF4 places its 16 values at **quantiles** of the normal distribution, so each value is used about equally often: values are dense near zero, where most weights are, and sparse in the tails. This is "information-theoretically optimal" for normal data (each code is equally likely). NF4 also has an exact zero. Weights are scaled into [−1, 1] per **block of 64** using the block's largest absolute value.

**Topic 3: double quantisation.** Each block of 64 weights needs its own 32-bit scale: 32/64 = **0.5 extra bits per parameter**. QLoRA quantises the **scales themselves** to 8 bits, in blocks of 256:
```
8/64 + 32/(64 × 256) = 0.125 + 0.002 ≈ 0.127 bits per parameter
```
That saves about 0.37 bits per parameter, roughly **3 GB** on a 65B model.

**Topic 4: paged optimisers.** Memory spikes (for example from long sequences) can crash training. Paged optimisers use NVIDIA unified memory to **page** optimiser states to CPU RAM when the GPU runs out, like virtual memory.

**Topic 5: LoRA everywhere.** Matching full 16-bit fine-tuning required LoRA on **all** linear layers, not just the query and value matrices.

**Topic 6: results.**
- QLoRA matched 16-bit full fine-tuning and 16-bit LoRA on benchmarks.
- The **Guanaco** models, fine-tuned with QLoRA on the OASST1 chat dataset, reached 99.3% of ChatGPT's performance on the Vicuna benchmark (as judged by GPT-4) after 24 hours on one GPU.
- The paper also found that data **quality** matters far more than quantity for chat fine-tuning.

**What we built and saw.** NF4 (matching the paper's table of values), other 4-bit formats, block-wise quantisation, double quantisation and QLoRA on our tiny model.
- QLoRA reached **99.3%** vs **99.7%** for 16-bit LoRA.
- LoRA only on q and v reached 94.9%, so all layers are needed.
- For a 65B model, our memory calculation gives about **786 GB** for full fine-tuning vs **43 GB** for QLoRA.

**Why it matters.** QLoRA made fine-tuning large models possible for individuals and small labs. It is the default in many fine-tuning tools.

</details>

#### 078 · [LLM-QAT: Data-Free Quantization Aware Training](12-Efficient-Finetuning-and-Quantization/078-Liu-et-al-2023-LLM-QAT/) — Liu et al., 2023
- **The idea:** train the model *while* it is quantised so it adapts, using text the model generates itself as training data.
- **What we built:** quantisation-aware training and self-generated training data.
- **What we saw:** at very low precision, simple rounding fell to 2.5%, while training on self-generated text recovered 96.7%.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** **Post-training quantisation** (papers 074–077) rounds a finished model. At very low precision (4 bits for weights, activations **and** the KV cache), quality collapses. **Quantisation-aware training** (QAT) trains the model **while** it is quantised so it can adapt. But QAT needs training data, and the original pre-training data of an LLM is often unavailable or too large. Zechun Liu and colleagues at Meta proposed **data-free** QAT for LLMs: **LLM-QAT**.

**Topic 1: fake quantisation.** During training, weights and activations are quantised in the forward pass (rounded to the low-bit grid and converted back), so the model "feels" the rounding error. The full-precision weights are kept and updated.

**Topic 2: the straight-through estimator (STE).** Rounding has a gradient of zero almost everywhere (a staircase is flat), so no learning signal could pass through it. The STE simply **pretends rounding is the identity** in the backward pass: the gradient flows straight through. Example: round(2.3) = 2 in the forward pass, but in the backward pass we treat d(round(x))/dx as 1.

**Topic 3: generating its own training data.** Instead of real text, LLM-QAT asks the **full-precision model** to generate text:
1. start from a random first token;
2. for the first few tokens, pick the most likely next tokens (greedy, for coherence);
3. after that, **sample** (for diversity).

This synthetic data reflects what the model "knows", in its own distribution, and worked better than using an existing dataset like C4.

**Topic 4: knowledge distillation.** The quantised model (the student) is trained to match the full-precision model's (the teacher's) **output probabilities** on that data, using a soft cross-entropy. Matching the full distribution gives much richer information than matching single next tokens.

**Topic 5: quantising the KV cache too.** For long contexts the KV cache (paper 035) dominates memory, so LLM-QAT also quantises the keys and values, which post-training methods struggled with.

**Topic 6: MinMax rather than clipping.** For LLMs, outliers carry important information (paper 074), so LLM-QAT uses the full range (MinMax), not clipped ranges, for weights and activations.

**Topic 7: results.** On LLaMA 7B–30B, LLM-QAT clearly beat post-training methods at low precision, for example 4-bit weights with a 4-bit KV cache, where post-training methods dropped sharply.

**What we built and saw.** Fake quantisation with STE, self-generated data, distillation and KV-cache quantisation, on our tiny LLaMA. At very low precision, simple post-training rounding fell to **2.5%**, while QAT on self-generated text recovered **96.7%**.

**Why it matters.** QAT is used to make models that run well at very low precision on phones and other edge devices, and "the model generates its own training data" is a useful trick when real data isn't available.

</details>

---

### Stage 13 · Systems: making training and inference fast

**The story.** Big models are only possible because of clever engineering: computing attention without wasting memory, splitting a model across many chips, and coordinating thousands of computers. This stage covers those systems ideas, from early distributed training to the techniques used to train today's largest models.

#### 079 · [FlashAttention](13-Systems-Inference-and-Training/079-Dao-et-al-2022-FlashAttention/) — Dao et al., 2022
- **The idea:** compute attention in small tiles that fit in the chip's fast memory, avoiding slow trips to main memory. Exact same answer, much faster.
- **What we built:** standard and tiled attention, with a counter for memory traffic.
- **What we saw:** identical results, with about 9 times less memory traffic at length 1,024 (and the traffic scaling as the paper's theory says).

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** Attention (paper 034) is slow and memory-hungry for long sequences. Most people thought the fix was to compute **less** (approximate or sparse attention). Tri Dao and colleagues showed that the real bottleneck is **memory traffic**, and that exact attention can be made much faster by moving data more cleverly.

**Topic 1: the GPU memory hierarchy.**
- **HBM** (main GPU memory): large (40–80 GB) but relatively slow, about 1.5–2 TB/s.
- **SRAM** (on-chip memory): tiny (about 20 MB in total) but about 10× faster.

Computation is very fast; **reading and writing HBM** is often what takes the time.

**Topic 2: standard attention wastes memory traffic.** With n tokens:
1. compute S = QKᵀ (an n×n matrix) and **write it to HBM**;
2. read S, compute the softmax P, **write P to HBM**;
3. read P, compute P·V.

For n = 4,096 that is 16.8 million numbers per head written and read several times.

**Topic 3: tiling.** Split Q, K and V into blocks that fit in SRAM. For each block of queries, loop over the blocks of keys and values, computing the partial attention entirely in fast memory. The full n×n matrix is **never written** to HBM.

**Topic 4: the online softmax trick.** Softmax needs the maximum and the sum over **all** scores, but we only see one block at a time. Keep a running maximum m and a running sum ℓ, and **rescale** previous results whenever the maximum changes.

**Worked example.** Scores arrive as the block [1, 3], then the block [2].
- After block 1: m = 3 and ℓ = e^(1−3) + e^(3−3) = 0.135 + 1 = 1.135.
- Block 2's maximum is 2 < 3, so m stays 3 and ℓ = 1.135 + e^(2−3) = 1.135 + 0.368 = 1.503.

This is exactly the softmax denominator computed all at once (relative to the maximum 3). The partial outputs are rescaled the same way.

**Topic 5: the backward pass by recomputation.** Instead of storing the n×n matrix P for backpropagation, store only m and ℓ and **recompute** the attention blocks during the backward pass. That is more arithmetic but far less memory traffic, so it is still faster.

**Topic 6: results.**
- **Exact** attention (identical results, not an approximation).
- Memory use grows **linearly** in sequence length instead of quadratically.
- Speed-ups: about 15% for BERT-large and 3× for GPT-2 training.
- It enabled much longer contexts (the first transformers to solve the Path-X task with 16K tokens).
- The paper proves the HBM traffic is O(n²d²/M) (M = SRAM size) instead of O(n² + nd), and that no exact method can do much better.

**What we built and saw.** Standard and tiled attention with an HBM-traffic counter. Results were **identical**, with about **9×** less memory traffic at n = 1,024, and the traffic scaled as the paper's formula predicts.

**Why it matters.** FlashAttention (and its successors FlashAttention-2/3) is used in essentially every modern LLM training and inference system. It is the main reason long-context models are practical.

</details>

#### 080 · [Efficiently Scaling Transformer Inference](13-Systems-Inference-and-Training/080-Pope-et-al-2022-Efficiently-Scaling-Transformer-Inference/) — Pope et al., 2022
- **The idea:** how to split a huge model across many chips to answer requests quickly and cheaply.
- **What we built:** the paper's cost model and a simulation of the different ways of splitting the work.
- **What we saw:** we reproduced the paper's table of maximum context lengths (1,333 / 666 / 42,654 vs its 1,320 / 660 / 43,000).

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** Serving a 500-billion-parameter model is hard: it doesn't fit on one chip, users want low **latency** (fast replies), and providers want high **throughput** (many users per chip). Reiner Pope and colleagues at Google showed how to partition PaLM (540B) across TPU chips for efficient inference.

**Topic 1: two phases of inference.**
- **Prefill:** process the whole prompt at once. Lots of parallel work, so it is **compute-bound**.
- **Decode:** generate one token at a time. Each step must read all the weights and the KV cache but does little arithmetic, so it is **memory-bandwidth-bound**.

**Topic 2: arithmetic intensity, a simple calculation.** For one token, a model with N parameters does about 2N arithmetic operations and must read 2N bytes of weights (16-bit). That is 1 operation per byte. Chips can do hundreds of operations per byte read, so at batch size 1 the chip mostly **waits for memory**. Batching B users shares one weight read across B tokens, giving B operations per byte. Bigger batches give better efficiency but more latency per user.

**Topic 3: partitioning the feed-forward layers.**
- **1D weight-stationary:** each chip keeps a slice of every weight matrix; activations are passed around.
- **2D weight-stationary:** weights are split along both dimensions across a 2D grid of chips, which needs less communication at large chip counts.
- **Weight-gathered:** for very large batches, move the **weights** to the activations instead (all-gather the weights), because then the activations are bigger than the weights.

The best choice depends on batch size and number of chips; the paper gives an analytical cost model to choose.

**Topic 4: partitioning attention with multi-query attention.** With MQA (paper 035), there is only one K/V head, so splitting by heads doesn't help the KV cache. Instead, split the KV cache **by batch** (each chip holds the cache for some sequences). This reduces memory per chip and allows much longer contexts.

**Topic 5: other techniques.**
- Int8 weight quantisation.
- A "parallel" attention + FFN block formulation.
- Low-level optimisations to overlap communication and computation.

**Topic 6: results.**
- PaLM 540B on 64 TPU v4 chips: about **29 ms per token** at low batch size (with int8 weights), for interactive use.
- **76% model FLOPs utilisation** (MFU) for large-batch prefill.
- Long contexts are supported thanks to the MQA layout.

**What we built and saw.** The paper's analytical cost model and a simulator of the partitioning strategies. We reproduced the paper's table of **maximum context lengths** (1,333 / 666 / 42,654 vs the paper's 1,320 / 660 / 43,000), showing the big gain from batch-sharded multi-query attention.

**Why it matters.** This paper is a standard reference for how LLM serving systems choose parallelism, and its prefill/decode and memory-bound analysis is used throughout inference engineering.

</details>

#### 081 · [Parallelized Stochastic Gradient Descent](13-Systems-Inference-and-Training/081-Zinkevich-et-al-2010-Parallelized-SGD/) — Zinkevich et al., 2010
- **The idea:** train separate copies of a model on different machines with no communication, then simply average them at the end.
- **What we built:** the method with many simulated machines, plus checks of its theory.
- **What we saw:** going from 1 to 10 machines helped far more than going from 10 to 100.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** Training on huge datasets is slow on one machine. Usual parallel training requires machines to **communicate** constantly (sharing gradients or parameters), which is expensive. Martin Zinkevich and colleagues proposed the simplest possible alternative and analysed it.

**Topic 1: the algorithm (SimuParallelSGD).**
1. Randomly split the data among k machines.
2. Each machine runs ordinary **SGD** on its own share, completely independently, with **no communication**.
3. At the end, **average** the k resulting models.

That's it: one communication step, at the very end.

**Topic 2: why averaging helps.** Each machine's SGD ends up in a slightly random place near the optimum (SGD with a fixed step size keeps bouncing around). Think of each machine's result as a sample from a "cloud" around the solution. Averaging k independent samples reduces the spread (variance) by about k, giving a more accurate result than any single machine.

**Topic 3: the theory (in plain terms).** The paper treats SGD as a random process that converges to a **stationary distribution** (the cloud above), using the idea of **contraction mappings**: each SGD step pulls points closer together. It proves that the cloud converges quickly and bounds how far its centre is from the true optimum.

**Topic 4: diminishing returns.** Averaging reduces the **variance** (the random spread), but not the **bias**: the systematic offset of the cloud's centre from the true optimum, caused by the fixed step size. So:
- going from 1 to 10 machines removes most of the variance, a big gain;
- going from 10 to 100 can't remove the bias, so the gain is small.

**Topic 5: assumptions.** The analysis assumes a convex, smooth problem (such as logistic regression or linear SVMs) and a small enough step size. For deep networks, simple averaging of independently trained models generally doesn't work (they end up in different "basins"), which is why modern training communicates often.

**What we built and saw.** The algorithm with many simulated machines, plus checks of the theory (contraction and variance reduction). Going from 1 to 10 machines improved the result far more than going from 10 to 100, exactly the bias–variance story.

**Why it matters.** It is a foundational paper on communication-efficient learning. Its idea of averaging locally trained models is the basis of **federated learning** (FedAvg) and "local SGD", which average more often.

</details>

#### 082 · [TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](13-Systems-Inference-and-Training/082-Abadi-et-al-2015-TensorFlow-Whitepaper/) — Abadi et al., 2015
- **The idea:** describe a computation as a graph of operations, and let the system work out gradients and where each piece should run.
- **What we built:** a miniature TensorFlow from scratch, with automatic gradients and device placement.
- **What we saw:** exact gradients, and smart placement ran 3 times faster (4.6 vs 15.7 ms) with identical results.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**What it is.** The 2015 white paper introducing **TensorFlow**, Google's system for expressing and running machine learning computations on anything from phones to clusters of GPUs. It explains the design ideas, which are now common to all deep learning frameworks.

**Topic 1: computations as dataflow graphs.** A program is a **graph**:
- **nodes** are operations (matrix multiply, add, ReLU…);
- **edges** carry **tensors** (multi-dimensional arrays) between them;
- special **control edges** enforce an order without passing data.

Building the graph first and running it later lets the system **optimise** and **distribute** the whole computation.

**Topic 2: variables and sessions.**
- **Variables** hold state (like weights) that persists across runs.
- A **Session** runs the graph. You choose which outputs to **fetch** and which inputs to **feed**, and TensorFlow runs only the part of the graph needed for them (**partial execution**).

**Topic 3: automatic differentiation by extending the graph.** To get gradients, TensorFlow walks backwards from the loss and **adds new nodes** that compute each gradient using the chain rule (reverse-mode automatic differentiation). Gradients are just more graph.

**Worked example.** f = a·b + b with a = 2, b = 3, so f = 9.
- ∂f/∂a = b = 3.
- ∂f/∂b = a + 1 = 3 (b is used twice, so its two contributions **add up**).

**Topic 4: devices and placement.** Each operation must run on some device (CPU or GPU, on one machine or many). A **placement algorithm** simulates running the graph with a **cost model** (estimated compute time and data-transfer time) and greedily puts each node where it would finish earliest, respecting constraints (some operations only exist on CPU).

**Topic 5: send/receive across devices.** When an edge crosses devices, TensorFlow replaces it with a **Send** node on one side and a **Receive** node on the other. Each tensor is sent only once per device, even if several nodes need it.

**Topic 6: optimisations and extras.**
- **Common subexpression elimination:** identical computations are merged.
- **Lossy compression:** 32-bit floats are cut to 16 bits when sent between machines (dropping low mantissa bits).
- **Control flow** with Switch/Merge nodes (for if/while inside graphs).
- Queues for input pipelines, and **TensorBoard** for visualisation.

**What we built and saw.** A miniature TensorFlow from scratch: graph, variables, sessions with partial execution, automatic gradients by graph extension, CSE, cost-model placement and Send/Recv partitioning.
- Gradients matched numerical estimates to 3×10⁻¹¹.
- Smart placement ran in 4.6 ms vs 15.7 ms with everything on CPU, with identical results.

**Why it matters.** TensorFlow (and later PyTorch) made deep learning accessible. Concepts like computational graphs, autodiff, devices and placement remain the foundations of every framework, including JAX and compilers like XLA.

</details>

#### 083 · [TensorFlow: A System for Large-Scale Machine Learning](13-Systems-Inference-and-Training/083-Abadi-et-al-2016-TensorFlow-OSDI/) — Abadi et al., 2016
- **The idea:** the production design of TensorFlow, including splitting huge tables across machines and using spare "backup" workers to avoid waiting for slow machines.
- **What we built:** these features on top of our mini-TensorFlow, plus a simulator of slow machines.
- **What we saw:** backup workers sped training up, with the best number close to the paper's.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**What it is.** The 2016 systems paper describing TensorFlow's production design: how it supports large-scale training across many machines more flexibly than earlier "parameter server" systems (such as Google's DistBelief).

**Topic 1: the parameter server model and its limits.** In the older design:
- **parameter servers** hold the model's weights;
- **workers** compute gradients on data and send updates.

The update rules lived inside the servers, so trying a new optimiser meant changing the system itself.

**Topic 2: unified dataflow with mutable state.** TensorFlow represents **everything**, including parameters, update rules and input pipelines, as one dataflow graph with mutable variables. A "parameter server" is just a set of graph nodes placed on certain machines. Researchers can write new optimisers or model-parallel schemes in user code.

**Topic 3: sparse embeddings at scale.** Models with huge embedding tables (for example millions of words or users) don't fit on one machine. The table is **sharded** across servers:
- a **Part** operation splits the requested IDs by shard;
- **Gather** fetches the rows from each shard;
- **Stitch** reassembles them in order.

The gradient only updates the rows that were used.

**Topic 4: synchronous training with backup workers.** Two styles:
- **Asynchronous:** each worker updates whenever it's done. Fast, but updates use **stale** parameters, which hurts accuracy.
- **Synchronous:** wait for all workers, then update together. Accurate, but you wait for the **slowest** worker (a "straggler").

The paper's fix: run n + b workers and update as soon as the **first n** gradients arrive, ignoring the stragglers. A few **backup workers** make synchronous training fast. Example: with 50 workers where one or two are often slow, waiting for the first 50 of 54 avoids most of the delay.

**Topic 5: fault tolerance.** Long training jobs will see machine failures. TensorFlow uses user-level **checkpointing** (Save/Restore operations in the graph) rather than complex automatic recovery.

**Topic 6: results.** Good scaling on image classification (Inception) and language modelling, and widespread adoption inside Google and outside.

**What we built and saw.** Sharded embeddings (Part/Gather/Stitch), synchronous training with backup workers, and a simulator of slow machines, on top of our mini-TensorFlow from paper 082. Backup workers sped up training, and the best number of backups was close to the paper's.

**Why it matters.** Its ideas, a single programmable graph for computation and state, and synchronous training with straggler mitigation, shaped how large models are trained at scale.

</details>

#### 084 · [GPipe: Pipeline Parallelism](13-Systems-Inference-and-Training/084-Huang-et-al-2019-GPipe/) — Huang et al., 2019
- **The idea:** split a model into stages on different chips, and feed small "micro-batches" through like an assembly line.
- **What we built:** a simulator and a real working pipeline in PyTorch.
- **What we saw:** the pipeline gives exactly the same results as ordinary training, and the speed-up (6.56 with 8 stages) is close to the paper's 6.3.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** Some models are too big to fit on one accelerator. Splitting the layers across devices (**model parallelism**) works, but naively only one device is busy at a time while the others wait. Yanping Huang and colleagues at Google introduced **GPipe**, which keeps devices busy with **pipelining**.

**Topic 1: partitioning.** Split the network's layers into K consecutive **stages**, one per accelerator, balancing the compute per stage.

**Topic 2: the naive problem.** With one batch, stage 2 must wait for stage 1, stage 3 for stage 2, and so on, and the backward pass runs in reverse. At any moment only one of K devices works, so utilisation is 1/K.

**Topic 3: micro-batching (the assembly line).** Split each mini-batch into M **micro-batches**. As soon as stage 1 finishes micro-batch 1, it hands it to stage 2 and starts micro-batch 2, like a factory assembly line. After all forward passes, the backward passes run in a similar pipelined way. Gradients from all micro-batches are **accumulated**, and the weights are updated **once** at the end, **synchronously**. So the result is **mathematically identical** to training on one big device.

**Topic 4: the bubble.** At the start and end of each step, some stages are idle while the pipeline fills and drains. The idle fraction ("bubble") is about:
```
bubble = (K − 1) / (M + K − 1)
```
**Example.** K = 4 stages with M = 8 micro-batches gives 3/11 ≈ 27% idle. With M = 32 it is 3/35 ≈ 9%. The paper notes the bubble is negligible when M ≥ 4K.

**Topic 5: re-materialisation (saving memory).** Storing all activations for the backward pass would take a lot of memory. GPipe stores only the activations at **stage boundaries** and **recomputes** the rest during the backward pass. This trades extra compute for much less memory.

**Topic 6: results.**
- **AmoebaNet** with 557M parameters reached 84.4% top-1 on ImageNet.
- A 6-billion-parameter, **128-layer transformer** for multilingual translation (over 100 languages) beat bilingual baselines.
- Speed-up was almost linear in the number of devices.

**What we built and saw.** A pipeline simulator and a real working pipeline in PyTorch with micro-batches and re-materialisation. The pipeline's weight updates were **identical** to ordinary training, and the speed-up with 8 stages (6.56) was close to the paper's 6.3.

**Why it matters.** Pipeline parallelism (with tensor and data parallelism) is one of the three main ways giant models are trained across thousands of GPUs, for example in Megatron-LM and DeepSpeed.

</details>

#### 085 · [PipeDream: Generalized Pipeline Parallelism](13-Systems-Inference-and-Training/085-Narayanan-et-al-2019-PipeDream/) — Narayanan et al., 2019
- **The idea:** a smarter pipeline that keeps every chip busy all the time, with an automatic planner for splitting the model.
- **What we built:** the planner, the schedule, and a real pipelined network.
- **What we saw:** the planner rediscovered the paper's own layout for a famous network. **Surprise:** a shortcut the paper warns against didn't cause problems on our small model.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** GPipe (paper 084) waits for all micro-batches and then flushes the pipeline at every step, which leaves bubbles. Deepak Narayanan and colleagues proposed **PipeDream** to keep every device busy **all the time**.

**Topic 1: one-forward-one-backward (1F1B) scheduling.** After a short start-up phase, each stage **alternates**: one forward pass for a new micro-batch, then one backward pass for an older one. Micro-batches flow continuously with no flush. In steady state, every device is always working.

**Topic 2: the stale-weights problem.** Because there's no flush, a micro-batch's backward pass may happen **after** the weights were updated by another micro-batch. The forward and backward passes would then use different weights, which gives incorrect gradients.

**Topic 3: weight stashing.** Each stage **keeps a copy** of the weight version used for each in-flight micro-batch's forward pass, and uses that same version for its backward pass. Gradients are then valid (for that version), at the cost of extra memory for the stashed copies. An optional stricter mode ("vertical sync") makes all stages use the same version for a micro-batch.

**Topic 4: automatic partitioning.** How should layers be split into stages, and should some stages be **replicated** (data parallel within a stage)? PipeDream profiles each layer (compute time, activation size, weight size) and uses **dynamic programming** to find the split that minimises the slowest stage's time, accounting for communication bandwidth between machines.

**Topic 5: results.** Up to **5.3× faster** time-to-target-accuracy than data-parallel training on several models (VGG-16, ResNet-50, AlexNet, GNMT, AWD-LSTM), especially when communication is slow.

**What we built and saw.**
- The 1F1B schedule simulator, weight stashing, the dynamic-programming planner, and a real pipelined network.
- The planner rediscovered the paper's own layout for VGG-16 (a replicated early stage and a single later stage).
- **Surprise:** the "no stashing" shortcut that the paper warns against didn't noticeably hurt our small model. Staleness matters more for larger models and larger learning rates.

**Why it matters.** The 1F1B schedule is now the default pipeline schedule in large-scale training frameworks (for example Megatron-LM's interleaved 1F1B).

</details>

#### 086 · [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](13-Systems-Inference-and-Training/086-Rajbhandari-et-al-2020-ZeRO/) — Rajbhandari et al., 2020
- **The idea:** stop every GPU from storing a full copy of everything; split the training state across GPUs instead.
- **What we built:** the memory formulas and a simulated multi-GPU training job.
- **What we saw:** the paper's memory figures were reproduced exactly (120 GB → 1.9 GB per GPU), with identical training results.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** In ordinary **data-parallel** training, every GPU holds a **full copy** of the model, its gradients and the optimiser state. For large models that is enormous duplication. Samyam Rajbhandari and colleagues at Microsoft proposed **ZeRO** (Zero Redundancy Optimizer) to remove it.

**Topic 1: where the memory goes.** With mixed-precision Adam, each parameter needs:
- 2 bytes: 16-bit parameter;
- 2 bytes: 16-bit gradient;
- 12 bytes: 32-bit master copy of the parameter + Adam's momentum + Adam's variance (4 + 4 + 4).

That is **16 bytes per parameter**. For a 7.5B model: 16 × 7.5×10⁹ = **120 GB per GPU**, far more than a 32 GB GPU holds.

**Topic 2: ZeRO's three stages.** With N_d GPUs and Ψ parameters:

| Stage | What is split across GPUs | Memory per GPU | 7.5B model, 64 GPUs |
|---|---|---|---|
| Baseline | nothing | 16Ψ | 120 GB |
| P_os | optimiser states | 4Ψ + 12Ψ/N_d | 31.4 GB |
| P_os+g | + gradients | 2Ψ + 14Ψ/N_d | 16.6 GB |
| P_os+g+p | + parameters | 16Ψ/N_d | **1.9 GB** |

Each GPU is responsible for updating only its own 1/N_d share of the parameters.

**Topic 3: communication.** Data parallelism already uses an **all-reduce** to average gradients, which costs about 2Ψ of data moved per GPU. ZeRO splits this into a **reduce-scatter** (each GPU gets the averaged gradients for its own share) plus an **all-gather** (everyone gets the updated parameters). Stages 1 and 2 therefore cost **the same** communication as ordinary data parallelism. Stage 3 must also gather parameters during the forward and backward passes, costing **1.5×**.

**Topic 4: ZeRO-R (residual memory).**
- **Activation partitioning:** split saved activations across GPUs when using model parallelism.
- **Constant-size buffers** for communication.
- **Memory defragmentation**, to avoid running out of memory due to fragmentation.

**Topic 5: results.** Models with over 100B parameters were trained with super-linear speed-up on 400 GPUs, and ZeRO was used to train **Turing-NLG** (17B), at the time the largest language model.

**What we built and saw.** The memory formulas and a simulated multi-GPU mixed-precision Adam job in all four modes, with counted collective communication.
- The paper's memory figures were reproduced **exactly** (120 / 31.4 / 16.6 / 1.9 GB).
- All modes produced **bit-identical** trained weights.
- Stage 3 used exactly 1.5× the communication.

**Why it matters.** ZeRO (in DeepSpeed) and its PyTorch equivalent **FSDP** (Fully Sharded Data Parallel) are standard tools for training large models.

</details>

---

### Stage 14 · Production machine learning: keeping AI working in the real world

**The story.** Building a model is the easy part. Keeping it working for years is harder: data changes, small code changes ripple unexpectedly, and models quietly get worse. These papers explain the hidden costs of real ML systems ("technical debt"), how to test and monitor them, how to detect when incoming data has shifted, and how to document models honestly.

#### 087 · [Machine Learning: The High-Interest Credit Card of Technical Debt](14-Production-ML/087-Sculley-et-al-2014-High-Interest-Credit-Card-Technical-Debt/) — Sculley et al., 2014
- **The idea:** ML systems pile up hidden maintenance costs: "changing anything changes everything", hidden feedback loops, and unused inputs that can break things later.
- **What we built:** each warning in the paper as a small experiment.
- **What we saw:** removing one input shifted other parts of the model, quietly cleaning up an old data field hurt a model that still relied on it, and a fixed decision threshold lost precision after retraining.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The metaphor.** In software, **technical debt** means shortcuts that save time now but cost more later, like borrowing money: you pay "interest" every time you work with the messy code. D. Sculley and colleagues at Google argued that machine learning systems take on debt **especially fast** and in ways that are hard to see, like a high-interest credit card.

**Topic 1: entanglement, "Changing Anything Changes Everything" (CACE).** A model mixes all its inputs together. If you change one input feature (its scale, its meaning, or remove it), the best weights for **all** the other features change too. No change is local. Improvements in one place can cause regressions somewhere else.

**Topic 2: hidden feedback loops.** A model's predictions can influence the data it will later be trained on. Example: a news recommender shows certain stories, users click what they're shown, and the next model learns those stories are popular. The effect is slow and hard to detect.

**Topic 3: undeclared consumers.** Other teams quietly start using a model's outputs as inputs to their own systems. When you change your model, you break theirs without knowing.

**Topic 4: data dependencies cost more than code dependencies.**
- **Unstable data dependencies:** an input produced by another system that may change behaviour without warning.
- **Underutilised data dependencies:** inputs that add little value but create risk:
  - **legacy features** that became redundant;
  - **bundled features** added together because a group helped, though some members don't;
  - **ε-features** that add tiny accuracy for a lot of complexity.

  The paper recommends **leave-one-feature-out** evaluations to find and remove them.

**Topic 5: correction cascades.** Training model B to fix model A's mistakes for a slightly different problem, then C on B, and so on. Now improving A can break B and C.

**Topic 6: system-level smells.**
- **Glue code:** lots of code just to fit data into a general package.
- **Pipeline jungles:** tangled data-preparation steps.
- **Dead experimental code paths.**
- **Configuration debt:** settings that are as complex as code but not reviewed or tested.

**Topic 7: changes in the outside world.**
- **Fixed thresholds** (for example "flag if score > 0.7") become wrong after retraining, because score distributions shift.
- Correlations the model relied on can change.
- The paper recommends monitoring **prediction bias** (are average predictions still matching average outcomes?) and setting **action limits** (sanity caps on automated actions).

**What we built and saw.** Each warning as a small experiment.
- Removing one feature shifted the other features' weights (CACE).
- Quietly "cleaning up" an old product-number field that the model still relied on left it **worse** than a model trained only on the new numbers (log-loss 0.527 vs 0.497).
- A pipeline change caused a prediction-bias spike of −0.17, which a monitor caught.
- A fixed threshold reused after retraining cut precision from 0.906 to 0.775.

**Why it matters.** It introduced the vocabulary that ML engineers still use to talk about why production ML is hard.

</details>

#### 088 · [Hidden Technical Debt in Machine Learning Systems](14-Production-ML/088-Sculley-et-al-2015-Hidden-Technical-Debt/) — Sculley et al., 2015
- **The idea:** the famous picture: the ML code is a tiny box inside a huge system of data pipelines, configuration and monitoring.
- **What we built:** feedback-loop simulations, a configuration checker, and monitoring tools.
- **What we saw:** a recommender that only learns from what it shows got stuck on a non-best item in 75% of runs, and in our own tiny pipeline the actual ML was only 19% of the code.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The follow-up.** The best-known paper on ML engineering (Sculley et al., NeurIPS 2015) expands paper 087. It is famous for one diagram: a tiny box labelled "**ML Code**" surrounded by huge boxes for:
- configuration;
- data collection;
- data verification;
- feature extraction;
- machine resource management;
- analysis tools;
- process management;
- serving infrastructure;
- monitoring.

**The model is a small part of an ML system.**

**Topic 1: the earlier debts, revisited.** Entanglement (CACE), correction cascades, undeclared consumers, unstable and underutilised data dependencies, and feedback loops, as in paper 087.

**Topic 2: two kinds of feedback loops.**
- **Direct:** a model influences its own future training data. Example: a recommender only learns about items it chooses to show, so it can get stuck showing a non-best item forever. This is related to the **explore/exploit** problem in bandits.
- **Hidden:** two separate systems influence each other through the world, for example two companies' pricing models reacting to each other.

**Topic 3: ML-system anti-patterns.**
- **Glue code:** often 95% of the code just connects to general-purpose packages.
- **Pipeline jungles**, and **dead experimental code paths** (Knight Capital lost $465M in 45 minutes partly due to obsolete code being reactivated).
- **Abstraction debt:** no good abstractions for ML systems, unlike databases.
- **Common smells:** using raw floats and strings everywhere ("plain-old-data"), mixing many languages, and putting prototypes into production.

**Topic 4: configuration debt.** Large systems have many configurable options (features used, data time windows, algorithm settings). Mistakes in configuration are common and costly. Configuration should be easy to review, diff, validate and test.

**Topic 5: dealing with changes in the outside world.** Fixed thresholds, monitoring and testing:
- prediction bias;
- action limits;
- upstream data producers.

Also **reproducibility debt** (randomness, parallelism, changing data), **process management debt** (many models to update and monitor), and **cultural debt** (research and engineering need to work as one team that values removing features and reducing complexity, not only accuracy gains).

**Topic 6: questions to ask.**
- How easily can a new algorithmic approach be tested at full scale?
- What is the transitive closure of all data dependencies?
- How precisely can the impact of a new change be measured?
- Does improving one model degrade others?
- How quickly can new team members get up to speed?

**What we built and saw.** Feedback-loop simulations, a configuration validator, unit-mismatch detection and monitoring.
- A greedy recommender that learned only from what it showed got stuck on a non-best item in **75%** of random seeds, against 5% for an exploring (UCB) strategy.
- A probability vs log-odds mix-up silently flipped 10.1% of decisions.
- In our own small pipeline, the ML part was only **19%** of the code.

**Why it matters.** It is the standard reference for why "MLOps" exists, and one of the most cited papers in ML engineering.

</details>

#### 089 · [The ML Test Score: A Rubric for ML Production Readiness](14-Production-ML/089-Breck-et-al-2017-ML-Test-Score/) — Breck et al., 2017
- **The idea:** a checklist of 28 tests (data, model, infrastructure, monitoring) and a score for how production-ready a system is.
- **What we built:** the scoring rule and 26 working automated tests on a toy system.
- **What we saw:** each of 11 deliberately injected bugs was caught by at least one test.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The goal.** How do you know if an ML system is **ready for production**? Eric Breck and colleagues at Google turned years of experience into a concrete checklist of **28 tests** and a scoring rule.

**Topic 1: four sections, seven tests each.**

**Data tests:**
- feature expectations are captured in a schema (ranges, types);
- every feature is beneficial;
- no feature costs too much (latency, maintenance);
- features follow meta-level requirements (for example no personal data);
- data pipelines have privacy controls;
- new features can be added quickly;
- all input feature code is tested.

**Model tests:**
- model specs are reviewed and checked in;
- offline and online metrics correlate;
- all hyperparameters have been tuned;
- the impact of model **staleness** is known;
- a simpler model is not better;
- model quality is sufficient on important **data slices**;
- the model is tested for inclusion (fairness).

**ML infrastructure tests:**
- training is reproducible;
- model specification code is unit tested;
- the full ML pipeline is integration tested;
- model quality is validated before serving;
- the model can be debugged step by step on a single example;
- models are tested via a **canary** process before full release;
- models can be **rolled back** quickly.

**Monitoring tests:**
- dependency changes result in a notification;
- data invariants hold in training and serving inputs;
- training and serving features compute the same values (no **training/serving skew**);
- models are not too stale;
- models are numerically stable (no NaNs or infinities);
- computational performance has not regressed;
- prediction quality has not regressed.

**Topic 2: scoring.**
- Each test earns **0.5 points** if done **manually** with documented results.
- It earns **1 point** if it is **automated** and run regularly.
- Each section's score is the sum of its 7 tests.
- The final ML Test Score is the **minimum** of the four section scores, because a system is only as strong as its weakest area.

| Score | Meaning |
|---|---|
| 0 | more of a research project than a production system |
| (0, 1] | not totally untested, but serious holes in reliability are possible |
| (1, 2] | a first pass at basic productionisation, but more investment may be needed |
| (2, 3] | reasonably tested, but more could be automated |
| (3, 5] | strong automated testing and monitoring, appropriate for mission-critical systems |
| > 5 | exceptional levels of automated testing and monitoring |

**Topic 3: what they learned at Google.** Teams scoring themselves often found gaps they hadn't noticed, especially in monitoring and data tests. The rubric was used for self-assessment and planning.

**What we built and saw.** The scoring rule and **26 working automated tests** on a toy production system (schema checks, skew checks, canary, rollback, NaN monitors…). The toy system scored 6.5 under our rubric implementation. We deliberately injected **11 bugs** (skewed features, NaNs, stale models, broken rollback…), and **each was caught** by at least one test.

**Why it matters.** It gives teams a practical, measurable target for ML reliability. Many MLOps checklists derive from it.

</details>

#### 090 · [Rules of Machine Learning](14-Production-ML/090-Zinkevich-Rules-of-ML/) — Zinkevich (Google)
- **The idea:** 43 practical rules, such as "launch with a simple heuristic first" and "log the features you actually used at serving time".
- **What we built:** experiments for the rules that make a measurable claim.
- **What we saw:** a simple heuristic got 44% of the way to the ML model (the guide says about 50%), and correctly weighting sampled data fixed a model that had doubled its predictions.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**What it is.** Martin Zinkevich's guide (Google) of **43 rules** for building ML products, written for engineers who know some ML but haven't shipped many ML systems. It is organised by phase.

**Topic 1: before machine learning.**
- **Rule 1:** don't be afraid to launch a product without ML. A simple heuristic can get you far.
- **Rule 2:** first, design and implement metrics.
- **Rule 3:** choose ML over a complex heuristic. Once heuristics get complicated, ML is easier to maintain.

**Topic 2: phase I, your first pipeline.**
- **Rule 4:** keep the first model simple and get the infrastructure right. Most early problems are infrastructure, not the model.
- **Rule 5:** test the infrastructure independently from the ML.
- **Rule 7:** turn existing heuristics into features.
- **Rules 8–11:** know the freshness requirements, detect problems before exporting models, watch for silent failures, and give feature columns owners and documentation.

**Topic 3: your first objective.** Choose a simple, observable, attributable objective (for example clicks). Don't optimise vague goals directly. Use simple, interpretable models (such as logistic regression) at first; they are easier to debug.

**Topic 4: phase II, feature engineering.**
- **Rule 16:** plan to launch and iterate.
- **Rule 17:** start with directly observed features, not learned ones.
- **Rule 22:** clean up features you're no longer using.
- **Rule 29:** the best way to make sure you train like you serve is to **save the features used at serving time** and train on those logs. This prevents **training/serving skew**.
- **Rule 30:** **importance-weight sampled data**; don't arbitrarily drop it. If you keep only 30% of negative examples, give each kept one weight 1/0.3 ≈ 3.33, or the model's predicted probabilities will be biased.
- **Rule 31:** beware that if you join data from a table at training and serving time, the table may have changed in between.
- **Rule 33:** if you train on data up to January 5th, **test on data from January 6th onward**. A random split leaks the future into training and is over-optimistic.
- **Rule 34:** in binary classification for filtering (like spam), make small short-term sacrifices in performance for very **clean data**, for example by holding out 1% of traffic from filtering to get unbiased labels.

**Topic 5: phase III, slowed growth and complex models.** When gains slow down, look at new sources of information, more complex models and ensembles. Revisit the objective; watch for metrics that are not aligned with long-term goals.

**What we built and saw.** Experiments for the rules that make testable claims.
- A simple heuristic got **44%** of the way to the ML model's gain (the guide suggests about half).
- Training on downsampled negatives **without** importance weights roughly doubled the predicted probabilities; weighting fixed it.
- A random train/test split was over-optimistic compared with a time-based split.
- Labels from a 1% unfiltered holdout matched the true ("oracle") quality estimate.

**Why it matters.** It is one of the most practical documents for ML engineers. Many of its rules (log your serving features, split by time, start simple) are now standard.

</details>

#### 091 · [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](14-Production-ML/091-Rabanser-et-al-2019-Failing-Loudly/) — Rabanser et al., 2019
- **The idea:** compares ways to detect when incoming data no longer looks like the training data.
- **What we built:** the full detection pipeline with eight ways of summarising data and four statistical tests.
- **What we saw:** like the paper, using the model's own outputs worked well with few samples. Large shifts were easy to catch; small or class-balance shifts were hard.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** Models assume future data looks like training data. When the data **shifts** (new camera, new users, a change in the world), accuracy can silently drop. Stephan Rabanser, Stephan Günnemann and Zachary Lipton systematically compared ways to **detect** shift, so that models "fail loudly" instead of silently.

**Topic 1: kinds of shift.**
- **Covariate shift:** the inputs change (photos get darker), but the rule from inputs to labels stays the same.
- **Label shift:** the class proportions change (more spam this month).
- **Concept shift:** the rule itself changes. This is the hardest kind.

**Topic 2: detection as a two-sample test.** Take a sample of training data and a sample of new data. Ask: could these two samples come from the same distribution? A statistical test gives a **p-value**; if it is below a threshold (for example 0.05), declare a shift.

**Topic 3: first reduce the dimensions.** Raw inputs (images) have thousands of dimensions, which makes tests weak. The paper compares ways to summarise each input:
- **NoRed:** raw features.
- **PCA:** the main directions of variation.
- **SRP:** sparse random projection.
- **Autoencoders:** untrained (random) or trained.
- **BBSD (Black-Box Shift Detection):** use a trained classifier's outputs, either softmax probabilities (BBSDs) or hard predicted labels (BBSDh). Its theory is strongest for label shift.
- **Domain classifier:** train a model to tell old from new data. If it does better than chance, there is a shift.

**Topic 4: the statistical tests.**
- **Kolmogorov–Smirnov (KS):** for one dimension, the largest gap between the two samples' cumulative distribution curves. For many dimensions, test each one and use a **Bonferroni correction**: with K dimensions, require p < α/K for each one, to avoid false alarms from testing many times.
- **MMD (maximum mean discrepancy):** a multivariate test comparing the samples through a kernel (a similarity function), with a permutation test for the p-value.
- **Chi-squared:** for hard labels.
- **Binomial test:** for the domain classifier's accuracy.

**Topic 5: findings.**
- **BBSDs** (softmax outputs + KS tests) was the best overall, especially with few samples.
- Univariate tests with Bonferroni worked about as well as the multivariate MMD.
- Large shifts are easy to detect; small shifts are hard.
- Some shifts are harmless (they don't hurt accuracy), so after detection you should **characterise** the shift (look at the most "anomalous" samples) and judge its **malignancy** (label a few new samples and check accuracy).

**What we built and saw.** The full pipeline on 8×8 digit images, with all eight reductions, four tests and the paper's shift types.
- The false-positive rate with no shift was 0.024 (below the 0.05 target).
- BBSDs was the best univariate method with only 10 samples.
- The domain classifier was weak with few samples but strong at 200.
- Large shifts were caught with 10 samples; small noise and class knock-out went undetected (and were harmless).
- **Honest difference:** at 200 samples, tests on raw pixels did best on our tiny 64-pixel images, unlike the paper.

**Why it matters.** It is the standard empirical reference for choosing a drift-detection method in production monitoring.

</details>

#### 092 · [Data Distribution Shifts and Monitoring](14-Production-ML/092-Huyen-Data-Distribution-Shifts-and-Monitoring/) — Chip Huyen, 2022 (blog post)
- **The idea:** the different kinds of data shift, how to detect them, and how to respond.
- **What we built:** each kind of shift, monitoring with sensible time windows, and retraining strategies.
- **What we saw:** the most damaging kind ("concept drift", where the right answer changes) was **invisible** to input monitoring; only checking accuracy against real outcomes revealed it.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**What it is.** A widely read blog post by Chip Huyen explaining why deployed ML models fail, the kinds of data distribution shift, and how to monitor for them. It reads well with paper 091 and the book in paper 094.

**Topic 1: why ML systems fail.**
- **Software failures:** dependency failures, deployment failures, hardware failures, downtime. Many real outages are these.
- **ML-specific failures:**
  - **training/serving skew:** the model sees different data in production;
  - **edge cases:** rare inputs with catastrophic errors (for example self-driving);
  - **degenerate feedback loops:** predictions influence future labels (recommenders);
  - **data distribution shift.**

**Topic 2: types of shift, with examples.**
- **Covariate shift:** P(X) changes and P(Y|X) stays. Example: a breast-cancer model trained mostly on women over 40, then used on younger women.
- **Label shift:** P(Y) changes and P(X|Y) stays. Example: a preventive drug makes the disease rarer.
- **Concept drift:** P(Y|X) changes. Example: the same apartment's price changes after an economic shock. This is the most damaging kind, and **invisible to input monitoring** because the inputs look the same.
- **Feature change** (a feature's unit or meaning changes) and **label schema change** (new classes are added).

**Topic 3: detecting shifts.**
- **Monitor accuracy-related metrics** if you get labels (natural labels like clicks, or delayed labels). This is the most direct method.
- **Monitor inputs and predictions** when labels are missing: summary statistics (min, max, mean, percentage missing) and **two-sample tests** (KS, MMD; see paper 091).
- **Time windows matter:** shifts happen at different speeds. **Cumulative** statistics can hide recent changes; **sliding** windows show them. Account for **seasonality** (compare Monday with last Monday, not with Sunday), or you will get false alarms.

**Topic 4: addressing shifts.**
- Train on more varied data.
- **Retrain** periodically, either from scratch (**stateless**) or by continuing from the current model with new data (**stateful** fine-tuning, which is cheaper).
- Choose features that are more stable over time (for example rank buckets rather than raw counts).
- Use separate models for very different groups.

**Topic 5: monitoring vs observability.**
- **Monitoring:** tracking metrics and alerting when something is wrong.
- **Observability:** designing the system so you can figure out **why** it went wrong (logs, traces, the ability to slice metrics).

Too many alerts cause **alert fatigue**, so alerts should be actionable.

**What we built and saw.** Simulations of each shift type, monitoring with time windows and seasonality, and retraining strategies.
- **Concept drift was invisible** to input monitoring; only checking accuracy against real outcomes revealed it.
- Label shift **raised** accuracy (more of the easier class) while **breaking calibration**.
- Seasonal comparison removed false alarms.
- Fine-tuning on recent data beat full retraining in our setting.

**Why it matters.** It is a clear, practical map of what to watch after deployment, which is where many real ML problems appear.

</details>

#### 093 · [Model Cards for Model Reporting](14-Production-ML/093-Mitchell-et-al-2019-Model-Cards/) — Mitchell et al., 2019
- **The idea:** ship every model with a short report card, including how well it works **for different groups of people**, not just on average.
- **What we built:** a model card generator and the paper's two worked examples (smile detection and toxicity scoring).
- **What we saw:** the averages looked fine while specific groups had much higher error rates, which only the group-by-group breakdown revealed.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** Models are shared and deployed with little information about **how well they work for whom**, or what they shouldn't be used for. A model that is 95% accurate on average may be much worse for a particular group. Margaret Mitchell and colleagues proposed **Model Cards**: short documents that accompany every trained model.

**Topic 1: the sections of a model card.**
1. **Model details:** who built it, when, version, type, licence, contact.
2. **Intended use:** primary uses and users; **out-of-scope** uses.
3. **Factors:** groups (age, gender, skin type…), instrumentation (camera type) and environment (lighting) that may affect performance.
4. **Metrics:** which performance measures, and why; decision thresholds; how uncertainty is measured.
5. **Evaluation data:** datasets, motivation, preprocessing.
6. **Training data:** as much as possible.
7. **Quantitative analyses:** results broken down by factor (**unitary**) and by combinations of factors (**intersectional**, for example "older men").
8. **Ethical considerations:** sensitive data, risks, mitigations.
9. **Caveats and recommendations.**

**Topic 2: error metrics explained.** Out of the model's predictions:
- **False positive rate (FPR):** of the truly negative cases, how many were wrongly flagged.
- **False negative rate (FNR):** of the truly positive cases, how many were missed.
- **False discovery rate (FDR):** of the cases the model **flagged**, how many were wrong. Example: if the model says 100 people are smiling and 28 aren't, FDR = 0.28.
- **False omission rate (FOR):** of the cases the model did **not** flag, how many were actually positive.

Which metric matters depends on the use: for a medical test you may care most about FNR; for an accusation, FDR.

**Topic 3: disaggregated and intersectional evaluation.** Report metrics **per group**, with **confidence intervals** (small groups have uncertain estimates). Intersectional groups can reveal problems that single-factor breakdowns hide: a model can look fine for "men" and for "older people" but be bad for "older men".

**Topic 4: the paper's two examples.**
- **Smiling detection** (trained on the CelebA face dataset): FPR, FNR, FDR and FOR by age and gender groups and their intersections.
- **Toxicity scoring** (Perspective API): an early version gave high toxicity scores to harmless sentences that merely mention identities ("I am a gay man"), because such words often appeared in toxic training examples. A later version reduced this. The card makes it visible.

**What we built and saw.** A model card generator (with intersectional breakdowns and bootstrap confidence intervals) and toy versions of both examples.
- The smile detector's overall numbers looked fine, but **older men** had the highest FDR (0.276).
- In the toxicity toy, harmless sentences mentioning identity terms scored 0.45 "toxic" (vs 0.05 for others), and rebalancing the training data largely fixed this.

**Why it matters.** Model cards are now standard on Hugging Face, in company AI documentation, and in regulations. System cards (paper 058) extend the idea to whole AI systems.

</details>

#### 094 · [Designing Machine Learning Systems](14-Production-ML/094-Huyen-2022-Designing-ML-Systems/) — Chip Huyen, 2022 (book)
- **The idea:** the whole life of an ML system: data, labels, features, evaluation, deployment and testing in production. (The book is paid, so this is built from its public table of contents, without quoting it.)
- **What we built:** a small experiment for each major technique.
- **What we saw:** "data leakage" made a model look 84% accurate on pure noise, an "invariance test" (does changing only a protected attribute change the decision?) caught a biased loan model that flipped 15.9% of decisions, and updating a model daily was 67 times cheaper than retraining from scratch, for the same quality.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**What it is.** Chip Huyen's book on the **whole life** of an ML system: from framing the business problem through data, features, models, deployment, monitoring and continual learning. The book is paid, so our explanation is built from its public table of contents and general knowledge, without quoting it.

**Topic 1: sampling.**
- **Reservoir sampling:** keep a fair random sample of k items from a stream of unknown length. Keep the first k; for item number n, replace a random slot with probability k/n. Every item ends up equally likely to be in the sample.
- **Stratified sampling:** sample within each group so that rare groups aren't missed.

**Topic 2: labels.**
- Hand labels are expensive.
- **Weak supervision** writes **labelling functions** (rules like "if the note mentions 'pneumonia', label positive") and combines their noisy votes (the Snorkel approach).
- **Active learning:** label the examples the model is most unsure about.

**Topic 3: class imbalance.** With 1% positives, a model that always says "negative" is 99% accurate and useless. Use better metrics:
- **ROC-AUC** can look good even when the positive class is handled badly;
- **PR-AUC** (precision vs recall) is more honest for rare positives.

Fixes include class weights and resampling.

**Topic 4: data leakage.** Information from the test set or the future sneaks into training. Examples:
- **feature selection on all data before splitting**;
- duplicates across train and test;
- random splits on time-based data;
- features that are only known after the label.

**Topic 5: feature engineering.**
- Scaling.
- Missing values.
- The **hashing trick**: map unbounded categories (user IDs) into a fixed number of buckets with a hash function. It accepts some collisions in exchange for fixed memory and handling new categories.

**Topic 6: evaluation beyond accuracy.**
- **Baselines** (random, simple heuristic, human).
- **Calibration:** do predicted probabilities match reality? Measured with ECE (expected calibration error) and fixed with **Platt scaling** (fit a small logistic regression on the model's scores).
- **Slice-based evaluation.**
- **Behavioural tests:**
  - **invariance tests:** changing a protected attribute shouldn't change the decision;
  - **directional tests:** more income shouldn't lower a loan score.

**Topic 7: deployment.**
- **Batch prediction** (precompute) vs **online prediction** (compute on request, with fresher features).
- Model compression.
- Testing in production:
  - **shadow** deployment;
  - **A/B tests** (compute the sample size needed to detect a given effect);
  - **canary** releases;
  - **interleaving** (mix two rankers' results in one list);
  - **bandits** (Thompson sampling sends more traffic to the better model while still exploring).

**Topic 8: continual learning.** **Stateless retraining** (from scratch every time) vs **stateful training** (continue from the last model with only new data), which is far cheaper.

**What we built and saw.** A small experiment for each technique.
- Weak labels from labelling functions beat 100 hand labels (0.857 vs 0.747).
- With 1.4% positives, PR-AUC was 0.155 while ROC-AUC looked great at 0.863.
- **Leakage:** feature selection before splitting scored 0.84 on **pure-noise labels** (honest result 0.43).
- Platt scaling cut ECE from 0.204 to 0.010 without changing ranking quality.
- An invariance test caught a biased loan model that flipped 15.9% of decisions when only a protected attribute changed.
- Online prediction beat nightly batch (0.789 vs 0.729).
- Thompson sampling halved the "regret" (81 vs 167).
- Stateful training was **67× cheaper** for the same quality.

**Why it matters.** It brings the whole production ML lifecycle together in one place and complements the more specific papers 087–093.

</details>

---

### Stage 15 · Reinforcement learning *(optional)*

**The story.** Reinforcement learning teaches by reward rather than by example: try something, see how well it went, do more of what worked. These papers show it at large scale: a simple "evolution" method, the team that beat world champions at Dota 2, and AlphaGo, the first program to beat a professional Go player.

#### 095 · [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](15-Reinforcement-Learning-optional/095-Salimans-et-al-2017-Evolution-Strategies/) — Salimans et al., 2017
- **The idea:** nudge all the weights randomly, keep the nudges that score better, repeat. It's simple and easy to spread over thousands of computers.
- **What we built:** the method, its many-computer version, and a balancing-pole game to test it on.
- **What we saw:** it solved the game, and the noise in its learning signal stayed steady as tasks got longer, while the usual RL method's noise grew a lot. The paper predicts exactly that.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The idea.** Reinforcement learning usually computes gradients through the policy (backpropagation through actions and their probabilities). Tim Salimans and colleagues at OpenAI revisited a much simpler, older idea, **Evolution Strategies (ES)**, and showed it scales extremely well.

**Topic 1: the basic algorithm.** Let θ be the policy's weights.
1. Draw n random noise vectors ε_1 … ε_n (each from a standard bell curve).
2. Try each perturbed policy θ + σ·ε_i in the environment and record its total reward F_i.
3. Move θ toward the noise directions that did well:
```
θ ← θ + α · (1 / (n·σ)) · Σ_i  F_i · ε_i
```
No backpropagation is needed; only the total reward of each episode counts.

**Worked 1-D example.** θ = 0, σ = 1, with reward F(θ) = −(θ − 3)².
- Try ε = +1: F(1) = −4.
- Try ε = −1: F(−1) = −16.
- The estimate is (1/2)·(−4·1 + (−16)·(−1)) = (1/2)·(12) = 6, which is positive.

So θ moves toward 3, the correct direction.

**Topic 2: why it parallelises so well.** Each worker needs to know the other workers' noise vectors to compute the update, and sending millions of numbers would be costly. Trick: all workers share the same **random seeds**, so each can **regenerate** everyone's noise locally. Workers then only exchange **one number each** (their reward). This scaled to **over a thousand CPU cores** with nearly linear speed-up.

**Topic 3: practical improvements.**
- **Mirrored (antithetic) sampling:** evaluate both +ε and −ε, which reduces noise.
- **Fitness shaping:** use the **ranks** of rewards instead of raw values, so a single lucky outlier doesn't dominate.
- **Weight decay.**
- **Virtual batch normalisation** for some tasks.

**Topic 4: properties.**
- **Invariant to action frequency and to delayed rewards:** it only looks at the total reward.
- **Long horizons:** policy-gradient methods (like REINFORCE) get noisier as episodes get longer, because they credit each of many actions. ES's noise does not grow with episode length.
- **No value function** is needed.
- It is less sample-efficient (it uses more episodes) but very fast in **wall-clock time** given many machines.

**Topic 5: results.**
- 3D humanoid walking (MuJoCo) learned in **10 minutes** using 1,440 CPU cores.
- Competitive Atari results with about one hour of training.

**What we built and saw.** ES with mirrored sampling and rank shaping, the shared-seed parallel version, and the CartPole balancing task. ES solved CartPole. As episodes got longer, REINFORCE's gradient variance grew from **5.6 to 367**, while ES stayed around **10**, as the paper explains.

**Why it matters.** ES is a strong, simple baseline for RL and black-box optimisation (tuning anything you can score but not differentiate), and the shared-seed trick is a neat lesson in communication-efficient design.

</details>

#### 096 · [Dota 2 with Large Scale Deep Reinforcement Learning (OpenAI Five)](15-Reinforcement-Learning-optional/096-Berner-et-al-2019-Dota2-OpenAI-Five/) — Berner et al., 2019
- **The idea:** a team of AIs beat the world champions at the video game Dota 2 after 10 months of training, using "surgery" to change the model without starting over.
- **What we built:** model surgery, the training method (PPO) on a small game, and the paper's data-quality experiments.
- **What we saw:** surgery kept learned skill when the model was changed, and using out-of-date data slowed learning the most, as the paper found.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The achievement.** OpenAI Five, a team of five neural networks, beat the **world champion** team (OG) at the complex video game **Dota 2** in April 2019. Christopher Berner and colleagues describe how.

**Topic 1: why Dota 2 is hard.**
- **Long games:** about 45 minutes, or roughly 20,000 decisions per player.
- **Partial observability:** you can't see the whole map.
- **High-dimensional observations:** about 16,000 numbers per step.
- **A huge action space:** which ability, on which target, where.
- **Teamwork** among 5 heroes.

**Topic 2: the model and training.**
- Each hero is controlled by a copy of the same network, centred on a **single-layer LSTM with 4,096 units** (about **159 million** parameters in total).
- Trained with **PPO** (Proximal Policy Optimization), a policy-gradient method that **clips** how much the policy can change per update (the probability ratio is limited to [1 − ε, 1 + ε], for example ε = 0.2) for stability.
- Advantage estimates use GAE (generalised advantage estimation).
- **Self-play:** 80% of games against the latest version, 20% against past versions, to avoid strategy collapse.
- **Reward shaping:** rewards for kills, gold, and so on, besides winning.
- **Team spirit τ:** each hero's reward blends its own reward and the team average. τ rose from 0 to 1 during training, from selfish to cooperative.

**Topic 3: scale.** About **10 months** of training on thousands of GPUs and over 100,000 CPU cores, playing the equivalent of thousands of years of games.

**Topic 4: surgery, changing the model without starting over.** During 10 months, the game updated and the team wanted to add new inputs and bigger layers. Retraining from scratch each time would waste months. **Surgery** transforms the trained network into a new architecture that computes the **same function**:
- new input weights start at zero;
- new units are added so that the outputs don't change;
- and so on.

Training then continues. The paper reports that a from-scratch "Rerun" eventually did better but needed about two months of compute to match.

**Topic 5: data quality, staleness and sample reuse.**
- **Staleness:** experience generated by an older version of the policy is less useful. Slower data delivery lengthened training.
- **Sample reuse:** training several times on the same data (to use expensive data fully) also slowed learning.

Fresh, used-once data was best.

**Topic 6: results.** OpenAI Five beat OG 2–0, and in a public online event won **99.4%** of over 7,000 games against human teams.

**What we built and saw.** Network surgery (adding inputs and units while preserving outputs exactly), PPO on a small team game with team spirit, and the staleness and reuse experiments.
- Surgery kept learned skill after architecture changes.
- Stale data slowed learning 2.3× and sample reuse 1.5×, the same ordering as the paper.
- Full team spirit (τ = 1) halved the signal-to-noise ratio of the learning signal, explaining why τ was raised gradually.

**Why it matters.** It showed that standard RL plus enormous scale can master complex team games, and that "surgery" and data freshness matter for very long training runs.

</details>

#### 097 · [Mastering the Game of Go with Deep Neural Networks and Tree Search (AlphaGo)](15-Reinforcement-Learning-optional/097-Silver-et-al-2016-AlphaGo/) — Silver et al., 2016
- **The idea:** **AlphaGo** combined networks that suggest moves and judge positions with a look-ahead search.
- **What we built:** the whole AlphaGo pipeline on a small board game that can be solved perfectly, so every part can be checked against perfect play.
- **What we saw:** adding search raised the share of perfect moves from 86% to about 99%. **Honest differences:** on this small game the simple fast policy beat the network, and self-play training helped less than in the paper.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The achievement.** Go was considered the grand challenge of game AI. A 19×19 board has about 250 legal moves per position and games last about 150 moves, so the number of possible games (about 250¹⁵⁰) is astronomically larger than the number of atoms in the universe. David Silver and colleagues at DeepMind built **AlphaGo**, which beat the European champion Fan Hui 5–0, the first time a program beat a professional on a full board without a handicap.

**Topic 1: four networks / policies.**
1. **SL policy network:** a 13-layer convolutional network trained by **supervised learning** on 30 million positions from expert games to predict the expert's move. Accuracy 57%.
2. **Fast rollout policy:** a simple linear model on small local patterns. Much weaker (24% accuracy) but about **1,000× faster**, used for quick simulated games.
3. **RL policy network:** starts from the SL network and improves by **playing against earlier versions of itself**, using policy gradient (win = +1, loss = −1). It won 80% of games against the SL network.
4. **Value network:** predicts the **winner** from a position. Trained on 30 million positions from self-play games, taking only **one position per game**, because positions from the same game are nearly identical and caused overfitting.

**Topic 2: Monte Carlo Tree Search (MCTS).** AlphaGo looks ahead by building a search tree. Each simulation:
1. **Select:** walk down the tree choosing moves with high Q + u, where Q is the average value found so far and u ∝ P / (1 + N) is a bonus favouring moves the policy network likes (prior P) that haven't been visited much (count N). This balances **exploration and exploitation**.
2. **Expand:** at a new position, use the SL policy network for priors P.
3. **Evaluate** the leaf in two ways and mix them:
```
V(leaf) = (1 − λ) · value_network(leaf) + λ · (result of a fast rollout to the end of the game)
```
   λ = 0.5 worked best in the paper.
4. **Backup:** update Q and N along the path.

After many simulations, play the **most visited** move.

**Topic 3: why the SL policy for priors?** Surprisingly, the weaker SL policy gave better search priors than the stronger RL policy. Humans consider a **diverse** set of good moves, which helps search, while the RL policy focuses too narrowly on its single favourite move.

**Topic 4: compute.** The distributed version used 1,202 CPUs and 176 GPUs. A few months later, AlphaGo beat Lee Sedol 4–1 (March 2016).

**What we built and saw.** The whole pipeline on a small board game that can be **solved perfectly**, so every component can be checked against perfect play: SL, rollout, RL and value networks, and MCTS with mixed evaluation.
- Adding search raised the share of perfect moves from **86% to about 99%**.
- **Honest differences:** on this small game, the fast rollout policy was a better evaluator than our value network (so pure rollouts beat λ = 0.5), and self-play RL helped less than in the paper.

**Why it matters.** AlphaGo led to AlphaGo Zero and AlphaZero (learning from self-play alone, without human games) and MuZero. Its "learned policy + learned value + search" recipe is also an inspiration for reasoning methods in language models.

</details>

---

### Stage 16 · Extras *(optional)*

**The story.** A mix of interesting side-roads: learning with labels in a contrastive way, a lesson from early feature learning, two early papers on representing relationships, and two papers about AI's impact on society, covering security risks and jobs.

#### 098 · [Supervised Contrastive Learning](16-Extras-optional/098-Khosla-et-al-2020-Supervised-Contrastive-Learning/) — Khosla et al., 2020
- **The idea:** pull together the representations of all images of the same class and push apart those of different classes.
- **What we built:** the paper's losses and its two-stage training, compared with ordinary training.
- **What we saw:** it was much more robust to noise and blur, but less robust to small shifts. With careful tuning, ordinary training tied it.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The idea.** **Contrastive learning** (for example SimCLR) learns representations without labels: two augmented views of the same image (different crops, colours) should have similar embeddings, and views of different images dissimilar ones. Prannay Khosla and colleagues asked: if we **do** have labels, can we use them in the same contrastive way? The result is **Supervised Contrastive Learning** (SupCon).

**Topic 1: self-supervised contrastive loss.** For each image in a batch, its one **positive** is the other augmented view of the same image; all other images are **negatives**. The loss is a softmax classification: pick the positive out of the batch using dot-product similarity with a temperature τ.

**Topic 2: the supervised version.** With labels, **all images of the same class** in the batch are positives. The SupCon loss for an anchor i, with P(i) the set of its positives:
```
L_i = − (1/|P(i)|) Σ_{p ∈ P(i)}  log [ exp(z_i·z_p / τ)  /  Σ_{a ≠ i} exp(z_i·z_a / τ) ]
```
This pulls together **all** same-class embeddings and pushes apart different classes. The paper compared two placements of the 1/|P(i)| average (outside vs inside the log) and found **outside** works better: it gives a more stable, better-balanced gradient.

**Topic 3: two-stage training.**
1. Train the encoder (for example a ResNet) with the SupCon loss, using a small **projection head** whose output is normalised to unit length.
2. Throw away the projection head, **freeze** the encoder, and train a simple linear classifier on top with ordinary cross-entropy.

**Topic 4: why it might be better than cross-entropy.**
- Cross-entropy only cares about separating classes at the output layer.
- SupCon explicitly shapes the **embedding space** into tight, well-separated class clusters.
- The paper shows its gradients focus on **hard** positives and negatives automatically.

**Topic 5: results.**
- ImageNet top-1 accuracy of **81.4%** with ResNet-200, better than cross-entropy.
- More robust on corrupted images (ImageNet-C: noise, blur, weather).
- Less sensitive to hyperparameters like the optimiser and augmentations.
- Large batches and a good temperature matter.

**What we built and saw.** The SupCon and SimCLR losses (both placements of the average), two-stage training and a robustness benchmark on small images.
- SupCon reached 0.982 vs 0.962 for cross-entropy and 0.944 for SimCLR.
- **But** with careful tuning, cross-entropy **tied** SupCon.
- SupCon was much more robust to noise and blur, but **less** robust to small shifts of the image. Gains depend on the type of corruption.

**Why it matters.** Supervised contrastive ideas are used in representation learning, retrieval, and learning with noisy or few labels, and the loss is a common building block.

</details>

#### 099 · [The Importance of Encoding Versus Training with Sparse Coding and Vector Quantization](16-Extras-optional/099-Coates-Ng-2011-Encoding-vs-Training-Sparse-Coding/) — Coates & Ng, 2011
- **The idea:** in early feature learning, **how you use** the learned patterns matters more than how carefully you learn them.
- **What we built:** every combination of five ways to learn the patterns and four ways to use them.
- **What we saw:** exactly the paper's finding. With a good way of using them, even randomly chosen patterns worked as well as carefully learned ones.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The question.** In early unsupervised feature learning (before deep learning took over), a common pipeline was:
1. learn a **dictionary** of patterns from image patches (**training**);
2. turn each patch into features using that dictionary (**encoding**);
3. pool the features and train a linear classifier.

Researchers spent effort on fancy dictionary-learning methods. Adam Coates and Andrew Ng asked which step **really** matters.

**Topic 1: the dictionary-learning (training) methods compared.**
- **Sparse coding (SC):** learn patterns such that each patch is a combination of only a few patterns.
- **OMP-1 / vector quantisation:** each patch is represented by its single best-matching pattern (similar to k-means).
- **Sparse RBMs and autoencoders.**
- **Random patches:** just use randomly chosen image patches as the dictionary, with no learning.
- **Random noise:** random Gaussian vectors.

**Topic 2: the encoders compared.**
- **Sparse coding encoder:** solve an optimisation to find a sparse combination (L1 penalty).
- **OMP-k:** greedily pick the k best-matching patterns.
- **Soft threshold:** f(x) = max(0, Dᵀx − α). Compute the similarity to every pattern and keep only the parts above a threshold α. It is a single cheap matrix multiply plus a ReLU-like step.
- **"Natural" encoding:** the encoder that matches the training method.

**Topic 3: the key finding.**
- **The encoder matters much more than the dictionary.** With a good encoder (sparse coding or soft threshold), even **random patches** as the dictionary performed nearly as well as carefully learned dictionaries.
- The simple soft-threshold encoder was competitive with expensive sparse coding when there was plenty of labelled data.
- With **few labelled examples**, the sparse-coding encoder had an advantage.

**Topic 4: the lesson.** Training and encoding can be **decoupled**: a cheap, fast dictionary method (even random) combined with a well-chosen encoder gives strong results. Effort spent on complicated dictionary learning was often misplaced.

**What we built and saw.** Every combination of the dictionary methods and encoders, on small images, with pooling and a linear classifier.
- With the sparse-coding or soft-threshold encoder, the dictionary choice barely mattered (0.954–0.963).
- With the OMP-1 encoder, results were much lower (0.876–0.911).

That is exactly the paper's finding.

**Why it matters.** It is a classic example of a careful ablation that overturns assumptions: test each component separately. The soft-threshold encoder resembles the ReLU layers at the heart of modern networks.

</details>

#### 100 · [Using Matrices to Model Symbolic Relationships](16-Extras-optional/100-Sutskever-Hinton-2008-Matrices-Symbolic-Relationships/) — Sutskever & Hinton, 2008
- **The idea:** represent both things and relationships as small matrices, so relationships can be combined and even learned from definitions, like learning "+3" from "3 plus".
- **What we built:** the model on clock arithmetic and on family trees.
- **What we saw:** results close to the paper's tables, including learning a relationship it was never shown directly.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The idea.** How can a neural network represent **relationships** like "father of", "+3" or "opposite of", and combine them? Ilya Sutskever and Geoffrey Hinton proposed **Linear Relational Embedding with matrices**: represent both **things** and **relations** as **matrices**, and applying a relation becomes **matrix multiplication**.

**Topic 1: facts as matrix equations.** A fact (A, R, B), "A related by R gives B", is modelled as:
```
A · R ≈ B          (all three are small square matrices)
```
To answer "what is A·R?", compute the product and find the **closest** object matrix. Training makes the correct answer the closest, using a softmax over distances to all objects.

**Topic 2: why matrices?** Matrices can be **composed**: applying R1 then R2 is the matrix R1·R2, which is itself a relation. Relations become objects you can calculate with.

**Topic 3: modular arithmetic, an intuition.** (The paper uses numbers 0–11, "mod 12"; here we use a 10-hour clock to keep the numbers easy.) Consider numbers 0–9 arranged on a clock. If each number is a **rotation** matrix (rotate by n × 36°), then "+3" is rotation by 108°, and "+3" applied to 8 gives a rotation of 8×36° + 108° = 396° = 36°, which is 1. That is correct: (8 + 3) mod 10 = 1. The model can learn representations like this from examples alone.

**Topic 4: family trees.** The classic dataset from paper 004: 24 people in two families (English and Italian), with 12 relations (father, mother, husband, wife, son, daughter, uncle, aunt, brother, sister, nephew, niece). The model learns from most of the facts and is tested on held-out facts.

**Topic 5: higher-order relations.** Relations can apply to relations. Example: "plus" applied to the number 3 gives the relation "+3". If "plus" is a matrix too, then 3 · plus ≈ (+3). The model can then learn a relation **only from higher-order facts**, without ever seeing it applied to numbers directly. It can even add **new** relations later (incremental learning) without retraining everything.

**Topic 6: limitations.** Results vary between training runs: sometimes a run gets everything right and sometimes it fails on certain relations (all-or-nothing behaviour). Larger matrices and careful training help.

**What we built and saw.** The matrix model on modular arithmetic and family trees, with higher-order and incremental relations.
- With 4×4 matrices, test errors were close to the paper's tables (for example 0–1 errors with 30 arithmetic facts held out, where the paper reports 0.0).
- Relations were learned **from higher-order facts alone**.
- Failures were indeed all-or-nothing.
- We also found and fixed a sign error in one gradient during development, verified with a numerical gradient check.

**Why it matters.** It is an early, elegant example of **compositional** representation learning. Representing relations as transformations reappears in knowledge-graph embeddings (TransE, RESCAL, RotatE) and in how transformers apply learned transformations to embeddings.

</details>

#### 101 · [Modelling Relational Data using Bayesian Clustered Tensor Factorization](16-Extras-optional/101-Sutskever-Salakhutdinov-Tenenbaum-2009-Bayesian-Clustered-Tensor-Factorization/) — Sutskever, Salakhutdinov & Tenenbaum, 2009
- **The idea:** predict missing facts in a database while also grouping similar things, using careful probability reasoning.
- **What we built:** the model and its comparisons, on made-up data with hidden groups.
- **What we saw:** with very little data the simple method failed badly while the careful probabilistic one kept working, and it found the hidden groups when there was enough data.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The problem.** Relational databases contain facts like "drug X treats disease Y" or "country A exports to country B", stored as a 3-D table (a **tensor**): T[object i, relation k, object j] = true or false. Many entries are missing; we want to **predict** them and also **discover groups** of similar objects. Ilya Sutskever, Ruslan Salakhutdinov and Joshua Tenenbaum proposed **Bayesian Clustered Tensor Factorization** (BCTF).

**Topic 1: tensor factorisation.** Give every object a vector a_i and every relation a matrix R_k. The score for a fact is:
```
score(i, k, j) = a_iᵀ · R_k · a_j
```
A high score means the fact is likely true. Learning the vectors and matrices from known facts lets us predict unknown ones.

**Topic 2: MAP training and overfitting.** The simplest training finds **one best set** of parameters (MAP: maximum a posteriori, which is training with a penalty on large values). With **sparse data** (few known facts), it overfits: it becomes confident about patterns that are just noise.

**Topic 3: Bayesian averaging.** Instead of one best set, consider **many** plausible parameter sets, weighted by how well they explain the data (the **posterior**), and average their predictions. This is computed by **MCMC sampling** (repeatedly making random changes that are accepted or rejected so that, over time, samples follow the posterior). Averaging is much more robust when data is scarce.

**Topic 4: clustering.** Similar objects should have similar vectors. BCTF puts objects into **clusters**, and each cluster has its own prior centre: an object's vector is pulled toward its cluster's average. Objects in the same cluster **share statistical strength**, so an object with few facts can borrow information from its cluster-mates.

**Topic 5: the Chinese restaurant process (CRP).** How many clusters should there be? The CRP lets the data decide. Imagine customers entering a restaurant: each new customer sits at an existing table with probability proportional to how many people are already there, or starts a **new table** with probability proportional to a parameter α. Popular clusters grow, but new ones can always form.

**Topic 6: results.** On real datasets (animals and their features, the UMLS medical ontology, kinship systems, international relations between nations), BCTF predicted missing facts better than previous methods and found **interpretable clusters** (for example groups of related medical concepts).

**What we built and saw.** MAP factorisation, a Bayesian version without clusters (BTF), and full BCTF, on synthetic relational data with hidden groups.
- With **plenty** of data, all three performed about equally, and BCTF recovered the true clusters.
- With **sparse** data, MAP collapsed (PR-AUC 0.38) while the Bayesian methods kept working (0.88), exactly the paper's motivation.

**Why it matters.** It is a clear demonstration of why **Bayesian averaging** and **shared structure** help when data is scarce, and an ancestor of modern knowledge-graph embedding models.

</details>

#### 102 · [The Malicious Use of Artificial Intelligence](16-Extras-optional/102-Brundage-et-al-2018-Malicious-Use-of-AI/) — Brundage et al., 2018
- **The idea:** a policy report on how AI could be misused, and what researchers and governments should do about it.
- **What we built:** a simple model of how cheaper automation expands attacks, and a "red team" check of a classifier. This is defensive only, with no attack tools.
- **What we saw:** a targeted data-poisoning attack fooled the model and slipped past a common cleaning defence, but a simple check on 100 trusted examples caught it.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**What it is.** A 2018 report by 26 authors from Oxford, Cambridge, OpenAI, the Electronic Frontier Foundation and others. It forecasts how AI could be **misused** and what to do about it. It is a policy report, not an algorithm, so our code makes two of its arguments measurable, **from the defender's side only**.

**Topic 1: why AI changes security.** Properties of AI that matter:
- **dual-use:** the same tool helps or harms, for example vulnerability scanners;
- **efficient and scalable:** once built, copies are cheap;
- **can exceed human ability** at some tasks;
- **anonymity and psychological distance:** an attacker can act remotely;
- **rapid diffusion:** code and papers spread quickly;
- **novel vulnerabilities:** AI systems themselves can be attacked through adversarial examples and data poisoning.

**Topic 2: three changes to the threat landscape.**
1. **Expansion of existing threats:** attacks get cheaper, so more people can do them, more often, against more targets.
2. **New threats:** attacks impractical for humans, and attacks on defenders' AI systems.
3. **Changed character:** attacks become more effective, finely targeted and harder to attribute.

**Topic 3: three domains.**
- **Digital:** automated hacking, spear-phishing at scale, voice impersonation.
- **Physical:** drones, attacks on autonomous systems.
- **Political:** surveillance, targeted propaganda, fake videos.

**Topic 4: recommendations.**
- Policymakers should work closely with technical researchers.
- Researchers should take dual-use seriously.
- Borrow best practices from computer security: red teaming, responsible disclosure.
- Involve more stakeholders.

Research areas include openness models (staged release), a culture of responsibility, and technical and policy tools.

**Topic 5: our cost model of "expansion".** Imagine 10,000 potential targets of varying value. An attacker can make a cheap **generic** attempt (cost 0.01, success 0.2%) or a **tailored** attempt (success 5%, cost c). The tailored attempt is worth it when 0.048 × value > c − 0.01.
- **Example:** with c = 20, only targets worth more than about 417 get tailored attacks; with c = 0.2, anything worth more than about 4 does.

AI lowers c, so many more targets receive effective attacks.

**Topic 6: AI-specific vulnerabilities.**
- **Adversarial examples (FGSM):** move every pixel by a small ε in the direction that most increases the model's error: x_adv = x + ε · sign(gradient). See also paper 017.
- **Data poisoning:** change some training labels. Random label noise mostly adds noise; **targeted** relabelling (all 7s labelled as 1) teaches a specific wrong rule.
- **Loss-based sanitisation** (remove the training points the model finds most surprising) catches random noise but **not** consistent targeted poisoning, because the model learns the poisoned rule and finds those points unsurprising.

**What we built and saw.**
- In the cost model, as tailoring became 400× cheaper, tailored attempts grew from 0.3% to 91% of targets, harm rose about 9×, and the share of actors who could afford them went from 2% to 97%.
- FGSM with ε = 0.2 cut a digit classifier's accuracy to 0.24; adversarial training helped only a little.
- Targeted poisoning made every test 7 into a "1" while overall accuracy still looked fine (0.86), and sanitisation **didn't** catch it.
- A simple **audit** on 100 trusted, correctly labelled images (per-class accuracy showing class 7 at 0%) caught it immediately.

Process beats clever filtering.

**Why it matters.** Many of its recommendations, including pre-deployment red teaming, staged release, responsible disclosure and misuse sections in model and system cards (papers 093 and 058), are now common practice.

</details>

#### 103 · [Automation and New Tasks: How Technology Displaces and Reinstates Labor](16-Extras-optional/103-Acemoglu-Restrepo-2019-Automation-and-New-Tasks/) — Acemoglu & Restrepo, 2019
- **The idea:** economics of AI and jobs. **Automation** takes tasks from workers and lowers labour's share of income. **New tasks** create work and raise it. US wages grew slowly after 1987 because automation sped up while new tasks slowed.
- **What we built:** the paper's model of tasks and its method for measuring these effects, tested on data where we know the true answer.
- **What we saw:** automation always lowered labour's share, but whether wages fell depended on how much better the machines were ("so-so automation" lowers wages). The measurement method underestimated both effects, as the paper itself warns.

<details>
<summary><b>Read the detailed explanation of this paper</b></summary>

**The question.** Does technology destroy jobs or create them? Economists Daron Acemoglu and Pascual Restrepo argue that the answer depends on **what kind** of technology it is, and they measure it for the US economy.

**Topic 1: thinking in tasks.** Production is a long list of **tasks**. Each task is done either by a machine (capital) or by a person (labour). Two kinds of technology change the list:
- **Automation:** machines take over tasks people used to do (the boundary I moves up).
- **New tasks:** new activities appear that people do best (software engineers, technicians; the top of the list N moves up).

**Topic 2: the labour share.** The **labour share** is the fraction of income paid to workers as wages.
- **Simple example (equal substitutability, σ = 1):** tasks run from 0 to 1. If machines do tasks up to I = 0.4, workers do 60% of tasks and the labour share is 0.6.
- **Automation** up to I = 0.5 gives a labour share of 0.5: always lower.
- **New tasks** shifting the range to [0.1, 1.1] with I = 0.4 give 0.7: always higher.

**Topic 3: three effects.**
- **Displacement effect** (automation): workers lose tasks. It always reduces labour demand.
- **Reinstatement effect** (new tasks): workers gain tasks. It always increases labour demand.
- **Productivity effect** (both): cheaper production raises demand for everything, including labour in the remaining tasks.

So automation raises wages **only if** its productivity effect beats its displacement effect.

**Topic 4: "so-so automation".** Some technologies replace workers with machines that are **only slightly cheaper** (automated phone menus, self-checkout). They displace workers but add little productivity, so wages can **fall**. Automation that is much more productive tends to raise wages despite displacement.

**Topic 5: measuring it in data.** The paper splits the growth of total US wages into:
- **productivity** (GDP growth);
- **composition** (activity moving between more and less labour-intensive industries);
- **substitution** (labour share changes predicted by changing wages and capital costs);
- **task content**, which is what's left over and is then split into **displacement** (industries where it fell) and **reinstatement** (where it rose).

**Topic 6: the findings for the US.**

| Period | Wage bill growth per year | Displacement | Reinstatement |
|---|---|---|---|
| 1947–1987 | 2.5% | −0.48% | +0.47% (balanced) |
| 1987–2017 | 1.33% | −0.70% (faster) | +0.35% (slower) |

After 1987, automation accelerated while new-task creation slowed, so labour demand grew much more slowly than productivity. In manufacturing, displacement was about 30% cumulatively.

**Topic 7: the link to AI.** Will AI mostly automate existing tasks (displacement, possibly "so-so"), or create new tasks for people (reinstatement)? The authors argue that the **direction** of technology matters as much as its size, and that it can be steered by research choices and policy.

**What we built and saw.** The paper's task model and its wage-bill decomposition, tested on synthetic industries where we know the truth.
- Automation lowered the labour share for **every** substitutability value tested.
- Whether **wages** fell depended on how productive the machines were: −3.5% for "so-so" machines, +4.5% for very productive ones.
- The decomposition recovered the net change in task content well, but **underestimated** both displacement and reinstatement (more so when industries automate and add tasks at the same time). That is why the paper calls its estimates **lower bounds**.

**Why it matters.** This task framework is now the standard way economists study the effect of AI on jobs, including measures of which occupations are most exposed to large language models.

</details>

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
