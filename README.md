# Research Papers Implementation

Implementing the key papers of AI, from the first artificial neuron (1943) to today's LLMs, **from scratch**: reading each paper closely, rebuilding the method in code, and testing that the code reproduces the paper's claims.

**Start here: [READING_ORDER.md](READING_ORDER.md)** has all 103 papers in one learning path, grouped into 16 stages, with why each paper comes where it does.

## Layout

```
01-Foundations/                         ← a stage (16 stages, in learning order)
   001-McCulloch-Pitts-1943/            ← one folder per paper, numbered 001–103
      EXPLAINED.md                      ← the paper, explained simply
      CODE_EXPLAINED.md                 ← how the code works, mapped to the paper
      *.py, demo.py, test_*.py          ← the implementation, a demo, and tests
      *.pdf                             ← the paper (kept locally, not on GitHub)
02-Training-Deep-Networks/
   006-LeCun-et-al-1998-Efficient-BackProp/  ← not built yet: just the PDF for now
...
```

## Built so far

| # | Paper | Year | What's implemented |
|---|---|---|---|
| 001 | [A Logical Calculus of the Ideas Immanent in Nervous Activity](01-Foundations/001-McCulloch-Pitts-1943/) (McCulloch & Pitts) | 1943 | The first artificial neuron: a network simulator, all of Figure 1 (including the heat/cold illusion), memory loops, and a compiler that turns logic formulas into networks (Theorem II). 21 tests. |
| 002 | [The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](01-Foundations/002-Rosenblatt-1958/) (Rosenblatt) | 1958 | The first learning neural network. The probability theory of randomly wired A-units (Eqs. 1–3, checked against simulation), the full photoperceptron with α/γ/bivalent learning, and Figures 4–7, 10, 11: memorizing vs generalizing, distributed memory, trial-and-error learning. 13 tests. |
| 003 | [Perceptrons: An Introduction to Computational Geometry, Introduction](01-Foundations/003-Minsky-Papert-1969/) (Minsky & Papert) | 1969 | What a perceptron can't compute. The formal perceptron, the order-3 convexity perceptron, the seesaw, and machine-checked proofs that connectedness isn't local (Theorems 0.6.1 and 0.8). Extension: parity's order, by linear programming. 26 tests. |
| 004 | [Learning Representations by Back-Propagating Errors](01-Foundations/004-Rumelhart-Hinton-Williams-1986/) (Rumelhart, Hinton & Williams) | 1986 | Backpropagation, Eqs. (1)–(9), on any layered net (checked against finite differences), plus backprop through time (Figure 5). Reproduces the mirror-symmetry net of Figure 1, with the same 1 : 2 : 4 weight structure, and the family-tree net of Figures 2–4, with its learned nationality/generation/branch features. Finds that the paper's family-tree recipe stalls from vanishing gradients. 14 tests. |
| 005 | [Perceptron: Learning, Generalization, Model Selection, Fault Tolerance, and Role in the Deep Learning Era](01-Foundations/005-Du-et-al-2022/) (Du, Leung, Mow & Swamy) | 2022 | A survey of 70 years of perceptron research. Built from scratch in NumPy: the perceptron, pocket and LMS rules; an MLP with backpropagation; 11 training algorithms (momentum, RProp, Levenberg–Marquardt, BFGS, conjugate gradients); early stopping, weight decay and fault injection. Reproduces the paper's Iris experiment (Table 1, Figure 4). 40 tests. |
| 006 | [Efficient BackProp](02-Training-Deep-Networks/006-LeCun-et-al-1998-Efficient-BackProp/) (LeCun, Bottou, Orr & Müller) | 1998 | Every trick (input normalization/whitening, 1.7159·tanh(2x/3), ±1 targets, fan-in init, SGD vs batch) tested on digits; Hessian tools (diagonal Hessian by backprop, Hessian-vector products, power method, on-line eigenvalue); stochastic diagonal Levenberg–Marquardt. Finds the paper's own learning-rate bound (2.38) is wrong: the bias makes it 2.0. 11 tests. |
| 007 | [Understanding the difficulty of training deep feedforward neural networks](02-Training-Deep-Networks/007-Glorot-Bengio-2010-Difficulty-Training-Deep-FF/) (Glorot & Bengio) | 2010 | Standard vs normalized (Xavier) init, sigmoid/tanh/softsign, per-layer activation and gradient monitoring, Jacobian singular values (0.49 vs 0.80, matching the paper's 0.5/0.8), a Shapeset-3×2 generator, Table 1 experiment script. 9 tests. |
| 008 | [On the importance of initialization and momentum in deep learning](02-Training-Deep-Networks/008-Sutskever-et-al-2013-Initialization-and-Momentum/) (Sutskever, Martens, Dahl & Hinton) | 2013 | Classical and Nesterov momentum by hand, Theorem 2.1 checked exactly, the momentum schedule, sparse and echo-state initialization, deep autoencoder and RNN addition-problem scripts. Finds NAG can diverge where CM converges when lr·λ > 1. 11 tests. |
| 009 | [Improving neural networks by preventing co-adaptation of feature detectors](02-Training-Deep-Networks/009-Hinton-et-al-2012-Preventing-Co-adaptation/) (Hinton et al.) | 2012 | Dropout with explicit masks, max-norm, the mean network; proves (by enumerating all 1,024 sub-networks) that it is exactly their geometric mean; MNIST experiment script. 6 tests. |
| 010 | [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](02-Training-Deep-Networks/010-Srivastava-et-al-2014-Dropout/) (Srivastava et al.) | 2014 | ReLU dropout nets, Bernoulli and Gaussian dropout, max-norm and other regularizers, Monte-Carlo averaging, sparsity, dropout = ridge regression (Section 9.1) checked; scripts for Table 9 and Figures 7–11. 7 tests. |
| 011 | [Adam: A Method for Stochastic Optimization](02-Training-Deep-Networks/011-Kingma-Ba-2015-Adam/) (Kingma & Ba) | 2015 | Adam, AdaMax, AdaGrad, RMSProp, AdaDelta, SGD Nesterov and temporal averaging from scratch; matches torch.optim.Adam; bias correction, step bound, scale invariance and AdaGrad-as-a-limit checked; scripts for Figures 1–4 (not run). 17 tests. |
| 012 | [Batch Normalization](02-Training-Deep-Networks/012-Ioffe-Szegedy-2015-Batch-Normalization/) (Ioffe & Szegedy) | 2015 | BN forward and backward by hand (the paper's chain rule, checked against finite differences and PyTorch), Algorithm 2 inference statistics, conv BN, the Section 2 bias-drift example, scale invariance; a hand-written torch BN layer; scripts for Figure 1 and learning-rate/deep-sigmoid findings (not run). 13 tests. |
| 013 | [Gradient-Based Learning Applied to Document Recognition (LeNet-5)](03-CNNs-and-Vision/013-LeCun-et-al-1998-LeNet5-Gradient-Based-Learning/) (LeCun et al.) | 1998 | LeNet-5 exactly as in the paper (C3 partial connections, trainable subsampling, 7x12-bitmap RBF outputs, exactly 60,000 parameters), convolution by hand, MSE vs MAP loss (collapse), distortions, stochastic diagonal Levenberg-Marquardt; scripts for Figures 5, 6, 9 (not run). 17 tests. |
| 014 | [ImageNet Classification with Deep CNNs (AlexNet)](03-CNNs-and-Vision/014-Krizhevsky-et-al-2012-AlexNet/) (Krizhevsky, Sutskever & Hinton) | 2012 | AlexNet with the two-GPU split as grouped convs (60,965,224 parameters), LRN by hand, overlapping pooling, 2012-style dropout, the paper's update rule and init, crops/flips/10-crop/PCA colour augmentation; CIFAR-10 ablation scripts and a full ImageNet-folder script (not run). 13 tests. |
| 015 | [Very Deep Convolutional Networks (VGG)](03-CNNs-and-Vision/015-Simonyan-Zisserman-2015-VGG/) (Simonyan & Zisserman) | 2015 | All six configs of Table 1 with exact Table 2 parameter counts (D = 138,357,544), 3x3 receptive fields measured, init from net A, FC-to-conv dense evaluation, scale jittering, multi-crop; init explosion/vanishing shown; CIFAR-10 scripts for depth, 3x3 vs 5x5, init, jittering (not run). 14 tests. |

## Running

Python 3.10+. Paper 001 needs nothing else; later papers need `numpy`, some `scipy`, `scikit-learn`, `torch`/`torchvision`, and the figures `matplotlib`. The tests need `pytest`.

```bash
cd 01-Foundations/001-McCulloch-Pitts-1943      # or any built paper's folder
python3 demo.py
python3 -m pytest -q
```

## Notes

- From paper 007 on, the heavy experiments (`experiments.py`) are written but **not run** on the author's laptop; each paper's EXPLAINED.md says what was checked and gives the paper's numbers as the reference.

- Paper PDFs are **not** committed (see `.gitignore`); [SOURCES.txt](SOURCES.txt) lists where each was downloaded from.
- Stage 01 follows the classic history: 001–004 are the original papers, and 005 is a modern review of them.
