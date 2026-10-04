# The reading path: 103 papers in the order that makes them easiest

Every paper from all the lists, in **one** order. Each paper only needs the ones **before** it.
Each stage is a folder in this repository, and each paper has its own folder inside it, named by its number (e.g. `01-Foundations/004-Rumelhart-Hinton-Williams-1986/`). The PDF is in there, and so are the code and explanations once the paper is built.

**How to use this**
- Go stage by stage, top to bottom. **Don't skip a stage**: later papers assume the earlier ideas.
- **★** = short and easy, **★★** = some math, **★★★** = long or hard (read the parts marked).
- **✅** = already implemented in this repository (a folder with the same number).
- Papers marked *(optional)* are good but not needed for the next stage.

**For each paper, read in 3 passes** (from your list):
1. **15 minutes:** abstract → introduction → figures → conclusion. What's the one new idea?
2. **1–2 hours:** the method and experiments. Rewrite the key equations yourself.
3. **Build it:** a small version in code, and check it against the paper's results.

---

## Stage 01: Foundations. What is a neural network?
**Why first:** every later paper uses these ideas: neurons, weights, learning from errors, and the limits of one layer.

| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 001 | McCulloch & Pitts (1943), *A Logical Calculus of the Ideas Immanent in Nervous Activity* | A neuron is a logic gate; networks compute | ★★ | ✅ |
| 002 | Rosenblatt (1958), *The Perceptron* | The first network that **learns** | ★★ | ✅ |
| 003 | Minsky & Papert (1969), *Perceptrons* (Introduction) | What one layer **can't** do | ★★ | ✅ |
| 004 | Rumelhart, Hinton & Williams (1986), *Learning Representations by Back-Propagating Errors* | **Backpropagation**: training hidden layers | ★ (4 pages) | ✅ |
| 005 | Du et al. (2022), *Perceptron: Learning, Generalization, …* | A review of all of the above, and of training algorithms | ★★ | ✅ |

## Stage 02: Training deep networks. Making learning actually work
**Why here:** backprop works in theory, but deep networks train badly without these tricks. You'll use every one of them later.

| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 006 | LeCun, Bottou, Orr & Müller (1998), *Efficient BackProp* | Practical tips: normalize inputs, pick learning rates, choose activations | ★★ (44 pages) | ✅ |
| 007 | Glorot & Bengio (2010), *Understanding the Difficulty of Training Deep Feedforward Neural Networks* | Why deep nets get stuck; **Xavier initialization** | ★★ |✅ |
| 008 | Sutskever, Martens, Dahl & Hinton (2013), *On the Importance of Initialization and Momentum in Deep Learning* | **Nesterov momentum** + good initialization | ★★ |✅ |
| 009 | Hinton et al. (2012), *Improving Neural Networks by Preventing Co-adaptation of Feature Detectors* | The first **dropout** idea | ★ |✅ |
| 010 | Srivastava et al. (2014), *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* | Dropout, fully explained | ★★ (30 pages) |✅ |
| 011 | Kingma & Ba (2015), *Adam: A Method for Stochastic Optimization* | **Adam**, the default optimizer | ★★ | ✅ |
| 012 | Ioffe & Szegedy (2015), *Batch Normalization* | Normalizing inside the network | ★★ | ✅ |

## Stage 03: CNNs and computer vision. Networks for images
**Why here:** CNNs are the first big deep-learning success. They use everything from Stage 02.

| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 013 | LeCun et al. (1998), *Gradient-Based Learning Applied to Document Recognition* (LeNet-5) | **Convolutions**, pooling, weight sharing | ★★★ (read Sections I–III) | ✅ |
| 014 | Krizhevsky, Sutskever & Hinton (2012), *ImageNet Classification with Deep CNNs* (AlexNet) | ReLU + dropout + GPUs = the deep learning revolution | ★ | ✅ |
| 015 | Simonyan & Zisserman (2015), *Very Deep Convolutional Networks* (VGG) | Deeper is better: stacks of small 3×3 filters | ★ | ✅ |
| 016 | He et al. (2016), *Deep Residual Learning* (ResNet) | **Skip connections**: how to train 100+ layers | ★★ | ✅ |

## Stage 04: What networks really learn. Robustness and generalization
**Why here:** now that you've seen big networks work, two surprising papers show their strange side.

| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 017 | Szegedy et al. (2013), *Intriguing Properties of Neural Networks* | **Adversarial examples**: invisible changes fool the network | ★★ | ✅ |
| 018 | Zhang et al. (2017), *Understanding Deep Learning Requires Rethinking Generalization* | Networks can memorize **random labels** | ★ | ✅ |

## Stage 05: Words and sequences. RNNs, LSTMs, Seq2Seq, attention
**Why here:** text and speech come in sequences. This stage builds up, step by step, to **attention**, the key idea behind transformers.

| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 019 | Mikolov et al. (2013), *Distributed Representations of Words and Phrases* (Word2Vec) | Words as vectors; **embeddings** | ★ | ✅ |
| 020 | Mikolov, Le & Sutskever (2013), *Exploiting Similarities among Languages for MT* | Word vectors line up across languages *(optional)* | ★ | ✅ |
| 021 | Hochreiter & Schmidhuber (1997), *Long Short-Term Memory* | The **LSTM**: memory that doesn't fade | ★★★ (read Sections 1–4) | ✅ |
| 022 | Sutskever, Martens & Hinton (2011), *Generating Text with Recurrent Neural Networks* | Character-level text generation | ★★ | ✅ |
| 023 | Sutskever (2013), *Training Recurrent Neural Networks* (PhD thesis) | Deep dive into RNN training *(optional, 101 pages)* | ★★★ | ✅ |
| 024 | Zaremba, Sutskever & Vinyals (2014), *Recurrent Neural Network Regularization* | Dropout for LSTMs, done right | ★ | ✅ |
| 025 | Jozefowicz, Zaremba & Sutskever (2015), *An Empirical Exploration of Recurrent Network Architectures* | Which LSTM parts matter (the forget gate) | ★★ | ✅ |
| 026 | Cho et al. (2014), *Learning Phrase Representations using RNN Encoder–Decoder* | The **GRU** and encoder–decoder | ★★ | ✅ |
| 027 | Sutskever, Vinyals & Le (2014), *Sequence to Sequence Learning* | **Seq2Seq**: translate a whole sentence | ★ | ✅ |
| 028 | Bahdanau, Cho & Bengio (2015), *Neural Machine Translation by Jointly Learning to Align and Translate* | **Attention**: look back at the right words | ★★ | ✅ |
| 029 | Vinyals et al. (2015), *Show and Tell* | Image → sentence (CNN + LSTM) | ★ | ✅ |
| 030 | Vinyals et al. (2015), *Grammar as a Foreign Language* | Seq2Seq for parsing *(optional)* | ★ | ✅ |
| 031 | Vinyals, Fortunato & Jaitly (2015), *Pointer Networks* | Attention that **points** at the input | ★★ | ✅ |
| 032 | Vinyals, Bengio & Kudlur (2015), *Order Matters* | Input order changes Seq2Seq results *(optional)* | ★★ | ✅ |
| 033 | Kalchbrenner et al. (2016), *Neural Machine Translation in Linear Time* (ByteNet) | Convolutions instead of RNNs for sequences *(optional)* | ★★ | ✅ |

## Stage 06: Transformers. Attention is all you need
**Why here:** you now know embeddings, Seq2Seq and attention, which are exactly the pieces the transformer is made of.

| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 034 | Vaswani et al. (2017), *Attention Is All You Need* | **The Transformer**: self-attention, multi-head, positional encoding | ★★ | ✅ |
| 035 | Shazeer (2019), *Fast Transformer Decoding: One Write-Head is All You Need* | **Multi-Query Attention**: faster inference | ★★ | ✅ |
| 036 | Child et al. (2019), *Generating Long Sequences with Sparse Transformers* | Sparse attention for long inputs | ★★ | ✅ |
| 037 | Devlin et al. (2019), *BERT* | **Bidirectional** pretraining with masked words | ★★ | ✅ |
| 038 | Dosovitskiy et al. (2021), *An Image is Worth 16x16 Words* (ViT) | Transformers for **images** | ★★ | ✅ |

## Stage 07: Generative models. Networks that create
**Why here:** these need probability plus the networks you know. VAEs and GANs are the two big ideas; the rest improve on them.

| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 039 | Kingma & Welling (2014), *Auto-Encoding Variational Bayes* | The **VAE** and the reparameterization trick | ★★★ | ✅ |
| 040 | Kingma et al. (2016), *Improved Variational Inference with Inverse Autoregressive Flow* | Better VAEs with flows *(optional)* | ★★★ | ✅ |
| 041 | Chen et al. (2017), *Variational Lossy Autoencoder* | Controlling what a VAE stores *(optional)* | ★★★ | ✅ |
| 042 | Goodfellow et al. (2014), *Generative Adversarial Nets* | **GANs**: generator vs discriminator | ★★ | ✅ |
| 043 | Chen et al. (2016), *InfoGAN* | GANs with meaningful controls | ★★ | ✅ |
| 044 | Arjovsky, Chintala & Bottou (2017), *Wasserstein GAN* | Stable GAN training | ★★★ | ✅ |
| 045 | Salimans et al. (2017), *PixelCNN++* | Generating images pixel by pixel *(optional)* | ★★ | ✅ |
| 046 | Kingma & Dhariwal (2018), *Glow* | Invertible networks (normalizing flows) *(optional)* | ★★★ | ✅ |
| 047 | Ramesh et al. (2021), *Zero-Shot Text-to-Image Generation* (DALL-E) | Text → image with a transformer | ★★ | ✅ |

## Stage 08: Pretraining and scaling. How LLMs are made
**Why here:** you know transformers (Stage 06). Now: pretrain them on huge text, then make them bigger.

| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 048 | Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training* (GPT) | Pretrain, then fine-tune | ★ | ✅ |
| 049 | Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2) | One model does many tasks without fine-tuning | ★ | ✅ |
| 050 | Brown et al. (2020), *Language Models are Few-Shot Learners* (GPT-3) | **In-context learning** at 175B parameters | ★★ (read Sections 1–3) | ✅ |
| 051 | Kaplan et al. (2020), *Scaling Laws for Neural Language Models* | Loss falls as a **power law** with size, data and compute | ★★ | ✅ |
| 052 | Hoffmann et al. (2022), *Training Compute-Optimal LLMs* (Chinchilla) | **Correction to Kaplan:** train on far more data | ★★ | ✅ |
| 053 | Radford et al. (2021), *CLIP* | Images and text in one space; zero-shot vision | ★★ | ✅ |
| 054 | Chen et al. (2021), *Evaluating LLMs Trained on Code* (Codex) | Code models; **pass@k** evaluation | ★ | ✅ |
| 055 | Radford et al. (2023), *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper) | Speech recognition from 680k hours | ★ | ✅ |
| 056 | Touvron et al. (2023), *LLaMA* | Open models: RMSNorm, SwiGLU, RoPE | ★ | ✅ |
| 057 | OpenAI (2023), *GPT-4 Technical Report* | What's reported (and not) about GPT-4 *(optional)* | ★ | ✅ |
| 058 | OpenAI (2024), *GPT-4o System Card* | How safety evaluation is done *(optional)* | ★ | ✅ |

## Stage 09: Reasoning and agents. Getting more out of LLMs
**Why here:** you know what LLMs are. Now: make them reason, check their work, and use tools.

| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 059 | Wei et al. (2022), *Chain-of-Thought Prompting* | Step-by-step examples → better reasoning | ★ | ✅ |
| 060 | Kojima et al. (2022), *Large Language Models are Zero-Shot Reasoners* | "Let's think step by step" | ★ | ✅ |
| 061 | Cobbe et al. (2021), *Training Verifiers to Solve Math Word Problems* | A second model **checks** answers (GSM8K) | ★★ | ✅ |
| 062 | Yao et al. (2023), *ReAct* | **Reason + act** loops, the basis of agents | ★ | ✅ |
| 063 | Nakano et al. (2021), *WebGPT* | An LLM that **browses the web** | ★★ | ✅ |
| 064 | Park et al. (2023), *Generative Agents* | Agents with memory, reflection, planning | ★ | ✅ |
| 065 | Wang et al. (2023), *A Survey on LLM-based Autonomous Agents* | The whole agent field *(survey)* | ★★ | |

## Stage 10: Alignment. Teaching models what people want
**Why here:** it builds on pretraining (Stage 08). This is how ChatGPT-style assistants are trained.

| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 066 | Stiennon et al. (2020), *Learning to Summarize from Human Feedback* | **Reward models** + RL from human feedback | ★★ | |
| 067 | Ouyang et al. (2022), *Training Language Models to Follow Instructions* (InstructGPT) | The full **RLHF** recipe: SFT → reward model → PPO | ★★ | |
| 068 | Bai et al. (2022), *Constitutional AI* | AI feedback instead of human feedback | ★★ | |
| 069 | Rafailov et al. (2023), *Direct Preference Optimization* (DPO) | Alignment **without** RL, as a simple loss | ★★ | |

## Stage 11: Retrieval (RAG). Giving models knowledge
**Why here:** it uses embeddings (Stage 05) and transformers (Stage 06). It's one of the most-used patterns in AI engineering.

| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 070 | Karpukhin et al. (2020), *Dense Passage Retrieval* | Search with embeddings instead of keywords | ★★ | |
| 071 | Lewis et al. (2020), *Retrieval-Augmented Generation* | **RAG**: retrieve, then generate | ★★ | |
| 072 | Izacard et al. (2022), *Atlas* | Retrieval + few-shot learning | ★★ | |

## Stage 12: Efficient fine-tuning and quantization. Big models on small hardware
**Why here:** you need to know transformers and LLMs first. These are daily tools for an AI engineer.

| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 073 | Hu et al. (2022), *LoRA* | Fine-tune with small **low-rank** updates | ★★ | |
| 074 | Xiao et al. (2023), *SmoothQuant* | **INT8** inference: move the difficulty from activations to weights | ★★ | |
| 075 | Frantar et al. (2023), *GPTQ* | **4-bit** weights, after training | ★★★ | |
| 076 | Lin et al. (2023), *AWQ* | Protect the important weights when quantizing | ★★ | |
| 077 | Dettmers et al. (2023), *QLoRA* | LoRA on a 4-bit model: fine-tune on one GPU | ★★ | |
| 078 | Liu et al. (2023), *LLM-QAT* | Quantization-**aware** training | ★★ | |

## Stage 13: Systems. Fast inference and huge-scale training
**Why here:** now you know what's being computed; this is about computing it **fast** and at **scale**.

| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 079 | Dao et al. (2022), *FlashAttention* | **Tiled** attention that respects GPU memory | ★★★ | |
| 080 | Pope et al. (2022), *Efficiently Scaling Transformer Inference* | Partitioning, KV cache, latency vs throughput | ★★★ | |
| 081 | Zinkevich et al. (2010), *Parallelized Stochastic Gradient Descent* | The start of distributed SGD *(PDF: see below)* | ★★ | |
| 082 | Abadi et al. (2015), *TensorFlow: … Heterogeneous Distributed Systems* | Dataflow graphs *(optional)* | ★★ | |
| 083 | Abadi et al. (2016), *TensorFlow: A System for Large-Scale ML* | The production TensorFlow design | ★★ | |
| 084 | Huang et al. (2019), *GPipe* | **Pipeline** parallelism with micro-batches | ★★ | |
| 085 | Narayanan et al. (2019), *PipeDream* | Pipeline parallelism, improved | ★★ | |
| 086 | Rajbhandari et al. (2020), *ZeRO* | Split optimizer state across GPUs (DeepSpeed) | ★★ | |

## Stage 14: Production ML (MLOps). Keeping ML working in the real world
**Why last among the core stages:** these papers make the most sense once you've built models yourself.

| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 087 | Sculley et al. (2014), *Machine Learning: The High-Interest Credit Card of Technical Debt* | Why ML code rots | ★ | |
| 088 | Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems* | The full picture: the model is a tiny part of the system | ★ | |
| 089 | Breck et al. (2017), *The ML Test Score* | A 28-test checklist for production readiness | ★ | |
| 090 | Zinkevich, *Rules of Machine Learning* (Google, web page) | 43 practical rules | ★ | see links below |
| 091 | Rabanser et al. (2019), *Failing Loudly* | Detecting **data drift** | ★★ | |
| 092 | Huyen, *Data Distribution Shifts and Monitoring* (blog post) | Drift in practice | ★ | see links below |
| 093 | Mitchell et al. (2019), *Model Cards for Model Reporting* | Documenting models responsibly | ★ | |
| 094 | Huyen (2022), *Designing Machine Learning Systems* (book) | The whole ML-system lifecycle | ★★ | book, not free |

## Stage 15: Reinforcement learning *(optional)*
| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 095 | Salimans et al. (2017), *Evolution Strategies as a Scalable Alternative to RL* | Optimization without gradients | ★★ | |
| 096 | Berner et al. (2019), *Dota 2 with Large Scale Deep RL* (OpenAI Five) | RL at huge scale | ★★ | |
| 097 | Silver et al. (2016), *Mastering the Game of Go* (AlphaGo) | Deep RL + tree search | ★★★ | Nature paywall, see below |

## Stage 16: Extras *(optional)*
| # | Paper | You'll learn | Level | Status |
|---|---|---|---|---|
| 098 | Khosla et al. (2020), *Supervised Contrastive Learning* | Contrastive learning with labels | ★★ | |
| 099 | Coates & Ng (2011), *The Importance of Encoding Versus Training …* | Early feature learning | ★★ | |
| 100 | Sutskever & Hinton (2008), *Using Matrices to Model Symbolic Relationships* | Relational learning | ★★ | see below |
| 101 | Sutskever, Salakhutdinov & Tenenbaum (2009), *Bayesian Clustered Tensor Factorization* | Tensor models of relations | ★★★ | see below |
| 102 | Brundage et al. (2018), *The Malicious Use of Artificial Intelligence* | AI security threats | ★ | |
| 103 | Acemoglu & Restrepo (2019), *Automation and New Tasks* | AI's effect on jobs (economics) | ★★ | see below |

---

## Not downloaded: get these by hand if you want them

| # | What | Where |
|---|---|---|
| 081 | Zinkevich et al. (2010), *Parallelized SGD* | NeurIPS 2010 proceedings: search the title on papers.nips.cc |
| 090 | Zinkevich, *Rules of Machine Learning* | https://developers.google.com/machine-learning/guides/rules-of-ml (web page) |
| 092 | Huyen, *Data Distribution Shifts and Monitoring* | Chip Huyen's blog, huyenchip.com (web page) |
| 094 | Huyen, *Designing Machine Learning Systems* | O'Reilly book (paid) |
| 097 | Silver et al. (2016), *AlphaGo* | Nature (paywalled): doi 10.1038/nature16961 |
| 100, 101 | Sutskever's 2008 and 2009 papers | NeurIPS proceedings (papers.nips.cc), search the titles |
| 103 | Acemoglu & Restrepo (2019) | Journal of Economic Perspectives 33(2), free on aeaweb.org |

**Business-list items that aren't research papers** (HBR articles, books, blog posts on product-market fit) aren't included: Brynjolfsson & McAfee; Davenport & Ronanki; *Prediction Machines*; the 2021/2025 strategy articles; BVP/CRV/Saha on product-market fit; Schulman's RLHF talk; Wills (2019).

## Corrections to the original lists
- **Pope et al. (2022)** is **not** about FlashAttention. FlashAttention is Dao et al. (079).
- **Kaplan et al. (2020)** is not "the compute-optimal bible". **Chinchilla (052)** corrected it, which is why it's added here.
- **"Ilya's 50 papers":** Sutskever is **not** an author of Coates & Ng (099), *Parallelized SGD* (081), or *Supervised Contrastive Learning* (098). Two entries (#36, #49 in the original list) couldn't be found at all and are left out.
