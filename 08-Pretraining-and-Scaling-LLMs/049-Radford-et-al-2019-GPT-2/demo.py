"""A ~8-second tour of GPT-2.

  1. Byte-level BPE (Section 2.2): any Unicode string, invertible, and no 'dog.' / 'dog!' / 'dog?' tokens
  2. Table 2's model sizes: the paper's labels vs the exact parameter counts of the architecture
  3. Pre-LN + 1/sqrt(N) residual init (Section 2.3): the residual stream stays tame in deep stacks
  4. Zero-shot task transfer (the paper's thesis), in a toy 'web text': facts written as prose for every country,
     question-answer demonstrations for only half of them. Can the LM answer Q/A-format questions about the
     other half, with no fine-tuning?
  5. Section 4's 8-gram Bloom-filter overlap check between a 'training set' and 'test sets'
"""

import math
import random
import time

import torch

torch.set_num_threads(2)
from gpt2 import (GPT2, SIZES, BloomFilter, ByteBPE, model_size, ngrams, normalize, overlap_fraction,  # noqa: E402
                  pretokenize)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------------------------------- 1
section("1. Byte-level BPE")
text = ("The dog ran home. The dog barked! Was it a dog? Dogs and cats play together; the cat sat on the mat. " * 20)
bpe = ByteBPE(text, 80)
for s in ("dog. dog! dog?", "naïve café 🐶 漢字"):
    ids = bpe.encode(s)
    print(f"  {s!r:22} -> {len(ids):2d} tokens {[bpe.vocab[i] for i in ids][:8]}{' ...' if len(ids) > 8 else ''}"
          f"   decodes back exactly: {bpe.decode(ids) == s}")
print(f"  pre-tokenizer pieces: {pretokenize('I said: dog, dog!')}")
print(f"  base vocabulary 256 bytes + {len(bpe) - 256} learned merges; the paper's has 50,257 tokens.")
print("  -> bytes mean nothing is ever out of vocabulary (unseen characters fall back to bytes); splitting by character")
print("     category stops BPE from wasting vocabulary on 'dog.', 'dog!', 'dog?'.")

# --------------------------------------------------------------------------------------------- 2
section("2. Table 2: model sizes")
print(f"  {'label':>7} {'layers':>6} {'d_model':>7} {'exact count':>12}")
for k, (L, d) in SIZES.items():
    print(f"  {k:>7} {L:6d} {d:7d} {model_size(L, d) / 1e6:11.1f}M")
print("  -> the exact counts (tied embeddings, vocabulary 50,257, context 1024) are larger than the paper's labels: the")
print("     paper's figures are known to be slight undercounts; the smallest model has 124M parameters, not 117M.")

# --------------------------------------------------------------------------------------------- 3
section("3. Residual stream at initialisation, 48 layers, d = 64 (std of x after all blocks, before the final LN)")
torch.manual_seed(0)
x = torch.randint(0, 100, (4, 32))
for scaled in (False, True):
    torch.manual_seed(0)
    m = GPT2(vocab=100, n_ctx=32, d=64, layers=48, heads=4, scaled_init=scaled).eval()
    with torch.no_grad():
        h = m.wte(x) + m.wpe(torch.arange(32))
        s0 = h.std().item()
        causal = torch.triu(torch.ones(32, 32, dtype=torch.bool), 1)
        for b in m.blocks:
            h = b(h, causal)
    print(f"  residual init {'0.02 / sqrt(96)' if scaled else '0.02 (unscaled)  '}: std at input {s0:.3f} -> after 48 "
          f"blocks {h.std().item():.3f}")
print("  -> each of the N = 96 residual branches adds a little to the stream; scaling their output weights by")
print("     1/sqrt(N) keeps the total added variance roughly independent of depth. Pre-LN keeps every sub-block's input")
print("     normalized, and the extra final LN normalizes the sum before the output layer.")

# --------------------------------------------------------------------------------------------- 4
section("4. Zero-shot task transfer in a toy 'web text' (word-level, 2-layer model, d = 64)")
rng = random.Random(0)
SYL = ["ka", "lo", "mi", "ra", "ze", "tu", "vo", "ni", "be", "sa", "do", "fe"]


def name():
    return rng.choice(SYL) + rng.choice(SYL) + rng.choice(["n", "r", "s", "l"])


countries = list(dict.fromkeys(name() for _ in range(80)))[:40]
capital = {c: name() + "a" for c in countries}
with_qa = set(countries[:20])                                                      # Q/A demos exist for these only
filler = ["the weather was nice today .", "people like to travel .", "many towns have markets .", "the river is long ."]


def document():
    c = rng.choice(countries)
    out = []
    for _ in range(3):
        r = rng.random()
        if r < 0.4:
            out.append(rng.choice([f"the capital of {c} is {capital[c]} .", f"{capital[c]} is the capital of {c} .",
                                   f"in {c} , the capital city {capital[c]} is busy ."]))
        elif r < 0.6 and c in with_qa:
            out.append(f"q : what is the capital of {c} ? a : {capital[c]} .")
        else:
            out.append(rng.choice(filler))
    return " ".join(out).split()


docs = [document() for _ in range(8000)]
words = sorted({w for d in docs for w in d} | set(countries) | set(capital.values()) | {"q", ":", "what", "?", "a"})
V = {w: i for i, w in enumerate(words)}
qa_docs = sum("q" in d for d in docs)
print(f"  8,000 documents; {qa_docs} contain a Q/A demonstration, and only for countries 1-20")
torch.manual_seed(0)
lm = GPT2(len(V), n_ctx=64, d=64, layers=2, heads=4)
opt = torch.optim.AdamW(lm.parameters(), 3e-3)
for _ in range(800):
    batch = [docs[i] for i in torch.randint(0, len(docs), (32,))]
    T = max(len(b) for b in batch)
    loss = lm.loss(torch.tensor([[V[w] for w in b] + [V["."]] * (T - len(b)) for b in batch]))
    opt.zero_grad(); loss.backward(); opt.step()
lm.eval()


def answer(prompt):
    return words[lm.generate([V[w] for w in prompt.split()], 1, greedy=True)[0]]


def accuracy(cs, template):
    return sum(answer(template.format(c=c)) == capital[c] for c in cs) / len(cs)


qa = "q : what is the capital of {c} ? a :"
print(f"  Q/A prompt '{qa}':")
print(f"    countries 1-20 (Q/A format seen in training):        {100 * accuracy(countries[:20], qa):.0f}%")
print(f"    countries 21-40 (facts ONLY as prose, never as Q/A): {100 * accuracy(countries[20:], qa):.0f}%")
print(f"  prose prompt 'the capital of {{c}} is', countries 21-40: {100 * accuracy(countries[20:], 'the capital of {c} is'):.0f}%")
print("  -> the format ('q : ... a :') was learned from demonstrations on some facts and transferred to facts the model")
print("     only ever read as prose: a task performed with no fine-tuning, because predicting text well required it.")
print("     (At this toy scale it works with a 2-layer model; we did not reproduce the paper's 'bigger is better'.)")

# --------------------------------------------------------------------------------------------- 5
section("5. Section 4: how much of a test set already appears in the training data? (8-gram Bloom filter)")
rng2 = random.Random(1)
vocab = [f"w{i}" for i in range(400)]
train_docs = [" ".join(rng2.choice(vocab) for _ in range(60)) for _ in range(300)]
bloom = BloomFilter(m=2 ** 20, k=5)
for d in train_docs:
    for g in ngrams(normalize(d).split()):
        bloom.add(g)
fresh = " ".join(rng2.choice(vocab) for _ in range(600))
leaky = fresh[:1800] + " " + train_docs[3] + " " + train_docs[7]
print(f"  filter: {bloom.n:,} training 8-grams in {bloom.m:,} bits, k = {bloom.k}; "
      f"estimated false-positive rate {bloom.false_positive_rate():.1e}")
print(f"  fresh test text: {100 * overlap_fraction(bloom, fresh):.1f}% of its 8-grams found in training")
print(f"  test text containing two copied training documents: {100 * overlap_fraction(bloom, leaky):.1f}%")
print("  -> the paper found 1-6% overlap between common test sets and WebText (average 3.2%), often smaller than the")
print("     overlap between those test sets and their OWN training splits (5.9%).")

print(f"\n(total {time.time() - T0:.1f} s)")
