"""A ~12-second tour of GPT (generative pre-training, then discriminative fine-tuning).

Toy world: synthetic movie reviews built from sentences like 'the acting was brilliant .' (80% of a review's
sentences share its sentiment); unlabeled reviews end with 'overall it was very good/bad .'.

  1. Sizes: the paper's 12-layer, 768-wide model has ~117M parameters
  2. Task input transformations (Figure 1): structured inputs become one token sequence
  3. Pre-train a tiny GPT on 5,000 unlabeled reviews (Eq. 1)
  4. Zero-shot (Section 5): append 'very', compare P(good) vs P(bad); no labels at all
  5. Fine-tune with only 40 labels (Eqs. 3-5): pre-trained vs from scratch, with and without the auxiliary LM loss
  6. Figure 2 (left): how many pre-trained layers are transferred
"""

import copy
import random
import time

import torch

torch.set_num_threads(2)
from gpt import (GPT, FineTuner, Specials, classification_input, count_params, entailment_input,  # noqa: E402
                 multiple_choice_inputs, optimizer, pad_batch, similarity_inputs, transfer_layers,
                 zero_shot_sentiment)

T0 = time.time()
POS = ["great", "wonderful", "lovely", "brilliant", "charming", "fun"]
NEG = ["awful", "boring", "dull", "terrible", "clumsy", "bland"]
NOUN = ["movie", "film", "book", "show", "play", "story"]
ASP = ["acting", "plot", "music", "ending", "pacing", "cast"]
WORDS = ["the", "was", ".", "i", "saw", "with", "a", "friend", "released", "last", "year", "overall", "it", "very",
         "good", "bad"] + POS + NEG + NOUN + ASP
V = {w: i + 1 for i, w in enumerate(WORDS)}                                        # 0 = padding


def review(rng, s, final=True):
    n = rng.choice(NOUN)
    out = rng.choice([f"i saw the {n} with a friend .", f"the {n} was released last year ."]).split()
    for _ in range(rng.randint(2, 4)):
        pol = s if rng.random() < 0.8 else 1 - s
        out += f"the {rng.choice(ASP)} was {rng.choice(POS if pol else NEG)} .".split()
    if final:
        out += ("overall it was very " + ("good" if s else "bad") + " .").split()
    return [V[w] for w in out]


def section(t):
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------------------------------- 1
section("1. Sizes")
with torch.device("meta"):
    big = GPT(40478)
print(f"  12 layers, d = 768, 12 heads, FFN 3072, context 512, BPE vocab ~40k: {count_params(big) / 1e6:.1f}M "
      f"parameters (the paper's model is usually quoted as 117M)")

# --------------------------------------------------------------------------------------------- 2
section("2. Task input transformations (Figure 1), shown with words")
sp = Specials("<s>", "$", "<e>")
print("  classification:  ", " ".join(classification_input(["the", "plot", "was", "dull"], sp)))
print("  entailment:      ", " ".join(entailment_input(["a", "man", "sleeps"], ["a", "person", "rests"], sp)))
for k, s in enumerate(similarity_inputs(["he", "left"], ["he", "departed"], sp)):
    print(f"  similarity ({k + 1}/2):", " ".join(s), "  -> the two final states are ADDED")
for s in multiple_choice_inputs(["the", "cat", "sat", "?"], [["yes"], ["no"]], sp):
    print("  multiple choice: ", " ".join(s), "  -> one score each, softmax across answers")
print("  -> the pre-trained network is never changed; only <s>, $, <e> embeddings and W_y are new.")

# --------------------------------------------------------------------------------------------- 3
section("3. Pre-training a tiny GPT (2 layers, d = 64) on 5,000 unlabeled reviews")
rng = random.Random(0)
corpus = [review(rng, rng.randint(0, 1)) for _ in range(5000)]
labeled = [(review(rng, s, False), s) for s in [rng.randint(0, 1) for _ in range(40)]]
test = [(review(rng, s, False), s) for s in [rng.randint(0, 1) for _ in range(500)]]
make = lambda dropout: GPT(len(V) + 1, n_ctx=48, d=64, layers=2, heads=4, ff=256, dropout=dropout)
torch.manual_seed(0)
lm = make(0.0)
opt = optimizer(lm, 3e-3)
for step in range(1, 1001):
    ids, mask = pad_batch([corpus[i] for i in torch.randint(0, len(corpus), (32,))])
    loss = lm.lm_loss(ids, mask)
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (300, 1000):
        zs = sum(zero_shot_sentiment(lm, x + [V["overall"], V["it"], V["was"]], V["very"], V["good"], V["bad"]) == y
                 for x, y in test) / len(test)
        print(f"  step {step:4d}: LM loss {loss.item():.3f} nats/token, zero-shot sentiment accuracy {100 * zs:.1f}%")

# --------------------------------------------------------------------------------------------- 4
section("4. Zero-shot (no labels): append 'overall it was very', compare P(good) and P(bad)")
print(f"  accuracy on 500 test reviews: {100 * zs:.1f}% (chance 50%)")
print("  -> to predict the review's last word, the language model had to learn to read sentiment from the whole review.")
print("     The paper saw these zero-shot abilities rise steadily during pre-training (Figure 2, right); here the skill")
print("     appears abruptly (near chance at step 300).")


# --------------------------------------------------------------------------------------------- 5
def finetune(lam=0.5, pretrained=True, layers=None, seed=0):
    torch.manual_seed(seed)
    fresh = make(0.1)
    if layers is not None:
        g = transfer_layers(lm, fresh, layers)
    else:
        g = copy.deepcopy(lm) if pretrained else fresh
    sp = Specials(*g.extend_vocab(3))
    f = FineTuner(g, 2)
    o = optimizer(f, 5e-4)
    ids, mask = pad_batch([classification_input(x, sp) for x, _ in labeled])
    y = torch.tensor([s for _, s in labeled])
    for _ in range(40):
        f.train()
        loss = f.loss(ids, mask, y, lam)
        o.zero_grad(); loss.backward(); o.step()
    f.eval()
    ti, tm = pad_batch([classification_input(x, sp) for x, _ in test])
    with torch.no_grad():
        return (f(ti, tm).argmax(1) == torch.tensor([s for _, s in test])).float().mean().item()


section("5. Fine-tuning with only 40 labeled reviews (3 seeds each)")
rows = [("pre-trained, lambda = 0.5 (L3 = L2 + 0.5 L1)", dict(lam=0.5)),
        ("pre-trained, lambda = 0 (no auxiliary LM)", dict(lam=0.0)),
        ("no pre-training (random init)", dict(lam=0.0, pretrained=False))]
for name, kw in rows:
    accs = [finetune(seed=s, **kw) for s in range(3)]
    print(f"  {name:46s} {' '.join(f'{100 * a:.1f}%' for a in accs)}   mean {100 * sum(accs) / 3:.1f}%")
print("  -> pre-training helps (the paper: -14.8 points average without it). The auxiliary LM loss makes no clear")
print("     difference on 40 examples, in line with the paper: 'larger datasets benefit ..., smaller datasets do not'.")
print("     Note zero-shot (section 4) beat all of these: here, 40 labels did not get fine-tuning past the zero-shot score.")

# --------------------------------------------------------------------------------------------- 6
section("6. Figure 2 (left): transferring 0, 1 or 2 pre-trained layers (embeddings always transferred)")
for n in (0, 1, 2):
    accs = [finetune(lam=0.5, layers=n, seed=s) for s in range(3)]
    print(f"  {n} layer(s): {' '.join(f'{100 * a:.1f}%' for a in accs)}   mean {100 * sum(accs) / 3:.1f}%")
print("  -> each transferred layer adds accuracy, as in the paper (up to +9% on MultiNLI for full transfer).")
print("     Odd detail: with embeddings only, every seed predicts a single class (50.2%), worse than no pre-training at")
print("     all (section 5); we did not investigate why. (2 layers here, 75%, differs from section 5's 80% only in using")
print("     a dropout-0.1 copy of the network.)")

print(f"\n(total {time.time() - T0:.1f} s)")
