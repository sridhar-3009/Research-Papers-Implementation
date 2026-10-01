"""A light tour of the multiplicative RNN (Sutskever, Martens & Hinton 2011). A few seconds.

    python3 demo.py
"""

import torch

from mrnn import MRNN, CharRNN, HessianFree, TensorRNN, bits_per_char, count_params, nll, one_hot, sample

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. Three ways to let the input character change the dynamics (sizes for 86 characters)")
for name, make in (("standard RNN, 1500 units", lambda: CharRNN(86, 1500)),
                   ("tensor RNN, 1500 units (one 1500x1500 matrix PER character)", lambda: TensorRNN(86, 1500)),
                   ("MRNN, 1500 units, 1500 factors (the paper's model)", lambda: MRNN(86, 1500, 1500))):
    with torch.device("meta"):
        print(f"{name:62s} {count_params(make()) / 1e6:7.1f}M parameters")
print("-> the full tensor is ~40x too big; factoring it keeps per-character matrices for ~2x the RNN's size.")

line("2. Each character synthesizes its own hidden-to-hidden matrix (Eq. 6)")
m = MRNN(5, 6, 4)
for c in range(3):
    W = m.effective_matrix(c)
    print(f"character {c}: W^(c) has rank {torch.linalg.matrix_rank(W).item()} (<= F = 4), "
          f"first row {[round(v, 3) for v in W[0, :4].tolist()]}...")
print("-> all W^(c) are blends of the same F rank-one matrices, with character-specific gains W_fx[:, c].")

line("3. Why multiplicative? A conjunction of context AND character")
print("An additive RNN computes tanh(A h + B x): the input only SHIFTS the hidden state.")
print("The MRNN computes tanh(W^(x) h + B x): the input changes HOW the past state is used, e.g.")
print("'i' after a verb stem ('fix', 'break') should lead to 'n' (-> 'ing'); 'i' elsewhere should not.")

line("4. Hessian-free optimization learns a repeating pattern in a few steps")
seq = torch.tensor([[0, 1, 2, 3, 2, 1] * 6] * 8).T                      # (36, 8)
model = MRNN(4, 10, 10)
hf = HessianFree(model, lam=1.0, cg_iters=25)
print(f"start: {bits_per_char(model, seq):.3f} bits per character (uniform = {torch.log2(torch.tensor(4.0)):.0f} bits)")
for i in range(6):
    before, after, rho = hf.step(seq, seq)
    print(f"HF step {i + 1}: loss {before:.3f} -> {after:.3f} nats   rho {rho:+.2f}   lambda {hf.lam:.3f}")
print(f"end: {bits_per_char(model, seq):.3f} bits per character")
print("-> each step solves a damped Gauss-Newton system with conjugate gradient; lambda adapts to how")
print("   well the quadratic model predicted the actual change (rho).")
gen = sample(model, [0, 1, 2], 18, temperature=0.5, generator=torch.Generator().manual_seed(0))
print("sample after the prefix 0 1 2:", " ".join(map(str, gen)))
