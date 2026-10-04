"""GPTQ in ~25 seconds: the quantization grid and round-to-nearest, one error-compensating update worked by hand, RTN vs
greedy OBQ vs GPTQ on a layer with correlated inputs (and the check that GPTQ's fast Algorithm 1 equals the literal
column-by-column update), the speed-up over OBQ, and 4 / 3 / 2-bit quantization of a tiny LLaMA (Table 3)."""

import time

import numpy as np
import torch

from gptq import (REPORTED, L, calibration_batch, gptq, gptq_naive, grid, hessian, layer_error, model_loss, obq, quant,
                  quantize_model, rtn, task_accuracy)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. A 2-bit grid and round-to-nearest (RTN)")
w = torch.tensor([[0.9, -0.4, 0.1, 0.35]])
scale, zero = grid(w, 2)
print(f"  w = [0.9, -0.4, 0.1, 0.35]: min-max grid over [-0.4, 0.9] with 2^2 = 4 levels: scale {scale.item():.4f}, zero {zero.item():.0f}")
print(f"  levels {[round(scale.item() * (k - zero.item()), 3) for k in range(4)]}; RTN gives {[round(v, 3) for v in quant(w, scale, zero, 2).tolist()[0]]}")

section("2. One OBQ / GPTQ step by hand: quantize w1, let w2 absorb the error")
X = torch.tensor([[1.0, 0.9, 1.1, 0.95], [1.0, 1.1, 0.9, 1.05]], dtype=torch.float64)    # two strongly correlated inputs
W = torch.tensor([[0.3, 0.3]], dtype=torch.float64)
H = 2 * X @ X.T
Hinv = torch.linalg.inv(H)
q1 = 0.5                                                                                  # w1 rounds to 0.5 (grid {0, 0.5})
err = (W[0, 0] - q1) / Hinv[0, 0]
w2_new = W[0, 1] - err * Hinv[0, 1]
print(f"  inputs x1, x2 almost equal; w = [0.3, 0.3]; grid {{0, 0.5}}; H = 2XX^T, H^-1 = {np.round(Hinv.numpy(), 2).tolist()}")
print(f"  w1 -> 0.5 (error -0.2). Eq. 2: w2 -= (w1 - q1)/[H^-1]_11 * [H^-1]_12 = 0.3 - ({err.item():.3f}) * ({Hinv[0, 1].item():.2f}) "
      f"= {w2_new.item():.3f}, which rounds to 0.0")
for name, Q in (("RTN [0.5, 0.5]", torch.tensor([[0.5, 0.5]])), (f"compensated [0.5, 0.0]", torch.tensor([[0.5, 0.0]]))):
    print(f"    {name:24s} layer error ||WX - QX||^2 = {layer_error(W, Q.double(), X):.3f}")
print("  -> because x1 ~ x2, pushing w1 up and w2 down keeps w1 x1 + w2 x2 almost unchanged (0.6 vs 0.5 instead of 1.0)")

section("3. One layer with correlated inputs: RTN vs greedy OBQ vs GPTQ")
torch.manual_seed(0)
d_row, d_col, n = 32, 48, 512
X = ((torch.randn(d_col, d_col) @ torch.randn(d_col, n)) / 5).double()
W = torch.randn(d_row, d_col).double()
H = hessian(X)
print("    bits     RTN error     OBQ error (greedy order)    GPTQ error (fixed order)   GPTQ / RTN")
for b in (4, 3, 2):
    e = {k: layer_error(W, f().double(), X) for k, f in (("rtn", lambda: rtn(W, b)), ("obq", lambda: obq(W, H, b)),
                                                         ("gptq", lambda: gptq(W, H, b, blocksize=16)))}
    print(f"    {b}     {e['rtn']:10.0f}     {e['obq']:10.0f}                  {e['gptq']:10.0f}               {e['gptq'] / e['rtn']:.2f}")
print("  -> error compensation roughly halves RTN's error; quantizing in a FIXED column order (GPTQ) is nearly as good")
print("     as OBQ's greedy order, the paper's 'arbitrary order insight'.")
d = (gptq_naive(W, H, 3) - gptq(W, H, 3, blocksize=16)).abs().max().item()
print(f"  Algorithm 1 (Cholesky + lazy blocks of 16) vs the literal column-by-column update: max |difference| = {d:.1e}")

section("4. Speed: OBQ is O(d_row d_col^3), GPTQ O(max(d_row d_col^2, d_col^3))")
Wb, Xb = torch.randn(64, 128).double(), torch.randn(128, 1024).double()
Hb = hessian(Xb)
times = {}
for name, f in (("OBQ (greedy, per row)", lambda: obq(Wb, Hb, 4)), ("GPTQ (Algorithm 1, B = 128)", lambda: gptq(Wb, Hb, 4))):
    t = time.time()
    f()
    times[name] = time.time() - t
    print(f"    64 x 128 layer, {name:28s} {1000 * times[name]:8.1f} ms")
print(f"  -> {times['OBQ (greedy, per row)'] / times['GPTQ (Algorithm 1, B = 128)']:.0f}x faster on this small layer; the gap grows with d_row and d_col")
print("  (the paper: a 175B model in ~4 GPU hours, >1000x faster than OBQ at that scale)")

section("5. Quantizing a whole tiny LLaMA (Table 3)")
t = time.time()
base = L.pretrain(L.new_model(), np.random.default_rng(0))
calib = calibration_batch()
print(f"  paper 073's tiny LLaMA pre-trained on copy / reverse / sort ({time.time() - t:.0f}s); all 14 block linear layers are")
print("  quantized in forward order, each with a Hessian from 126 calibration sequences passed through the already-")
print("  quantized earlier layers; embeddings and output head stay in floating point")
print(f"    full precision:                  accuracy {task_accuracy(base):6.1%}   loss {model_loss(base):.4f}")
res = {}
for bits in (4, 3, 2):
    for gs in (None, 16):
        r, g = quantize_model(base, calib, bits, "rtn", gs), quantize_model(base, calib, bits, "gptq", gs)
        res[(bits, gs)] = (task_accuracy(r), model_loss(r), task_accuracy(g), model_loss(g))
        tag = f"{bits}-bit" + (", groups of 16" if gs else ", per row     ")
        print(f"    {tag:24s}  RTN accuracy {res[(bits, gs)][0]:6.1%} loss {res[(bits, gs)][1]:.4f}   |   GPTQ accuracy "
              f"{res[(bits, gs)][2]:6.1%} loss {res[(bits, gs)][3]:.4f}")
r2 = res[(2, None)]
print(f"  -> at 4 bits even RTN is fine on this small model; the gap opens as bits drop: at 2 bits per row RTN falls to")
print(f"     {r2[0]:.1%} (loss x{r2[1] / model_loss(base):.0f}) while GPTQ keeps {r2[2]:.1%}. Grouping (one grid per 16 columns) helps RTN most, as in the")
print("     paper (g128 / g1024). The paper: OPT-175B 4-bit RTN 10.54 vs GPTQ 8.37 perplexity (FP16 8.34).")

section("6. Memory")
for bits in (16, 4, 3):
    print(f"    175B parameters at {bits:2d} bits: {175e9 * bits / 8 / 2**30:6.0f} GiB")
print("  -> at 3 bits OPT-175B fits on a single 80 GB A100 (the paper's headline deployment result)")

section("7. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
