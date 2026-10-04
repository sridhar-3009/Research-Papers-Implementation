"""ZeRO in ~2 seconds: where the 16 bytes per parameter go, per-GPU memory of the three ZeRO-DP stages (Figure 1,
Table 1), the largest model that fits, and a simulated multi-GPU mixed-precision Adam job in plain DP and the three
ZeRO stages -- identical weights, very different memory, 1.5x communication only for the last stage."""

import time

import numpy as np

from zero import REPORTED, comm_volume, max_params, model_state_bytes, train

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. Model states: 16 bytes per parameter with mixed-precision Adam")
print("  fp16 parameter 2 B + fp16 gradient 2 B + fp32 master parameter 4 B + fp32 momentum 4 B + fp32 variance 4 B")
print("  = 2 + 2 + 12 = 16 bytes (K = 12). A 7.5B-parameter model: 16 x 7.5e9 = 120 GB -- on EVERY data-parallel GPU")

section("2. Per-GPU memory of model states (Figure 1: 7.5B parameters, N_d = 64)")
names = ("plain DP", "P_os (optimizer states)", "P_os+g (+ gradients)", "P_os+g+p (+ parameters)")
formulas = ("(2 + 2 + K) Psi", "2 Psi + 2 Psi + K Psi / N_d", "2 Psi + (2 + K) Psi / N_d", "(2 + 2 + K) Psi / N_d")
for s in range(4):
    print(f"    {names[s]:26s} {formulas[s]:28s} {model_state_bytes(7.5e9, 64, s) / 1e9:6.1f} GB   "
          f"(communication {comm_volume(s):.0f} Psi per step)")
print("  -> the paper's 120 / 31.4 / 16.6 / 1.9 GB: 4x, 8x and 64x less, with the same communication as DP for the")
print("     first two stages and 1.5x for the third")

section("3. A 1-trillion-parameter model (Table 1): per-GPU GB vs data-parallel degree")
print("    N_d        P_os       P_os+g    P_os+g+p     fits a 32 GB V100?")
for nd in (1, 4, 16, 64, 256, 1024):
    v = [model_state_bytes(1e12, nd, s) / 1e9 for s in (1, 2, 3)]
    print(f"    {nd:5d}   {v[0]:9.1f}  {v[1]:9.1f}  {v[2]:9.1f}      {'P_os+g+p' if v[2] < 32 else '-'}")
print("  -> only full partitioning can fit 1T parameters: 16 TB / 1024 GPUs = 15.6 GB each")

section("4. The largest model whose model states fit in a 32 GB GPU")
print("    N_d     plain DP    P_os      P_os+g     P_os+g+p")
for nd in (1, 8, 64, 1024):
    print(f"    {nd:5d}   " + "   ".join(f"{max_params(32e9, nd, s) / 1e9:6.1f}B" for s in range(4)))
print("  -> plain DP is stuck at 2B whatever the cluster size (the paper: ~1.4B once activations are counted);")
print("     ZeRO's limit grows with N_d -- P_os+g+p linearly")

section("5. A simulated data-parallel job: 2,177-parameter MLP, mixed-precision Adam, 50 steps")
for nd in (4, 16):
    res = {s: train(s, nd=nd) for s in range(4)}
    psi = res[0]["psi"]
    print(f"  N_d = {nd} ranks:")
    print("    stage                       model-state bytes per rank   x psi    sent per rank per step   final loss")
    for s in range(4):
        r = res[s]
        print(f"    {names[s]:26s}      {r['persistent bytes']:8,d}              {r['persistent bytes'] / psi:5.2f}       "
              f"{r['sent per step (x psi)']:5.2f} psi             {r['losses'][-1]:.4f}")
    same = all(np.array_equal(res[0]["weights"], res[s]["weights"]) for s in (1, 2, 3))
    print(f"    all four stages end with BIT-IDENTICAL weights: {same}")
print("  -> each stage only changes WHERE the states live; the arithmetic is the same. Bytes per parameter match the")
print("     formulas (16, 4 + 12/N_d, 2 + 14/N_d, 16/N_d), and communication is 2(N_d-1)/N_d psi for DP, P_os, P_os+g")
print("     and 3(N_d-1)/N_d psi for P_os+g+p (two all-gathers of the parameters + one reduce-scatter): 1.5x.")
print("     (Our simulation computes each full gradient at once; real ZeRO frees gradient and parameter buckets layer")
print("     by layer, so its transient memory stays small.)")

section("6. ZeRO-R: the residual memory")
print("  P_a: activation checkpoints are partitioned across the model-parallel ranks instead of replicated -- the paper's")
print(f"  100B model (batch 32, sequence 1024, MP degree 16): ~33 GB per GPU -> ~33 / 16 = {33 / 16:.1f} GB (paper: about 2 GB),")
print("  or ~0 if offloaded to the CPU (P_a+cpu). C_B: constant-size fused buffers (a 3B model's fp32 fused buffer would be")
print("  12 GB). M_D: copy activation checkpoints and gradients into pre-allocated contiguous memory to avoid fragmentation.")

section("7. The paper's numbers")
for k, v in REPORTED.items():
    print(f"  {k}: {v}")
print(f"\nTotal time: {time.time() - T0:.1f}s")
