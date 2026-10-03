"""A ~3-second tour of Chinchilla's compute-optimal scaling.

  1. The parametric loss L(N, D) = E + A/N^alpha + B/D^beta: entropy floor + model-size gap + training gap
  2. The closed-form frontier (Eq. 4) vs Kaplan's recipe; Gopher vs Chinchilla at the same budget
  3. Table 3: compute-optimal tokens for each model size (~20 tokens per parameter)
  4. The three estimation approaches, run on synthetic training runs from a known law (do they recover it?)
  5. Appendix F: counting FLOPs properly vs 6ND
  6. A caveat: how sensitive the frontier is to the published (rounded) exponents
"""

import time

import numpy as np

from chinchilla import (ALPHA, BETA, GOPHER_FLOPS, TABLE_3, A, B, E, approach1_envelope, approach2_isoflop,
                        fit_parametric, flops_appendix_f, frontier, kaplan_N_opt, loss, params_count,
                        synthetic_curves, synthetic_isoflop)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------------------------------- 1
section("1. L(N, D) = 1.69 + 406.4 / N^0.34 + 410.7 / D^0.28   (Eq. 10; nats per token on MassiveText)")
print(f"  {'model':>22} {'entropy E':>9} {'model gap':>9} {'data gap':>9} {'total':>7}")
for name, N, D in (("Gopher 280B, 300B tok", 280e9, 300e9), ("Chinchilla 70B, 1.4T", 70e9, 1.4e12),
                   ("GPT-3 175B, 300B tok", 175e9, 300e9), ("1B model, 20B tokens", 1e9, 20e9)):
    print(f"  {name:>22} {E:9.3f} {A / N ** ALPHA:9.3f} {B / D ** BETA:9.3f} {loss(N, D):7.3f}")
print("  -> E is what a perfect model would still pay (the entropy of text); A/N^alpha is the price of a finite model;")
print("     B/D^beta is the price of finite training data/steps. Gopher spends its budget on the model gap and leaves a big")
print("     data gap; Chinchilla balances them and ends up lower.")

# --------------------------------------------------------------------------------------------- 2
section("2. Compute-optimal size: Chinchilla (Eq. 4) vs Kaplan et al. (2020)")
a, b = BETA / (ALPHA + BETA), ALPHA / (ALPHA + BETA)
print(f"  Eq. 4: N_opt ~ C^{a:.2f}, D_opt ~ C^{b:.2f}  (Approaches 1 and 2 found 0.50 / 0.50 and 0.49 / 0.51;"
      f" Kaplan: 0.73 / 0.27)")
print(f"  {'FLOPs':>9} {'Chinchilla N_opt':>16} {'D_opt':>10} {'tokens/param':>12} {'Kaplan N_opt':>12}")
for C in (1e19, 1e21, 1e23, GOPHER_FLOPS, 1e25):
    N, D = frontier(C)
    print(f"  {C:9.2e} {N:16.2e} {D:10.2e} {D / N:12.1f} {kaplan_N_opt(C):12.2e}")
print(f"  Gopher's budget ({GOPHER_FLOPS:.2e} FLOPs): Kaplan would build ~{kaplan_N_opt(GOPHER_FLOPS) / 1e9:.0f}B "
      f"parameters; Chinchilla's fit says ~{frontier(GOPHER_FLOPS)[0] / 1e9:.0f}B.")
print("  -> as compute grows 10x, Chinchilla grows the model ~3x and the data ~3x; Kaplan grows the model ~5.4x and the")
print("     data ~1.9x. That is why GPT-3 / Gopher-style models were 'considerably over-sized' and under-trained.")

# --------------------------------------------------------------------------------------------- 3
section("3. Table 3 (Approach 1): tokens needed to train each model size compute-optimally")
for N, (flops, tokens) in TABLE_3.items():
    print(f"  {N:8.0e} params -> {tokens:9.2e} tokens ({tokens / N:4.1f} per param), {flops:8.2e} FLOPs "
          f"(6ND = {6 * N * tokens:8.2e})")
print("  -> the famous rule of thumb: about 20 training tokens per parameter. A 175B model would need ~3.7T tokens.")

# --------------------------------------------------------------------------------------------- 4
section("4. The three approaches, applied to synthetic runs generated from the fitted law (true a = 0.452)")
curves = synthetic_curves(list(np.logspace(7.5, 11, 40)), 3e12, n_points=80)
_, a1 = approach1_envelope(curves)
iso = synthetic_isoflop(np.logspace(18, 22, 9), noise=0.002)
_, a2 = approach2_isoflop(iso)
Ns, Ds, Ls = zip(*[(N, C / (6 * N), l) for C, rows in iso.items() for N, l in rows])
fit = fit_parametric(Ns, Ds, Ls, grid=[(5, 5, 0, 0.5, 0.5), (6, 6, 0.5, 0.3, 0.3), (0, 0, -1, 1.0, 1.0)])
print(f"  Approach 1 (lower envelope of 40 training curves):        a = {a1:.3f}")
print(f"  Approach 2 (parabola minima of 9 IsoFLOP profiles):        a = {a2:.3f}   (with 0.2% noise)")
print(f"  Approach 3 (Huber + L-BFGS fit of E, A, B, alpha, beta):    E = {fit[0]:.3f}, alpha = {fit[3]:.3f}, "
      f"beta = {fit[4]:.3f}  -> a = {fit[4] / (fit[3] + fit[4]):.3f}")
print("  -> all three recover the planted answer, the paper's argument that three independent methods agreeing is")
print("     strong evidence. (With only 5 model sizes, Approach 1's envelope is too coarse: it gave a ~ 0.9 in our tests.)")

# --------------------------------------------------------------------------------------------- 5
section("5. Appendix F FLOPs vs the 6ND approximation (sequence length 2048)")
for cfg in (dict(vocab=32000, d_model=640, n_layers=10, ffw=2560, n_heads=10, kq_size=64),
            dict(vocab=32000, d_model=8192, n_layers=80, ffw=32768, n_heads=64, kq_size=128)):
    N = params_count(**cfg)
    r = flops_appendix_f(seq_len=2048, **cfg) / (6 * N * 2048)
    print(f"  {N / 1e6:9.0f}M params: full count / 6ND = {r:.3f}")
print("  -> in our count, attention over a 2048-token context and the embedding/logit layers add ~68% for a 70M model")
print("     (width 640 is small next to the context length), but under 5% at 65B. The paper reports that its full count")
print("     and 6ND differ too little to change its conclusions, which are driven by the large models.")

# --------------------------------------------------------------------------------------------- 6
section("6. Caveat: the frontier is very sensitive to the exponents")
for al, be in ((0.34, 0.28), (0.3392, 0.2849)):
    N, D = frontier(GOPHER_FLOPS, (E, A, B, al, be))
    print(f"  alpha = {al}, beta = {be}: Gopher-budget N_opt = {N / 1e9:5.1f}B, {D / N:5.1f} tokens per parameter")
print("  -> rounding the published exponents moves the optimum from ~40B (the paper's quoted Approach 3 value) to ~32B.")
print("     And Approach 3's ~59-93 tokens per parameter disagrees with Approaches 1-2 (~20); later replication work")
print("     (Besiroglu et al. 2024) traced this to the Approach 3 fit. The paper's headline (a ~ b ~ 0.5) rests mostly on")
print("     Approaches 1-2.")

print(f"\n(total {time.time() - T0:.1f} s)")
