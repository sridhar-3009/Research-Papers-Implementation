"""A ~2-second tour of the scaling laws (all numbers from the paper's fitted constants).

  1. The three power laws: what a 10x bigger model / dataset / compute budget buys
  2. Combining N and D (Eq. 1.5): when does a model start to overfit? (Eq. 4.4)
  3. The critical batch size (Eq. 1.4) and the steps-vs-compute trade-off (Eq. 5.1)
  4. Spending a compute budget (Section 6): solve for the best model size and compare with Table 5
  5. Why power laws at all? A toy where 'patterns' have power-law importance
"""

import time

import numpy as np

from scaling import (A_CMIN, A_D, A_N, NS, B_crit, L_of_Cmin, L_of_D, L_of_N, L_of_ND, L_of_NS,
                     allocation_exponents, compute_optimal_N, fit_power_law, overfit_penalty, spectrum_loss_vs_D,
                     spectrum_loss_vs_N, spectrum_toy, table5, tokens_to_avoid_overfitting, training_compute)

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------------------------------- 1
section("1. The three power laws (loss in nats per token on WebText2)")
print(f"  {'N (non-embedding)':>18} {'L(N)':>6}     {'D (tokens)':>11} {'L(D)':>6}     {'C_min (PF-days)':>15} {'L(C)':>6}")
for k in range(6):
    N, D, C = 10.0 ** (5 + k), 10.0 ** (7 + k), 10.0 ** (-6 + 1.5 * k)
    print(f"  {N:18.0e} {L_of_N(N):6.3f}     {D:11.0e} {L_of_D(D):6.3f}     {C:15.1e} {L_of_Cmin(C):6.3f}")
print(f"  -> 10x more parameters multiplies the loss by 10^-{A_N} = {10 ** -A_N:.3f}; 10x data by {10 ** -A_D:.3f};")
print(f"     10x compute by {10 ** -A_CMIN:.3f}. Small, steady, PREDICTABLE gains over 6-8 orders of magnitude, and they")
print("     depend very weakly on the shape (depth vs width, heads) once N is fixed.")

# --------------------------------------------------------------------------------------------- 2
section("2. Model size and data together (Eq. 1.5): the overfitting penalty L(N, D) / L(N, inf) - 1")
print(f"  {'N':>8} {'D = 1e8':>9} {'D = 1e9':>9} {'D = 1e10':>9} {'D = 1e11':>9}   tokens needed (5e3 N^0.74)")
for N in (1e6, 1e8, 1e9, 1e10):
    pens = " ".join(f"{100 * overfit_penalty(N, D):8.2f}%" for D in (1e8, 1e9, 1e10, 1e11))
    print(f"  {N:8.0e} {pens}   {tokens_to_avoid_overfitting(N):.1e}")
print("  -> required data grows SUB-linearly in model size (D ~ N^0.74): an 8x bigger model needs only ~5x more data")
print("     to stay within the ~2% seed noise. (Chinchilla, paper 052, will revise how much data is OPTIMAL.)")

# --------------------------------------------------------------------------------------------- 3
section("3. Critical batch size (Eq. 1.4): grows as the loss falls")
for L in (5.0, 4.0, 3.0, 2.5, 2.0):
    print(f"  loss {L:.1f} nats: B_crit = {B_crit(L):.2e} tokens")
print("  -> B_crit roughly doubles for every 13% drop in loss. Training at B = B_crit costs 2x the minimum steps")
print("     AND 2x the minimum compute (Eq. 5.1: (S/S_min - 1)(E/E_min - 1) = 1): the balanced choice.")

# --------------------------------------------------------------------------------------------- 4
section("4. The compute-efficient frontier: best model size for each budget (from L(N, S) + B_crit)")
e = allocation_exponents()
print(f"  Eq. 1.8: a_C = 1/(1/a_S + 1/a_B + 1/a_N) = {e['a_C_min']:.3f} (fitted directly: 0.050); "
      f"N ~ C^{e['N']:.2f}, B ~ C^{e['B']:.2f}, S ~ C^{e['S']:.2f}")
Cs = np.logspace(-4, 2, 7)
print(f"  {'C (PF-days)':>11} {'our N_opt':>10} {'Table 5 N_opt':>13} {'loss':>6} {'converged L(N)':>14} {'ratio':>6}")
Ns = []
for C in Cs:
    N, L = compute_optimal_N(C)
    Ns.append(N)
    Linf = L_of_NS(N, 1e30)
    print(f"  {C:11.0e} {N:10.2e} {table5(C)['N_opt']:13.2e} {L:6.3f} {Linf:14.3f} {L / Linf:6.3f}")
slope = np.polyfit(np.log(Cs), np.log(Ns), 1)[0]
print(f"  -> fitted N_opt ~ C^{slope:.2f} (theory {e['N']:.2f}, paper's empirical 0.73), within ~2x of Table 5.")
print(f"     At the optimum the loss sits {100 * (L / Linf - 1):.0f}% above where the same model would converge "
      f"(theory: a_N/a_S = {100 * NS['a_N'] / NS['a_S']:.0f}%):")
print("     compute-efficient training STOPS EARLY and spends the budget on a bigger model instead.")
gpt3 = training_compute(175e9, 300e9) / 8.64e19
print(f"  GPT-3 175B used ~{gpt3:.0f} PF-days; for that budget Table 5's recipe gives N_opt = {table5(gpt3)['N_opt']:.1e}"
      f" and D_opt = {table5(gpt3)['D_opt']:.1e} tokens.")
print("     GPT-3 (1.75e11 parameters, 3e11 tokens) is in the same spirit: very large model, relatively little data.")
print("     Paper 052 (Chinchilla) argues this recipe under-weights data.")

# --------------------------------------------------------------------------------------------- 5
section("5. Why power laws? (a standard explanation, not from this paper)")
lam, w = spectrum_toy(n_features=4000, alpha=1.5)
Nsz = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
Ds = np.array([16, 32, 64, 128, 256, 512, 1024])
aN, _ = fit_power_law(Nsz, spectrum_loss_vs_N(lam, w, Nsz))
aD, _ = fit_power_law(Ds, spectrum_loss_vs_D(lam, w, Ds))
print("  data = 4,000 'patterns' whose importance falls as k^-1.5 (a few common, many rare);")
print(f"  a model that can hold N patterns:   loss ~ N^-{aN:.2f}   (theory alpha - 1 = 0.50)")
print(f"  a learner that sees D examples:     loss ~ D^-{aD:.2f}")
print("  -> if the world is made of ever-rarer patterns, each extra parameter or example captures the next one, and the")
print("     error that remains is the tail of a power law. Language's long tail of rare words and facts fits this picture.")

print(f"\n(total {time.time() - T0:.1f} s)")
