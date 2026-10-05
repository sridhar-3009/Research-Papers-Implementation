"""The Malicious Use of AI, made quantitative from the defender's side in ~5 seconds: (1) an abstract cost-benefit model
of 'expansion of existing threats' -- what happens when automation makes tailoring cheap; (2) the AI-specific
vulnerabilities the report names (adversarial examples, data poisoning) tested against a digit classifier with standard
defences, as a small red-team report."""

import time

import numpy as np

import security as S

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. 'Alleviating the trade-off between scale and efficacy' (abstract cost-benefit model)")
print("  10,000 targets of lognormal value; per target the attacker picks the better of a generic attempt (cost 0.01,")
print("  success 0.2%) and a tailored one (success 5%, cost below), or nothing. Automation lowers the tailoring cost.")
budgets = np.random.default_rng(1).lognormal(0, 1.5, 1000)
print("    tailoring cost   targets attacked   given a tailored attempt   expected harm   actors able to afford it")
for c in (20, 5, 1, 0.2, 0.05):
    r = S.attack_economics(c)
    print(f"    {c:>12}       {r['attacked']:6.1%}             {r['tailored']:6.1%}                 {r['expected harm']:8.0f}"
          f"          {S.capable_actors(c, budgets):6.1%}")
print("  -> as tailoring gets 400x cheaper, tailored attempts go from 0.3% to 91% of targets, expected harm rises ~9x,")
print("     and the share of actors who can afford them goes from 2% to 97% -- the report's 'expansion of the set of")
print("     actors, the rate of attacks and the set of targets'")
for new in (1, 0.2):
    f = S.defence_needed(new, 20)
    print(f"  to bring harm back to the cost-20 level when tailoring costs {new}, defences must cut success rates to "
          f"{f:.0%} of their old value (a {1 - f:.0%} reduction)")

section("2. Vulnerabilities of AI systems: a red-team report for a digit classifier (softmax regression, 1,300 train images)")
Xtr, ytr, Xte, yte = S.load_digits()
report = S.red_team_report(Xtr, ytr, Xte, yte)
for test, res in report.items():
    print(f"    {test:40s} " + "   ".join(f"{k}: {v:.3f}" for k, v in res.items()))
print("  -> adversarial examples: perturbing every pixel by 0.2 (on a 0-1 scale) drops accuracy from 0.95 to 0.24;")
print("     adversarial training helps only a little (0.32) -- robustness is not free")
print("  -> random label noise barely hurts this regularised linear model, and loss-based sanitisation recovers the")
print("     small loss; but a TARGETED poisoning (every training 7 labelled 1) makes every test 7 a '1' while overall")
print("     accuracy still looks decent (0.86), and loss-based sanitisation does NOT catch it: the poisoned labels are")
print("     consistent, so they have low loss -- the kind of novel failure mode the report warns about")

section("The report's framework (verified against the PDF)")
for k, v in S.REPORTED.items():
    print(f"  {k}: {v}")
print(f"\n(done in {time.time() - T0:.1f} s)")
