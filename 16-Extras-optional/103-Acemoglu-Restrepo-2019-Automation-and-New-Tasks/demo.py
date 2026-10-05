"""Automation and new tasks in ~1 second: the task model (Eq. 1-2) with sigma = 0.8 -- displacement vs productivity
effects of automation, 'so-so' automation, reinstatement from new tasks, factor-augmenting technology -- and the
paper's wage-bill decomposition, validated on synthetic industries with planted automation and new-task shocks."""

import time

import numpy as np

import tasks as T

T0 = time.time()


def section(t):
    print(f"\n=== {t} ===")


section("1. The task model (tasks on [N - 1, N]; capital does z <= I; sigma = 0.8; K = 3, L = 1)")
e = T.equilibrium(**T.BASE, K=3.0)
print(f"  baseline I = 0.4, N = 1: task content Gamma = {e['Gamma']:.3f}, labour share {e['labor share']:.3f}, "
      f"wage {e['W']:.3f}, rental rate {e['R']:.3f}")
print("    change (step 0.02)        productivity d ln Y   labour share d ln s_L   wage bill d ln WL")
for ch in ("automation", "new tasks", "labour-augmenting"):
    r = T.effects(ch)
    print(f"    {ch:24s}    {r['productivity effect (d ln Y)']:+.4f}             {r['change in labour share (d ln s_L)']:+.4f}"
          f"                {r['wage bill (d ln W L)']:+.4f}")
print("  -> automation: positive productivity effect but a larger DISPLACEMENT effect -> labour share and wage bill fall;")
print("     new tasks: REINSTATEMENT raises the labour share and the wage bill; labour-augmenting technology works almost")
print("     only through productivity (labour share -0.0017 with sigma just below 1)")
print("  Honest note: here new tasks lower output slightly (the window [N - 1, N] drops a cheap capital task); the")
print("  paper's 'new tasks raise productivity' needs the new labour task to be cheaper than the task it replaces")

section("2. 'So-so' automation: the same automation step with capital of different productivity A_K")
print("    A_K    unit cost at task I: labour / capital    productivity    labour share    wage")
for AK, r in T.so_so_automation().items():
    print(f"    {AK:3.1f}        {r['labour cost']:.3f} / {r['capital cost']:.3f}                 {r['productivity']:+.4f}         "
          f"{r['labour share']:+.4f}       {r['wage']:+.4f}")
print("  -> the labour share falls whatever A_K is (displacement); whether WAGES fall depends on the productivity effect:")
print("     barely-better ('so-so') capital lowers wages, much better capital raises them")

section("3. Labour share response to automation / new tasks for different sigma")
for s in (0.5, 0.8, 1.0, 1.5):
    print(f"    sigma = {s}: automation d ln s_L {T.effects('automation', sigma=s)['change in labour share (d ln s_L)']:+.4f}, "
          f"new tasks {T.effects('new tasks', sigma=s)['change in labour share (d ln s_L)']:+.4f}")
print("  -> signs do not depend on sigma: 'automation always reduces the labour share, new tasks always raise it'")

section("4. The wage-bill decomposition on 20 synthetic industries over 30 years (planted shocks)")
for both in (0.0, 0.3):
    d = T.synthetic_industries(both_prob=both)
    dec = T.decompose(d)
    pd, pr = T.planted_split(d)
    pt = T.planted_task_content(d)
    m = {k: v.mean() * 100 for k, v in dec.items()}
    label = "each industry-year gets one kind of shock" if both == 0 else "30% of industry-years get BOTH shocks"
    print(f"  {label}:")
    print(f"    wage bill {m['wage bill']:+.2f}%/yr = productivity {m['productivity']:+.2f} + composition "
          f"{m['composition']:+.2f} + substitution {m['substitution']:+.2f} + task content {m['task content']:+.2f}")
    print(f"    task content estimated {m['task content']:+.3f} vs planted {pt.mean() * 100:+.3f}; displacement estimated "
          f"{m['displacement']:+.3f} vs planted {pd.mean() * 100:+.3f}; reinstatement {m['reinstatement']:+.3f} vs "
          f"{pr.mean() * 100:+.3f}")
d = T.synthetic_industries()
wrong = T.decompose(d, sigma_assumed=1.0)["task content"].mean() * 100
print(f"  with a mis-specified sigma = 1 (truth 0.8) the task-content estimate becomes {wrong:+.3f}")
print("  -> the net change in task content is recovered (first-order accounting, ~0.05 points off); the moving-average")
print("     displacement / reinstatement split UNDERSTATES both -- the paper calls its estimates lower bounds -- and more")
print("     so when industries automate and add tasks at the same time")

section("What the paper reports (verified against the PDF; IZA working-paper version)")
for k, v in T.REPORTED.items():
    print(f"  {k}: {v}")
print(f"\n(done in {time.time() - T0:.1f} s)")
