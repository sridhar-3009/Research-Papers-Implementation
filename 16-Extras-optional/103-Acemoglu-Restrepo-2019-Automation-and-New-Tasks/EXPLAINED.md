# Automation and new tasks, explained simply

**Paper:** Daron Acemoglu (MIT) and Pascual Restrepo (Boston University), *Automation and New Tasks: How Technology Displaces and Reinstates Labor*, Journal of Economic Perspectives 33(2): 3–30, 2019. (We read the IZA discussion-paper version, DP No. 12293, April 2019.)

**In one sentence:** technology doesn't just make workers more productive. Some technologies (**automation**) take tasks away from workers and give them to machines, which **always lowers labor's share of income**. Others (**new tasks**) create new work that people are better at, which **always raises it**. US labor demand grew slowly after 1987 because automation sped up while new-task creation slowed.

---

## 1. Why tasks, not just "capital and labor"

- **The standard economics model** says technology makes labor or capital more productive ("factor-augmenting" technology). That predicts wages rise in step with productivity and a fairly stable labor share. Recent decades don't look like that.
- **The task view:**
  - production is a list of **tasks** (say, ordered from z = N − 1 to N), and each task is done by either a machine or a person;
  - **automation** moves the boundary I up: tasks z ≤ I are now done by capital;
  - **new tasks** move N up: new, labor-intensive activities appear (e.g. software engineers, technicians, analysts).
- **Who does what:** workers have a **comparative advantage** in higher-numbered tasks. That is why machines take the lower ones.

---

## 2. The model in two equations

With a CES way of combining tasks (elasticity of substitution σ), output can be written as (Eq. 1):
```
Y = Π(I, N) · [ Γ^(1/σ) (A_L L)^((σ−1)/σ)  +  (1 − Γ)^(1/σ) (A_K K)^((σ−1)/σ) ]^(σ/(σ−1))
```
- **Γ(I, N)** is the **task content of production:** labor's (productivity-adjusted) share of tasks. When σ = 1 it is simply N − I, the fraction of tasks done by labor.
- **A_L, A_K** are factor-augmenting technologies.
- **Π(I, N)** is the productivity gain from assigning tasks to the cheaper factor.

The **labor share** (wage bill / value added) is (Eq. 2):
```
s_L = 1 / ( 1 + ((1 − Γ)/Γ) · (A_L R / (A_K W))^(1 − σ) )
```

**Worked example (σ = 1):**
- tasks from 0 to 1, with I = 0.4: labor does 60% of tasks, so Γ = 0.6 and **s_L = 0.6**;
- automating 10% more (I = 0.5): Γ = 0.5, so **s_L falls to 0.5**;
- creating new tasks that shift the range to [0.1, 1.1] with I = 0.4: Γ = 1.1 − 0.4 = 0.7 of a unit range, so **s_L rises to 0.7**.

---

## 3. The effects

```
Automation's effect on labor demand  = productivity effect + DISPLACEMENT effect      (displacement < 0 always)
New tasks' effect on labor demand    = productivity effect + REINSTATEMENT effect     (reinstatement > 0 always)
Factor-augmenting technology         = productivity effect + substitution effect (small when σ ≈ 1)
```

- **Automation can raise or lower wages:** it depends on whether its productivity gain outweighs the displacement.
- **"So-so technologies"** (e.g. automated customer service, self-checkout) replace workers with machines that are only slightly cheaper. Their productivity gain is small, so they mainly displace.
- **Two lessons:**
  1. Automation by itself does **not** raise wages in step with productivity.
  2. Stable labor shares in the past must have come from **new tasks** offsetting automation.

**History:**
- from about 1850 to 1870, mechanisation cut agriculture's labor share from **33% to 17%**;
- meanwhile industry's labor share rose from **47% (1850) to 55% (1890)**, as new factory and clerical tasks reinstated labor, and activity shifted toward the more labor-intensive industrial sector (a **composition** effect).

---

## 4. Measuring it: the wage-bill decomposition

The paper splits the growth of the US wage bill per person into:
```
Δ wage bill = productivity effect + composition effect + substitution effect + change in task content
```

| Piece | How it is measured |
|---|---|
| **Productivity** | growth of GDP per capita |
| **Composition** | Σ (industry labor share × change in its value-added share): activity moving between more and less labor-intensive industries |
| **Substitution** | from Eq. 2 with σ = 0.8: how much each industry's labor share should move given its factor prices, with A_L/A_K growing like labor productivity (2%/yr 1947–87, 1.46%/yr 1987–2017) |
| **Task content** | the **residual**: % change in an industry's labor share minus its substitution effect |

- **Splitting task content into displacement and reinstatement:**
  - industries whose task content falls (over 5-year moving averages) count as **displacement**;
  - those whose task content rises count as **reinstatement**.
- **Lower bounds:** if an industry automates and adds tasks in the same window, the two partly cancel, so both estimates are **lower bounds**.

**First-order formula used in our code** (from differentiating Eq. 2):
```
d ln s_L = (1 − s_L) · d ln(Γ / (1 − Γ))   +   (1 − s_L)(1 − σ) · d ln( (W/A_L) / (R/A_K) )
           └──── task content ────┘            └──────────── substitution ────────────┘
```

**Example:**
- an industry with s_L = 0.6 and σ = 0.8, where the effective price of labor rose 2% relative to capital;
- substitution effect = 0.4 · 0.2 · 0.02 = **+0.16%**;
- if its labor share actually fell 1%, the task content changed by −1% − 0.16% = **−1.16%**, which counts as displacement.

---

## 5. The paper's findings for the US

| Period | Wage bill per capita | Productivity effect | Displacement | Reinstatement | Net task content |
|---|---|---|---|---|---|
| 1947–1987 | **2.5%** a year | 2.4% | −0.48% | +0.47% | ≈ 0 |
| 1987–2017 | **1.33%** a year | 1.54% | **−0.70%** | **+0.35%** | **−0.35%** a year (≈ −10% of labor demand cumulatively) |

- **Manufacturing, 1987–2017:** displacement of about 1.1% a year, roughly **30% cumulatively**.
- **Their task-content estimates correlate with independent measures** of automation (robots, automation technologies) and of new tasks (new job titles, new occupations).
- **Interpretation:** slower wage growth after 1987 reflects **faster automation, weaker reinstatement**, and slower productivity growth.

---

## 6. What our code found

**The task model** (σ = 0.8, γ_L(z) = e^z, γ_K = 1, K = 3, L = 1; baseline I = 0.4, N = 1, so Γ = 0.566 and labor share 0.647). Each change is a step of 0.02:

| Change | Productivity d ln Y | Labor share d ln s_L | Wage bill d ln WL |
|---|---|---|---|
| automation (I ↑) | +0.0251 | **−0.0386** | −0.0135 |
| new tasks (N ↑) | −0.0141 | **+0.0350** | +0.0209 |
| labor-augmenting (A_L ↑ 2%) | +0.0128 | −0.0017 | +0.0111 |

**So-so automation** (the same automation step, with capital of different productivity):

| A_K | Unit cost at task I (labor / capital) | Productivity | Labor share | Wage |
|---|---|---|---|---|
| 0.5 | 1.424 / 0.918 | +0.0077 | −0.0430 | **−0.0353** |
| 1 | 1.967 / 0.533 | +0.0251 | −0.0386 | −0.0135 |
| 2 | 2.625 / 0.299 | +0.0417 | −0.0345 | **+0.0072** |
| 8 | 4.254 / 0.086 | +0.0718 | −0.0270 | **+0.0448** |

- **The labor share falls for every A_K** (displacement). **Wages fall** when the new capital is only modestly better ("so-so") and **rise** when it is much better. The crossover is around A_K ≈ 2 (E1 smoke run).
- **For every σ in {0.5, 0.8, 1.0, 1.5},** automation lowers and new tasks raise the labor share, as the paper states, "independently from σ".
- **Honest note:** in our parametrisation new tasks slightly *lower* output. Shifting the task window drops the cheapest capital task. Wages and the labor share still rise through reinstatement. The paper's "new tasks raise productivity" holds when the new labor task is cheaper than the task it replaces.

**The decomposition on synthetic industries:**
- **Setup:** 20 industries over 30 years, with planted automation and new-task shocks, drifting factor prices, and value-added reallocation.

| | One kind of shock per industry-year | 30% of industry-years with both |
|---|---|---|
| Wage-bill growth | +1.05%/yr = 1.50 + (−0.01) + 0.08 + (−0.56) | +0.87%/yr |
| Task content: estimated vs planted | −0.559 vs −0.511 | −0.746 vs −0.684 |
| Displacement: estimated vs planted | −0.80 vs −1.15 | −1.00 vs −1.81 |
| Reinstatement: estimated vs planted | +0.25 vs +0.63 | +0.24 vs +1.11 |

- **The net change in task content is recovered** to within about 0.05 points a year (the accounting is first-order).
- **The moving-average split understates both displacement and reinstatement,** and more so when industries automate and add tasks simultaneously. In the E4 smoke run the estimates were 0.69 / 0.39 of the planted values, falling to 0.55 / 0.22 with simultaneous shocks. This is exactly why the paper calls its estimates **lower bounds**.
- **Assuming σ = 1** when the truth is 0.8 changes the task-content estimate from −0.559 to −0.479.

**`experiments.py`:**
- **E1:** the so-so boundary across σ;
- **E2:** accuracy vs shock size;
- **E3:** sensitivity to the assumed σ;
- **E4:** window and simultaneous-shock bias.
- Only E1, E3 and E4 smoke runs were done here.

---

## 7. Why it matters (and the link to AI)

- **The question for AI:** will it mostly **automate** existing tasks (displacement, with a so-so risk), or **create new tasks** for people (reinstatement)?
- **The framework's answer:** the labor-market outcome depends on that **direction of technology**, not just on how much productivity AI adds.
- **Acemoglu and Restrepo argue** that research and policy can steer technology toward new tasks.
- **This task framework is now the standard way economists analyse AI and jobs,** e.g. task-exposure measures for large language models.

---

## 8. Check yourself

1. What is the "task content of production"?
<details><summary>Answer</summary>Γ(I, N): labor's (productivity-adjusted) share of the tasks used in production. With σ = 1 it is simply N − I, the fraction of tasks done by labor.</details>

2. Why does automation always lower the labor share in this model?
<details><summary>Answer</summary>It hands tasks from labor to capital, lowering Γ. By Eq. 2 the labor share rises with Γ for any σ, so it falls. This is the displacement effect.</details>

3. Can automation still raise wages? When?
<details><summary>Answer</summary>Yes, if its productivity effect is larger than the displacement effect, i.e. when machines are much cheaper than the workers they replace. "So-so" automation (machines barely cheaper) lowers wages.</details>

4. With σ = 1, tasks on [0, 1] and I = 0.3, what is the labor share? What if I rises to 0.45?
<details><summary>Answer</summary>s_L = N − I = 0.7, then 0.55.</details>

5. Compute the substitution effect for an industry with s_L = 0.5, σ = 0.8, and a 5% rise in the effective price of labor relative to capital.
<details><summary>Answer</summary>(1 − 0.5)(1 − 0.8)(0.05) = 0.005, a 0.5% rise in the labor share. With σ < 1, pricier labor raises labor's share.</details>

6. How is the change in task content measured in the data?
<details><summary>Answer</summary>As the residual: the observed % change in an industry's labor share minus the substitution effect predicted from factor prices and σ.</details>

7. Why are the displacement and reinstatement estimates lower bounds?
<details><summary>Answer</summary>They come from the sign of each industry's net task-content change over 5-year windows. If an industry automates and adds new tasks in the same window, the two partly cancel and only the net shows up. Our planted test shows estimates at 69% (displacement) and 39% (reinstatement) of the truth, and lower with simultaneous shocks.</details>

8. What changed between 1947–1987 and 1987–2017 in the US?
<details><summary>Answer</summary>Displacement accelerated (−0.48% → −0.70% a year), reinstatement slowed (+0.47% → +0.35%), and productivity growth slowed (2.4% → 1.54%), so wage-bill growth fell from 2.5% to 1.33% a year.</details>

9. What is the composition effect, and how did it matter in the 19th century?
<details><summary>Answer</summary>The change in labor demand from activity shifting between sectors with different labor shares. As agriculture mechanised, activity moved to the more labor-intensive industrial sector, which (with new factory tasks) offset agriculture's falling labor share.</details>

10. Why do factor-augmenting technologies have small effects on the labor share?
<details><summary>Answer</summary>Estimates put σ close to (just below) 1. Their only effect on the labor share is the substitution effect, proportional to (1 − σ), so it is small compared with their productivity effect (−0.0017 vs +0.0128 in our model).</details>
