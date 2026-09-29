# The code, explained simply

How the code in this folder implements Du et al. (2022).
Read [EXPLAINED.md](EXPLAINED.md) first for the paper itself.

---

## 1. The files at a glance

| File | What it is | Paper part |
|---|---|---|
| `perceptron.py` | Single-neuron learning: perceptron rule, pocket, LMS | Section 3 |
| `mlp.py` | The multilayer network: forward pass, **backpropagation**, Jacobian | Section 4, 5.3, 10 |
| `optimizers.py` | **11 training algorithms**, early stopping, fault injection | Sections 4, 5, 7, 8, 10 |
| `experiment_iris.py` | Reproduces the paper's **Iris experiment** (Table 1, Figure 4) | Section 12.1 |
| `demo.py` | One small experiment per section | Sections 3, 4, 5, 10 |
| `test_survey.py` | **40 tests**, each checks an equation or claim | Everything |
| `iris.csv` | The Iris dataset (150 flowers) | Section 12.1 |
| `results.md`, `figure4_learning_curves.png` | Output of `experiment_iris.py` | Table 1, Figure 4 |

**How they depend on each other:**
```
perceptron.py        mlp.py
                        ↑
                  optimizers.py
                        ↑
                experiment_iris.py
                        ↑
            demo.py, test_survey.py
```

**Run it** (from the `01-Foundations/005-Du-et-al-2022` folder):
```
python3 demo.py                    # the tour (~5 s)
python3 experiment_iris.py         # Table 1 + Figure 4 (~20 s)
python3 -m pytest -q               # all 40 tests (~10 s)
```
Needs `numpy`; the figure needs `matplotlib`; the tests need `pytest`.

**Suggested reading order:**
1. `perceptron.py`
2. `mlp.py` (`forward`, then `gradient`)
3. `optimizers.py` (`gd` → `gdm` → `rprop` → `lm`)
4. `experiment_iris.py`

---

## 2. `perceptron.py`: learning with one neuron

### 2.1 `perceptron_train`: the first learning rule (Eqs. 15–18)

```python
net = X[t] @ w + theta          # weighted sum
o = int(net > 0)                # 0 or 1
e = y[t] - o                    # error: -1, 0 or +1
if e:
    w = w + lr * X[t] * e       # only change when WRONG
    theta = theta + lr * e
```

**Why it works, in one example:** the answer should be 1, but the neuron said 0, so `e = +1`. Adding `x` to `w` makes `w·x` bigger next time, which pushes the neuron toward saying 1.

It stops after one full pass (**epoch**) with no mistakes. On XOR that never happens, so `max_epochs` stops it and it returns `converged=False`.

**Compare with Paper 1:** there, **you** chose the connection counts. Here, the rule **finds** the weights from examples.

### 2.2 `pocket_train`: best effort on impossible data
Runs the perceptron rule on random examples, but keeps a copy of the **best weights so far** (the "pocket"):
- `run` counts how many examples in a row the current weights got right.
- If that run is the longest yet **and** the weights score better on the whole dataset (the "ratchet"), they go in the pocket.

On XOR it returns 75%, the best any single line can do.

### 2.3 `lms_train`: Adaline (Eqs. 19, 18, 21)
```python
e = y[t] - net          # error uses the RAW sum, not 0/1
```
That one change turns the rule into **gradient descent on the squared error**, so it settles on the least-squares line even when classes overlap. `normalized=True` divides the step by `‖x‖² + 1` (the α-LMS rule, Eq. 21).

---

## 3. `mlp.py`: the multilayer network

### 3.1 Activations
```python
ACTIVATIONS = {"logistic": (function, derivative), "tanh": ..., "linear": ..., "relu": ...}
```
Each derivative is written using the **output** `o`. For the logistic function, the slope is `o·(1−o)`, so backprop can reuse the outputs it already computed.

The logistic function is coded as `0.5·(1 + tanh(a/2))`. It's the same function, but it can't overflow for huge inputs.

### 3.2 Creating a network
```python
net = MLP([4, 4, 3], hidden="logistic", output="linear", rng=0)
```
- `[4, 4, 3]` = 4 inputs, 4 hidden, 3 outputs.
- `W[m]` is a matrix (inputs × outputs of that layer); `b[m]` is the bias vector (the paper's θ).
- **Starting weights** are random in `±3/√(fan-in)` (Section 7.3.1). Random values let hidden neurons learn **different** things; the fan-in scaling keeps sigmoids out of their flat regions.

### 3.3 One flat vector of all weights
`get_params()` joins every W and b into **one long vector**, and `set_params(w)` splits it back.
The optimizers only see this vector. They don't need to know about layers. For a 4-4-3 network, it has 4·4+4+4·3+3 = **35 numbers**.

### 3.4 `forward`: run the network (Eqs. 24–25)
```python
o = phi(outs[-1] @ W + b)       # each layer: weighted sum, then activation
```
It returns **every** layer's output, because backprop needs them.

`masks` model **broken neurons** (Section 10). A mask of 0 forces that hidden node's output to 0.

### 3.5 `gradient`: backpropagation (Eqs. 26–31), the heart of the file
```python
delta = -(Y - out) * phi'(out) / N            # blame at the output layer
for each layer, going backwards:
    gW = previous_output.T @ delta            # blame for each weight
    gb = delta.sum(axis=0)                    # blame for each bias
    delta = (delta @ W.T) * phi'(previous)    # pass blame one layer back
```

**In words:**
1. At the output, each neuron's blame (δ) = its error × how sensitive it is (the slope of φ).
2. A weight's blame = the δ of the neuron it feeds × the value coming in on it.
3. To go back a layer, each hidden neuron collects the δs of the neurons it feeds, **weighted by the connection strengths**, times its own slope. That's the **chain rule**.

`+ 2 * decay * w` adds the gradient of weight decay (Eq. 34).

**How we know it's right:** the tests nudge every weight by ±0.000001, measure the change in error directly, and check that it matches `gradient()`. This is a **finite-difference check**, and it agrees to about 1e-9.

### 3.6 `jacobian`: for Levenberg–Marquardt
LM needs more than the total gradient: it needs **how every single output error changes with every weight**. That's a matrix **J** with one row per (sample, output) pair and one column per weight. It's the same backward pass, but run once per output neuron and **without summing** over samples.
Check: `Jᵀe / N` equals the normal gradient (Eq. 45). A test verifies this.

---

## 4. `optimizers.py`: 11 ways to train

Everything goes through one function:
```python
history = train(net, X, Y, method="lm", epochs=1000, goal=1e-3)
```
It returns the training MSE after every epoch, and stops early when:
- the MSE reaches `goal` (the paper's 0.001),
- the gradient is ~0 (a minimum was reached), or
- the method can't improve any more.

### 4.1 First-order methods: they use only the slope

| Method | Key line | Idea |
|---|---|---|
| `gd` | `w = w - lr * g` | walk downhill (Eq. 39) |
| `gdm` | `velocity = -lr*g + 0.9*velocity; w += velocity` | keep some speed from last step (Eq. 32) |
| `sgd` | update after **each** sample, shuffled | online BP (4.4) |
| `rprop` | `w -= step_sizes * sign(g)` | only the **sign**. Step grows ×1.2 while the sign holds, shrinks ×0.5 when it flips |

### 4.2 `lm`: Levenberg–Marquardt (Eqs. 45–48)
```python
H = J.T @ J / N                                    # curvature, from first derivatives
step = solve(H + sigma * I, -gradient)             # Eq. (47)
if error went down: accept, sigma /= 10            # trust the fast model more
else:               sigma *= 10, try again         # be more careful
```
- **Big σ** → a small, safe gradient-descent step.
- **Small σ** → a big Gauss–Newton jump.

LM automatically slides between the two. That's why it needs so few epochs.

### 4.3 `bfgs` and `oss`: quasi-Newton (Eqs. 49–53)
- **BFGS** keeps a matrix `H_inv` that **learns the curvature** from how the gradient changed: `s` = the step taken, `z` = the change in gradient. Eq. (51) updates `H_inv` so that `H_inv @ z = s`.
- **OSS** = BFGS that resets `H_inv` to the identity every step. It needs only vectors, with no matrix.
- Both use `line_search` to pick the step length.

### 4.4 `line_search`: how far to step
Given a downhill direction `d`:
- **Too far?** Halve the step until the error drops enough (the **Armijo** test).
- **First try already good?** Keep doubling while it still improves.

### 4.5 Conjugate gradient: `cgf`, `cgp`, `cgb`, `scg` (Eqs. 55–57)
```python
d = -g_new + beta * d       # new direction = downhill + some of the old direction
```
- `cgf`: β = Fletcher–Reeves
- `cgp`: β = Polak–Ribière (Eq. 57)
- `cgb`: restart (d = −g) when consecutive gradients stop being orthogonal (the Powell–Beale test)
- `scg`: **no line search**. It estimates curvature from two nearby gradients and uses LM-style damping `lam`.

### 4.6 Extra options
| Option | What it does | Section |
|---|---|---|
| `decay=1e-3` | weight decay, λ·Σw² | 5.3 |
| `val=(Xv, Yv)` | **early stopping**: at the end, the net gets the weights with the lowest **validation** error | 5.2 |
| `patience=20` | stop after 20 epochs without validation improvement | 5.2 |
| `fault_rate=0.2` | (sgd only) randomly break 20% of hidden nodes for each sample: **fault injection** | 10.1 |

---

## 5. `experiment_iris.py`: reproducing Table 1

```
for each of 50 runs:
    same train/test split, new random starting weights
    for each of the 8 methods:
        start from the SAME weights (fair comparison)
        train (goal 0.001, max 1000 epochs)
        record epochs, training MSE, test accuracy, time
print the table next to the paper's numbers; save results.md and the figure
```
- **Targets:** +1 for the right species, −1 for the others. The **largest** output wins.
- **Inputs** are scaled to [−1, 1]; the paper doesn't say, so this is a standard choice.
- **One fixed split:** the paper's zero standard deviations for LM and BFGS show it used one split. `--resplit` draws a new one every run instead.
- **Options:** `--runs 5` for a quick check, `--methods lm bfgs` to compare only some.

---

## 6. `test_survey.py`: what the 40 tests prove

| Tests | Prove |
|---|---|
| `test_perceptron_learns_separable_gates` | AND, OR, NAND, AND-NOT are learned |
| `test_perceptron_never_converges_on_xor` | XOR never converges (Section 3.2) |
| `test_perceptron_convergence_theorem...` | separable random data **always** converges, 10 datasets |
| `test_pocket_gets_best_line_for_xor` | pocket reaches 75% on XOR |
| `test_lms_reaches_least_squares_solution` | LMS and α-LMS end at the exact least-squares weights |
| `test_backprop_gradient_matches_finite_differences` | **backprop is correct** for 3 activation pairs, with weight decay and a dead node |
| `test_jacobian_matches...` | the Jacobian is correct, and `Jᵀe` = gradient (Eq. 45) |
| `test_mlp_solves_xor` | 7 methods solve XOR from most random starts |
| `test_second_order_gets_trapped_more_often_than_bp` | Section 8's claim: BFGS gets stuck on XOR more than BP |
| `test_momentum_speeds_up_bp` | momentum needs < 1/3 of the epochs |
| `test_every_method_learns_iris` | all 11 methods train and reach ≥ 90% test accuracy |
| `test_second_order_methods_stop_first` | Table 1's pattern: LM and BFGS need far fewer epochs than RProp |
| `test_weight_decay_shrinks_weights` | weight decay keeps weights small |
| `test_early_stopping_keeps_best_validation_weights` | early stopping returns the best epoch, not the last |
| `test_open_node_fault_zeroes_the_node` | a broken node outputs 0 |
| `test_fault_injection_training...` | fault injection makes the net survive a dead node better |

---

## 7. Paper → code map

| In the paper | In the code |
|---|---|
| Eq. (15)–(18) perceptron rule | `perceptron_train` |
| Pocket algorithm | `pocket_train` |
| Eq. (19), (21) LMS / α-LMS | `lms_train` |
| Eq. (3)–(6), (64) activations | `ACTIVATIONS` in `mlp.py` |
| Eq. (24)–(25) forward pass | `MLP.forward` |
| Eq. (26) MSE, (34) weight decay | `MLP.loss` |
| Eq. (29)–(31) backpropagation | `MLP.gradient` |
| Eq. (45)–(46) Jacobian, Gauss–Newton | `MLP.jacobian` |
| Eq. (32) momentum | `method="gdm"` |
| RProp (7.5) | `method="rprop"` |
| Eq. (47)–(48) Levenberg–Marquardt | `method="lm"` |
| Eq. (49)–(53) BFGS, line search | `method="bfgs"`, `line_search` |
| Eq. (55)–(57) conjugate gradient | `method="cgf"/"cgp"/"cgb"/"scg"` |
| Early stopping (5.2) | `train(..., val=...)` |
| Weight init ±3/√n (7.3.1) | `MLP.__init__` |
| Open-node fault, fault injection (10.1) | `MLP.masks`, `train(..., fault_rate=...)` |
| Table 1, Figure 4 (12.1) | `experiment_iris.py` |

**Not implemented** (the survey only mentions them briefly): pruning (OBD/OBS), cascade-correlation, Kalman filter/RLS training, EM, natural gradient, and variance-reduced SGD.

---

## 8. Try it yourself

1. In `demo.py`, change the XOR learning rate `lr=2.0` to `0.5` and `8.0`. What happens to the epochs?
2. Run `python3 experiment_iris.py --resplit --runs 20`. How much does accuracy change?
3. Train Iris with `hidden="relu"`. Does it still work?
4. Add **thermal perceptron** learning (Section 3.2): multiply each update by a factor that shrinks over time.
5. Add **OBD pruning** (Section 6.1.2): after training, remove the weight with the smallest `½·H_ii·w_i²` (use `J.T @ J` as H), retrain, repeat.
