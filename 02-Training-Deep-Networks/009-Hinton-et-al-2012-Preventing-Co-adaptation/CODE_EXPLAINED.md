# The code, explained simply

How the code in this folder implements Hinton et al. (2012).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `dropout.py` | The dropout network with explicit masks, max-norm, the paper's training rule, Monte-Carlo averaging |
| `experiments.py` | The geometric-mean check, the MNIST comparison (Figure 1), mean network vs averaging, the features (Figure 5) |
| `demo.py` | A guided tour (trains two MNIST nets for 15 epochs: ~1 minute on a GPU) |
| `test_dropout.py` | 6 quick tests |

**Run it** (from `02-Training-Deep-Networks/009-Hinton-et-al-2012-Preventing-Co-adaptation`):
```
python3 -m pytest -q             # ~1 s
python3 demo.py                  # ~1 min (GPU) - trains two small MNIST nets
python3 experiments.py --quick   # a few minutes
python3 experiments.py           # FULL: ~25 min on a GPU. Strong machine only.
```

---

## 2. `dropout.py`

### `DropoutNet`
```python
net = DropoutNet([784, 800, 800, 10], keep_input=0.8, keep_hidden=0.5)
```
- `keep` = the probability a unit is **kept** (so 0.5 means drop half).
- Weights start N(0, 0.01²) (Appendix A.1); logistic hidden units; the output is logits (softmax is in the loss).

### `forward(x, mode)`: three ways to run the network
| `mode` | What happens | Used for |
|---|---|---|
| `"train"` | `h = h * (rand < p)`: a **fresh random mask for every example and every unit** | training |
| `"mean"` | no masks; each layer's weights multiplied by p (halved for hidden layers) | testing: the mean network |
| `masks=[...]` | fixed, given masks | enumerating every thinned network exactly |

`torch.rand_like(h) < p` is a 0/1 tensor that is 1 with probability p. Multiplying by it switches off the dropped units.

### `max_norm_(net, 15)`
For each hidden unit, compute its incoming weight vector's squared length `(W**2).sum(0)`. Wherever it exceeds 15, multiply that column by √(15/length²), so it lands exactly on the limit.

### `train`: Appendix A.1's update rule
```
Δw_t = p_t · Δw_{t−1} − (1 − p_t) · ε_t · gradient
ε_t  = 10 · 0.998^epoch
p_t  = momentum, rising linearly 0.5 → 0.99 over 500 epochs
```
- The `(1 − p_t)` factor keeps the effective step from growing as momentum grows.
- After every update, max-norm is applied.
- For shorter runs, pass a stronger `lr_decay` and a shorter `mom_epochs` (experiments.py compresses them 30×).

### Testing helpers
- `test_errors(net, X, y)`: the number of mistakes using the mean network.
- `mc_average_errors(net, X, y, k)`: sample k thinned networks and average their **probabilities**.

---

## 3. `experiments.py`

| Part | What it does |
|---|---|
| `e1_geometric_mean` | enumerates all 1,024 thinned nets of a 10-unit net and compares their geometric mean to the mean network |
| E2 | MNIST 784-800-800-10: standard backprop (best of 3 learning rates), max-norm only, 50% hidden dropout, + 20% input dropout |
| E3 | the mean network vs averaging k = 1…100 sampled nets |
| E4 | 784-500-500 first-layer features, backprop vs dropout, saved as an image grid |

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_mean_network_is_the_geometric_mean_of_all_subnetworks` | the exact claim on page 2 |
| `test_mean_network_beats_average_log_probability` | the log-probability claim on page 2 |
| `test_training_mode_drops_the_right_fraction` | masks keep ~80% of inputs; training passes differ, mean passes don't |
| `test_mean_network_matches_expected_input_to_next_layer` | halving the weights = the average input under dropout |
| `test_max_norm_caps_squared_length` | the constraint works |
| `test_dropout_net_learns_a_small_problem` | end-to-end training works |

---

## 5. Try it yourself

1. Drop 80% of hidden units instead of 50%. Does it underfit?
2. Remove max-norm but keep the learning rate at 10. What happens?
3. Add dropout to only ONE hidden layer. The paper says all layers is better; check it.
4. Plot the first-layer weights of your trained nets as 28×28 images and compare.
