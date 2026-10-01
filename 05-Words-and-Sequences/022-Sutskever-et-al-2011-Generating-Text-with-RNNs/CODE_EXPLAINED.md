# The code, explained simply

How the code in this folder implements Sutskever, Martens & Hinton (2011).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `mrnn.py` | the standard RNN, tensor RNN and MRNN; sparse init; bits per character, sampling, debagging; a compact Hessian-free optimizer |
| `experiments.py` | RNN vs MRNN, HF vs Adam, samples, debagging on text8 / Anna Karenina (heavy, not run here) |
| `demo.py` | A light tour (about a second) |
| `test_mrnn.py` | 12 quick tests (about a second) |

**Run it** (from `05-Words-and-Sequences/022-Sutskever-et-al-2011-Generating-Text-with-RNNs`):
```
python3 -m pytest -q             # ~1 second
python3 demo.py                  # ~1 second
python3 experiments.py --quick   # 5M characters: ~30-60 minutes
```

---

## 2. `mrnn.py`: the models

All three models take one-hot characters `xs` of shape (T, N, M) and return `(logits (T, N, M), last hidden state)`. Each has a learned starting state `h_init`.

| Class | One step |
|---|---|
| `CharRNN(M, H)` | `tanh(W_hx x + W_hh h)` (the bias is inside `W_hx`) |
| `TensorRNN(M, H)` | `W_hh` is an (M, H, H) tensor; `einsum` picks each example's matrix from its one-hot x |
| `MRNN(M, H, F)` | `f = W_fx(x) * W_fh(h)`, then `tanh(W_hf(f) + W_hx(x))` |

- `MRNN.effective_matrix(c)` returns `W_hf @ diag(W_fx[:, c]) @ W_fh`, the matrix character c synthesizes (Eq. 6/10).
- `MRNN.sparse_init(k=15)` uses `sparse_init_`: each row keeps exactly k random nonzero weights.

## 3. `mrnn.py`: evaluation and generation

- `nll(model, ids, skip)`: the summed −log P of every next character in a (T, N) id tensor, skipping the first `skip` predictions. It returns (total, count).
- `bits_per_char`: total / count / ln 2.
- `sample(model, prefix, n, temperature)`: feed the prefix, then repeatedly sample from the softmax and feed the sample back.
- `log_prob_of` and `debag(model, words, encode, ...)`: score every permutation and return the best.

## 4. `mrnn.py`: `HessianFree`

1. **`loss_and_grad(ids)`:** an ordinary backward pass, flattened into one vector g.
2. **`gauss_newton_product(ids, v)`:**
   ```python
   logits, Jv = jvp(logits_fn, params, v)          # forward-mode: J v
   p = softmax(logits)
   HJv = p * Jv - p * sum(p * Jv)                  # softmax+cross-entropy Hessian times Jv
   JtHJv = vjp(logits_fn, params)(HJv / n)         # reverse-mode: J^T (...)
   ```
   `torch.func.functional_call` lets us treat the model as a function of a parameter dictionary.
3. **`conjugate_gradient(Avp, b, x0)`:** the textbook algorithm. It only needs products A·v.
4. **`step(grad_ids, curv_ids)`:**
   - solve (G + λI)·d = −g, warm-started from 0.95 × the previous d;
   - apply d;
   - compute ρ = actual change / predicted change; then λ ×2/3 if ρ > 3/4, or ×3/2 if ρ < 1/4;
   - undo the step if the loss went up.

(The RNN-specific "structural damping" of Martens & Sutskever 2011 isn't included.)

---

## 5. `experiments.py`

| Function | What |
|---|---|
| `e1` | CharRNN(27, 500) vs MRNN(27, 350, 350) with Adam + clipping; 250-character windows, loss on the last 200; test bits per character |
| `e2` | a small MRNN (100 × 100, sparse init) trained with HF, then Adam for the same wall-clock time |
| `e3` | samples after "the meaning of life is", and completions of "england spain france germany" |
| `e4` | debagging: 500 random spans of 11 words; the 5040 orders of the middle 7 are scored **in one batched pass** (padded, masked); correct if the original order wins |

text8 has only lowercase letters and spaces (27 symbols), so there are no parentheses to balance. A corpus with punctuation would be needed for that experiment.

---

## 6. The tests

| Test | Proves |
|---|---|
| `test_big_mrnn_has_about_4_9_million_parameters` | the paper's model size |
| `test_rnn_500_has_slightly_more_parameters_than_mrnn_350` | the fair comparison of Section 3.2 |
| `test_mrnn_step_uses_the_character_specific_matrix` | Eqs. 6–8 |
| `test_mrnn_is_a_factored_tensor_rnn` | the MRNN = a tensor RNN with factored matrices |
| `test_each_character_matrix_has_rank_at_most_F` | the rank-F structure |
| `test_sparse_init_gives_15_connections_per_unit` | sparse init |
| `test_untrained_uniform_model_costs_log2_M_bits` | bits per character |
| `test_sampling` | generation |
| `test_debagging_finds_the_order_the_model_prefers` | debagging logic |
| `test_gauss_newton_vector_product_is_exact` | G·v = (explicit JᵀHJ)·v; G is PSD |
| `test_conjugate_gradient_solves_spd_systems` | CG |
| `test_hessian_free_steps_reduce_the_loss` | HF learns |

---

## 7. Try it yourself

1. Train a CharRNN and an MRNN of equal size with Adam on a text that has parentheses (e.g. source code). Measure how often their samples close every "(" they open.
2. Add **structural damping** to `HessianFree`: also penalize changes in the hidden states, which helps RNNs.
3. Replace the MRNN with an LSTM of the same size (Paper 021, with a forget gate) and compare bits per character.
4. Plot the singular values of W^(c) for a few characters after training. Do characters like space and "e" use very different matrices?
