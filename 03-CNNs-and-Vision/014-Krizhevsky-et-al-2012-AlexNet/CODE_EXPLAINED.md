# The code, explained simply

How the code in this folder implements Krizhevsky, Sutskever & Hinton (2012).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `alexnet.py` | AlexNet, LRN by hand, 2012-style dropout, the paper's SGD update, the learning-rate rule, data augmentation, a small CIFAR-10 net |
| `experiments.py` | ReLU vs tanh, LRN, overlapping pooling, augmentation, dropout on CIFAR-10; the full AlexNet on an ImageNet-style folder (heavy, not run here) |
| `demo.py` | A light tour, with no training (about a second) |
| `test_alexnet.py` | 13 quick tests (a couple of seconds) |

**Run it** (from `03-CNNs-and-Vision/014-Krizhevsky-et-al-2012-AlexNet`):
```
python3 -m pytest -q                     # ~2 seconds
python3 demo.py                          # ~1 second
python3 experiments.py --quick           # CIFAR parts, small: ~10 minutes
python3 experiments.py --only e6 --imagenet DIR    # the real network: days on real ImageNet
```

---

## 2. `alexnet.py`

### `local_response_norm(a, k, n, alpha, beta)`
The Section 3.3 formula, vectorized:
```python
sq = a ** 2
padded = F.pad(sq, (0, 0, 0, 0, n // 2, n // 2))     # pad the CHANNEL axis with zeros
s = sum(padded[:, j:j + C] for j in range(n))        # s[i] = sum of sq over channels i-2 .. i+2
return a / (k + alpha * s) ** beta
```
- `F.pad`'s tuple is read from the last axis backwards: (W left, W right, H top, H bottom, C front, C back). So only the channel axis gets padded.
- The zero padding handles the "clip at 0 and N − 1" edges.
- **PyTorch's built-in version divides α by n**, so `F.local_response_norm(alpha=5e-4)` equals ours with α = 1e-4. The test checks this.

### `PaperDropout`
- Training: `x * (rand >= 0.5)`.
- Test: `x * 0.5`.

This is the 2012 form. Modern PyTorch `nn.Dropout` uses the "inverted" form (×2 in training, nothing at test), which gives the same expected values.

### `AlexNet`
- **`groups=2`** in conv2, conv4 and conv5 is the two-GPU split. A grouped convolution splits the input channels and the kernels into 2 halves, and each half of the kernels only sees its half of the channels. That's why conv4's kernels are 3×3×**192**, not 3×3×384.
- **`padding=2` in conv1** turns 224 into 55×55. The paper's 224 doesn't divide cleanly.
- `lrn`, `overlap`, `act` and `dropout` switch the paper's choices on or off, for ablation experiments.
- **`paper_init()`:** N(0, 0.01²) weights; bias 1 for conv2, conv4, conv5, fc6 and fc7; 0 for the rest.
- `forward(x, return_hidden=True)` also returns fc7's 4096-d vector, the representation behind Figure 4 (right).

### `paper_sgd_step(params, velocities, lr)`
The Section 5 formula, line for line:
```python
v.mul_(0.9).add_(p, alpha=-0.0005 * lr).add_(p.grad, alpha=-lr)   # v = 0.9 v - 0.0005 lr w - lr grad
p.add_(v)                                                          # w = w + v
```
It gives the same weights as `torch.optim.SGD(momentum=0.9, weight_decay=0.0005)` (the test checks this). PyTorch keeps a velocity without the −lr factor; with a constant learning rate the two are the same.

### `DivideOnPlateau`
Call `update(val_error)` after each evaluation. After `patience` evaluations without improvement, the learning rate is divided by 10.

### Data augmentation
- `random_crop_flip(img)`: a random 224×224 patch of a 256×256 image, mirrored half the time.
- `ten_crop(img)`: returns a (10, 3, 224, 224) stack: 4 corners + center, then the same 5 mirrored.
- `rgb_pca(images)`: the 3×3 covariance of all RGB pixels, then `torch.linalg.eigh` gives eigenvalues λ and eigenvectors p.
- `pca_color_augment(img, λ, p)`: draws α ~ N(0, 0.1²) three times, computes `shift = p @ (α·λ)` (3 numbers), and adds it to every pixel.

### `SmallCifarNet`
Three 5×5 conv layers (32, 32, 64 maps), each followed by pooling, then a 10-way fc layer.
- `act`, `lrn` and `overlap` are switches.
- The overlapping pool is 3×3/2 with padding 1, so both pooling kinds give 32 → 16 → 8 → 4 and the comparison is fair.

---

## 3. `experiments.py`

| Function | What it tests |
|---|---|
| `e1` | Figure 1: iterations for ReLU and tanh to reach 25% training error. Each tries 3 learning rates and keeps the fastest. No regularization. |
| `e2` | LRN on/off (paper: 13% → 11% on CIFAR-10) |
| `e3` | Overlapping vs plain pooling |
| `e4` | No augmentation / crops + flips / + PCA colour (CIFAR version: pad 4 + random 32×32 crop) |
| `e5` | A big fc head with and without dropout: compare the train–test gap |
| `e6` | Full AlexNet on `--imagenet DIR`: the paper's pipeline (256 resize, mean subtraction, random crops + flips + PCA colour, the update rule, divide-by-10, 10-crop testing), plus a picture of the conv1 kernels (Figure 3) |

All CIFAR runs use `train_cifar`: batch 128, the paper's update rule, and the training error measured during the epoch.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_about_60_million_parameters` | every layer's count, 60,965,224 total, 96% in the fc layers |
| `test_neuron_counts_of_figure_2` | 4 of the 5 caption numbers, the 54.25 problem, output shape |
| `test_two_gpu_split_in_conv4` | half the kernels ignore the other half's maps |
| `test_paper_initialization` | biases of 1 / 0, weight std 0.01 |
| `test_relu_does_not_saturate` | slope 1 vs slope ≈ 0 |
| `test_lrn_matches_formula_and_torch` | edge and middle channels, = PyTorch (α·n) |
| `test_lrn_big_activity_inhibits_neighbours` | lateral inhibition |
| `test_overlapping_and_plain_pooling_give_the_same_size` | Section 3.4's fair comparison |
| `test_crops_and_flips` | corner and center positions, mirror images, 2048 |
| `test_pca_color_shifts_every_pixel_by_the_same_rgb_vector` | Section 4.1 colour augmentation |
| `test_paper_dropout_matches_expectation_at_test_time` | 50% kept; ×0.5 at test |
| `test_update_rule_equals_torch_sgd_...` | Section 5's formula |
| `test_divide_learning_rate_by_10_on_plateau` | the schedule |

---

## 5. Try it yourself

1. Swap LRN for **Batch Normalization** (Paper 012) in `SmallCifarNet`. Which trains faster?
2. Set `groups=1` everywhere (one big GPU). How many parameters does that add?
3. After `e6`, compute the fc7 vectors of the validation images and show each image's 6 nearest neighbours (Figure 4, right).
4. Train with `act="tanh"` and the paper's initialization. What happens? (Hint: the biases of 1 were chosen for ReLUs.)
5. Replace `PaperDropout` with `nn.Dropout(0.5)` (inverted dropout) and check the outputs match on average.
