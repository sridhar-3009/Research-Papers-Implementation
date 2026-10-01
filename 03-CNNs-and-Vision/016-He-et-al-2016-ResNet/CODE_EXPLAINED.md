# The code, explained simply

How the code in this folder implements He et al. (2016).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `resnet.py` | basic and bottleneck blocks, shortcut options A/B/C, ImageNet ResNets (18–152), CIFAR ResNets (6n+2), FLOP counting, layer responses |
| `experiments.py` | Figure 6 / Table 6, Figure 7, shortcut options, ResNet-1202 on CIFAR-10 (heavy, not run here) |
| `demo.py` | A light tour, with no training (a few seconds) |
| `test_resnet.py` | 14 quick tests (a couple of seconds) |

**Run it** (from `03-CNNs-and-Vision/016-He-et-al-2016-ResNet`):
```
python3 -m pytest -q                 # ~2 seconds
python3 demo.py                      # ~4 seconds
python3 experiments.py --quick       # 2,000 iterations per net, ~20-30 minutes
python3 experiments.py --only e1     # the paper's 64k iterations: many hours
```

---

## 2. `resnet.py`

### `he_init`
Conv weights ~ N(0, 2/n) with n = k·k·C_in (He et al. 2015). BN starts at scale 1 and shift 0.

### Shortcuts: `make_shortcut(in_ch, out_ch, stride, option)`
| Situation | A | B | C |
|---|---|---|---|
| same shape | `nn.Identity()` | identity | 1×1 conv + BN |
| shape changes | `ZeroPadShortcut` | 1×1 conv (stride) + BN | 1×1 conv + BN |

`ZeroPadShortcut` takes every second pixel (`x[:, :, ::2, ::2]`) and appends zero channels with `F.pad`. It has no parameters.

### `BasicBlock(in_ch, out_ch, stride, option, residual)`
```python
a = bn1(conv1(x))
f = bn2(conv2(relu(a)))            # this is F(x)
return relu(f + shortcut(x))       # residual
return relu(f)                     # plain (residual=False): the SAME layers, no addition
```
- BN comes after each conv and before the ReLU, as in Section 3.4.
- `record=True` saves the std of `a` and `f`, which `layer_responses` uses for Figure 7.

### `Bottleneck(in_ch, mid_ch, stride, ...)`
1×1 (`in → mid`, carrying the stride) → 3×3 (`mid → mid`) → 1×1 (`mid → 4·mid`), plus the shortcut. Putting the stride in the **first 1×1** matches the paper's FLOPs (see EXPLAINED §8).

### `ImageNetResNet(depth, residual, option)`
- `IMAGENET[depth]` gives the block type and the number of blocks per stage (Table 1).
- conv1 7×7/2 → max-pool → 4 stages (64, 128, 256, 512 channels; ×4 for bottlenecks). The first block of stages 2–4 has stride 2. Then global average pooling and FC.

### `CifarResNet(n, residual, option)`
- conv 3×3 (16) → 3 stages of n basic blocks (16 / 32 / 64 channels; stride 2 at the start of stages 2 and 3) → global average pooling → FC 10.
- `self.depth = 6n + 2`.

### `weight_layers`, `count_params`, `count_macs`
- `weight_layers`: conv1 + FC + 2 per basic block (3 per bottleneck). Shortcut projections aren't counted, like in the paper.
- `count_macs` attaches **forward hooks** to every conv and FC layer. For a conv: `output elements × (C_in / groups) × k × k`.
  - Run it on a model built inside `with torch.device("meta"):`. The forward pass then only computes **shapes**, with no numbers and no memory. That's how the tests and demo measure ResNet-152 at no cost.

### `layer_responses(model, x)`
Turns on `record` in every block, runs one forward pass, and collects the stds in order: Figure 7's curve.

---

## 3. `experiments.py`

- `train()` follows the paper's CIFAR recipe:
  - SGD with momentum 0.9 and weight decay 1e-4, batch 128;
  - learning rate 0.1 → 0.01 → 0.001 at 50% and 75% of the run (that's 32k / 48k of 64k);
  - pad + crop + flip augmentation;
  - `warmup=True` holds the learning rate at 0.01 until the running training error is below 80% (for ResNet-110 and ResNet-1202).
- `e1` trains plain and ResNet at 20/32/44/56 layers, plus ResNet-110, and saves 5 of them for `e2`.
- `e2` loads them and records the layer responses (Figure 7).
- `e3` compares options A/B/C on ResNet-32. `e4` runs ResNet-1202, only with `--with-1202`.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_depths_of_table_1` | 18/34/50/101/152 weight layers |
| `test_flops_of_table_1` | 1.8 … 11.3 G multiply-adds (±4%); < VGG-16 |
| `test_parameter_counts_option_b` | = torchvision's ResNets |
| `test_plain_and_residual_have_the_same_parameters_with_option_a` | "no extra parameter" |
| `test_options_a_b_c_add_parameters_in_order` | A < B < C |
| `test_output_shape` | a real forward pass (ResNet-18) |
| `test_cifar_depths_and_parameter_counts_of_table_6` | 0.27M … 19.4M, 20 … 1202 layers |
| `test_cifar_forward_and_map_sizes` | 32×32×16 → 16×16×32 → 8×8×64 |
| `test_zero_residual_makes_a_block_an_identity` | F = 0 ⇒ identity |
| `test_deeper_net_can_copy_a_shallower_one` | the construction argument |
| `test_gradient_passes_through_the_shortcut` | ∂y/∂x = 1 when F = 0 |
| `test_option_a_shortcut_pads_zeros_and_subsamples` | option A |
| `test_bottleneck_shapes` | 256 → 512 channels, 56 → 28, stride in the 1×1 |
| `test_layer_responses_are_recorded_for_every_3x3_layer` | Figure 7 tool |

---

## 5. Try it yourself

1. Write a **pre-activation block** (BN → ReLU → conv → BN → ReLU → conv, then + x, with no ReLU after the addition), from He et al. 2016b. Compare the gradient numbers in demo part 3.
2. Initialize the last BN of every residual branch to **0** ("zero-init residual", Goyal et al. 2017). Then every block starts as an identity. Does ResNet-110 still need the warm-up?
3. Remove BN entirely from `CifarResNet` and rerun demo part 3. What happens to plain vs residual now?
4. After `e1`, **delete one block** from the trained ResNet-110 at test time (make it return its input) and measure the error; do the same to plain-56. Veit et al. (2016) found ResNets barely notice, while plain nets collapse: evidence for the "ensemble of paths" view.
