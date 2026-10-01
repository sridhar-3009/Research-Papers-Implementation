# The code, explained simply

How the code in this folder implements Simonyan & Zisserman (2015).
Read [EXPLAINED.md](EXPLAINED.md) first.

---

## 1. The files

| File | What it is |
|---|---|
| `vgg.py` | Table 1 as data, the VGG network, parameter counting, receptive fields, init from net A, FC → conv conversion, dense prediction, scale jittering, multi-crop |
| `experiments.py` | Depth, 3×3 vs 5×5, initialization, scale jittering, evaluation methods on CIFAR-10 (heavy, not run here) |
| `demo.py` | A light tour, with no training (about a second) |
| `test_vgg.py` | 14 quick tests (about a second) |

**Run it** (from `03-CNNs-and-Vision/015-Simonyan-Zisserman-2015-VGG`):
```
python3 -m pytest -q                   # ~1 second
python3 demo.py                        # ~1 second
python3 experiments.py --quick         # ~15 minutes
python3 experiments.py --only e1       # 1-2 hours on a CPU
```

---

## 2. `vgg.py`

### `CONFIGS`: Table 1 as lists
```python
"D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
```
- A number means a 3×3 conv with that many output channels.
- `("1x1", c)` is a 1×1 conv (config C).
- `"M"` is a 2×2 max-pool.
- `"LRN"` is AlexNet's normalization (config A-LRN).

`weight_layers(name)` counts the convs and adds 3.

### `count_params(name)`
It walks the list:
- each conv adds (k·k·C_in + 1)·C_out;
- each "M" halves the spatial size;
- at the end it adds the three FC layers.

So Table 2 is computed **without building anything**. The test also builds each network on `device="meta"` (PyTorch creates the tensors' shapes only, with no memory and no time) and checks the two counts agree.

### `VGG(name, num_classes, input_size, fc, width_div, init)`
- `features`: the conv/ReLU/pool stack built from the config. Padding is `k // 2`: 1 for 3×3, 0 for 1×1, so the size is preserved.
- `classifier`: Flatten → FC → ReLU → Dropout → FC → ReLU → Dropout → FC.
- `final_size`, `final_channels`: what reaches the classifier (7 and 512 for the paper's net). `to_fully_convolutional` needs these.
- **Shrinking for experiments:** `width_div` divides every channel count, `fc` sets the FC width, and `input_size` the image size. With input 32 there are five pools, so the classifier sees a 1×1 map.
- **`reset(init)`:**

  | Option | Weights |
  |---|---|
  | `"paper"` | std 0.1, the paper's "10⁻² variance" taken literally |
  | `"paper001"` | std 0.01, the common reading |
  | `"glorot"` | Glorot / Xavier uniform |

  Biases are 0 in all cases.

### `receptive_field(n, k=3)`
`n·(k − 1) + 1`. Each extra 3×3 layer grows the window by 2 pixels.

### `init_from_A(deep, net_a)`
- Copies A's **first four conv layers** into the deep net. Each one goes into the next layer **of the same shape**, because the deep nets have extra layers in between (for E, that's layers 1, 3, 5, 6).
- Copies all **three FC layers** directly.
- Returns the list of (A layer, deep layer) pairs.

### `to_fully_convolutional(net)`
Reshapes the FC weights into convolution kernels:
```python
conv6.weight = fc6.weight.view(4096, 512, 7, 7)     # FC on a 7x7x512 block = a 7x7 conv
conv7.weight = fc7.weight[:, :, None, None]          # FC = a 1x1 conv
```
- Dropout disappears, because it does nothing at test time.
- On a 224 image the output is a 1×1 map that equals the original output.
- On a bigger image it's an H×W map of class scores.

### `dense_predict(fcn, x)`
Averages the score map over positions, applies softmax, and averages with the flipped image.

### Scales and crops
- `rescale_shorter_side(img, S)`: an isotropic resize with bilinear interpolation.
- `random_crop_flip(img)`: a random 224×224 crop, mirrored half the time.
- `scale_jitter(img, 256, 512)`: S ~ U[256, 512] for each image, then rescale and crop. It returns the crop and S.
- `multi_crop(img)`: a 5×5 grid of crop positions, plus their flips = 50 crops.

---

## 3. `experiments.py`

Every network is `VGG(name, num_classes=10, input_size=32, fc=512, width_div=4)`: the same depth as the paper, a quarter of the width.

| Function | Tests | Paper |
|---|---|---|
| `e1` | all 6 configs | error falls with depth; LRN useless; C < D |
| `e2` | B vs `ShallowB` (each pair of 3×3 replaced by one 5×5) | shallow is 7% worse |
| `e3` | E with 3 random inits, and E initialized from a trained A | random init can stall |
| `e4` | D with fixed scale vs jitter S ∈ [32, 48] | jitter is better |
| `e5` | test images upscaled to 40×40: dense vs a 3×3 grid of 32×32 crops (+ flips) vs both | both together is best |

`train()` follows Section 3.1: SGD with momentum 0.9 and weight decay 5e-4, with the learning rate divided by 10 on a plateau (PyTorch's `ReduceLROnPlateau`). The batch is 128 instead of 256.

---

## 4. The tests

| Test | Proves |
|---|---|
| `test_depths_of_table_1` | 11/11/13/16/16/19 layers, 5 pools |
| `test_parameter_counts_of_table_2` | 133/133/134/138/144M; formula = built net |
| `test_most_parameters_are_in_the_first_fc_layer` | 74% of D is fc6 |
| `test_width_doubles_after_each_pool_until_512` | 64 → 512 |
| `test_receptive_fields_and_parameter_savings` | 5×5, 7×7; 27C² vs 49C² |
| `test_two_3x3_convs_really_see_a_5x5_window` | measured with gradients |
| `test_convs_preserve_size_and_pools_halve_it` | 224 → 7 |
| `test_config_c_uses_1x1_convs` | three 1×1 convs |
| `test_init_from_A_copies_first_four_convs_and_all_fc` | shape matching for D and E |
| `test_paper_init_is_std_0_1_as_written` | the literal reading |
| `test_fully_convolutional_net_equals_original_on_training_size` | the conversion is exact |
| `test_dense_evaluation_works_on_bigger_images` | score maps |
| `test_rescale_and_scale_jitter` | S varies; crop is 224 |
| `test_multi_crop_is_50_per_scale` | 5×5 × 2 |

---

## 5. Try it yourself

1. Add **He initialization** (`nn.init.kaiming_normal_`) to `reset()` and rerun demo part 3. Does the activation size stay constant?
2. Add **Batch Normalization** (Paper 012) after every conv in config E and train it from random init in `e3`. Is the pre-training trick still needed?
3. Replace the classifier with **global average pooling + one FC layer**. How many parameters does D lose? (Hint: about 120M.)
4. Use torchvision's pretrained `vgg16` (a 528 MB download) as a frozen feature extractor + logistic regression on CIFAR-10, like Appendix B.
