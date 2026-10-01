"""A light tour of AlexNet (Krizhevsky, Sutskever & Hinton 2012). No training; a few seconds.

    python3 demo.py
"""

import torch
import torch.nn.functional as F

from alexnet import AlexNet, PaperDropout, local_response_norm, pca_color_augment, rgb_pca, ten_crop

torch.manual_seed(0)


def line(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


line("1. The network, layer by layer (Section 3.5)")
net = AlexNet().eval()
x = torch.randn(1, 3, 224, 224)
h = x
print("(each shape is the layer's output AFTER its pooling, if it has one: conv1 makes 55x55, pooled to 27x27)")
print(f"{'input':8s} {tuple(h.shape[1:])}")
for name, f in [("conv1", lambda h: net.pool(net.norm(F.relu(net.conv1(h))))),
                ("conv2", lambda h: net.pool(net.norm(F.relu(net.conv2(h))))),
                ("conv3", lambda h: F.relu(net.conv3(h))),
                ("conv4", lambda h: F.relu(net.conv4(h))),
                ("conv5", lambda h: net.pool(F.relu(net.conv5(h))))]:
    h = f(h)
    layer = getattr(net, name)
    print(f"{name:8s} {tuple(h.shape[1:])}   kernels {tuple(layer.weight.shape)}   params {layer.weight.numel() + layer.bias.numel():>10,}")
for name in ("fc6", "fc7", "fc8"):
    layer = getattr(net, name)
    print(f"{name:8s} {layer.out_features:5d} units          params {layer.weight.numel() + layer.bias.numel():>10,}")
total = sum(p.numel() for p in net.parameters())
fc = sum(p.numel() for n, p in net.named_parameters() if n.startswith("fc"))
print(f"total {total:,} parameters; {100 * fc / total:.0f}% of them are in the 3 fully connected layers")
print("-> the conv layers do most of the COMPUTATION, the fc layers hold most of the PARAMETERS.")
print("   That's why dropout (a regularizer) goes in fc6 and fc7.")

line("2. ReLU vs tanh: why ReLU trains faster (Section 3.1)")
for v in (0.5, 2.0, 5.0):
    t = torch.tensor(v, requires_grad=True)
    gt = torch.autograd.grad(torch.tanh(t), t)[0].item()
    gr = torch.autograd.grad(F.relu(t), t)[0].item()
    print(f"input {v:3.1f}: slope of tanh {gt:.5f}   slope of ReLU {gr:.0f}")
print("-> once a tanh unit's input is large, its gradient is almost 0 (it 'saturates');")
print("   a ReLU passes the gradient unchanged for any positive input.")

line("3. Local response normalization: loud maps quiet their neighbours (Section 3.3)")
a = torch.full((1, 7, 1, 1), 10.0)
print("7 maps, all with activity 10     ->", [round(v, 2) for v in local_response_norm(a).flatten().tolist()])
a[0, 3] = 300.0
print("map 3 fires at 300               ->", [round(v, 2) for v in local_response_norm(a).flatten().tolist()])
print("-> maps 1, 2, 4, 5 (within n/2 = 2 of map 3) are pushed down; maps 0 and 6 are not.")

line("4. Test-time augmentation: 10 crops (Section 4.1)")
img = torch.rand(3, 256, 256)
crops = ten_crop(img)
print("one 256x256 image ->", tuple(crops.shape), "= 4 corners + center, each also mirrored")
print("the network's softmax outputs on these 10 are averaged (paper: 39.0% -> 37.5% top-1)")

line("5. PCA colour augmentation (Section 4.1)")
imgs = torch.rand(20, 3, 8, 8) * torch.tensor([1.0, 0.8, 0.6]).view(1, 3, 1, 1)
vals, vecs = rgb_pca(imgs)
print("eigenvalues of the RGB covariance:", [f"{v:.4f}" for v in vals.tolist()])
out = pca_color_augment(imgs[0], vals, vecs, generator=torch.Generator().manual_seed(1))
print("added to every pixel of this image (R, G, B):", [f"{v:+.4f}" for v in (out - imgs[0])[:, 0, 0].tolist()])
print("-> one small colour/brightness shift per image, along the directions colours naturally vary.")

line("6. Dropout the 2012 way (Section 4.2)")
d = PaperDropout()
v = torch.ones(10)
d.train(); print("training (random half set to 0):", d(v).tolist())
d.eval(); print("test (all kept, times 0.5):      ", d(v).tolist())
