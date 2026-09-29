"""A guided tour of dropout (Hinton et al., 2012).  Run:  python3 demo.py   (~1 minute)
(For the MNIST numbers and figures, run experiments.py.)
"""

import itertools

import torch
import torchvision

from dropout import DropoutNet, mc_average_errors, test_errors, train


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


section("1. What dropout does to one training case")
torch.manual_seed(0)
net = DropoutNet([6, 8, 3], keep_input=0.8, keep_hidden=0.5, init_std=1.0)
x = torch.ones(1, 6)
with torch.no_grad():
    for k in range(3):
        h_in = x * (torch.rand_like(x) < 0.8).float()
        h = torch.sigmoid(h_in @ net.W[0] + net.b[0])
        mask = (torch.rand_like(h) < 0.5).float()
        print(f"   pass {k + 1}: inputs kept {h_in.int().tolist()[0]}   hidden units kept {mask.int().tolist()[0]}")
print("   Every pass trains a different 'thinned' network - but they all share the same weights.")

section("2. At test time: ONE mean network = the average of ALL thinned networks")
net = DropoutNet([5, 10, 3], keep_hidden=0.5, init_std=1.0, seed=1).double()
x = torch.randn(4, 5, dtype=torch.float64)
with torch.no_grad():
    L = torch.stack([torch.log_softmax(net(x, masks=[torch.ones(5, dtype=torch.float64),
                                                     torch.tensor(m, dtype=torch.float64)]), 1)
                     for m in itertools.product([0, 1], repeat=10)])
    geo = torch.softmax(L.mean(0), 1)
    mean = torch.softmax(net(x, mode="mean"), 1)
print(f"   geometric mean of all {len(L)} sub-networks (first example): {geo[0].numpy().round(4)}")
print(f"   mean network (all units, outgoing weights halved)     : {mean[0].numpy().round(4)}")
print("   Identical: 1,024 networks averaged for the price of one forward pass.")

section("3. MNIST, 20,000 training images, 784-800-800-10, 15 epochs")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tr = torchvision.datasets.MNIST("../../data", train=True, download=True)
te = torchvision.datasets.MNIST("../../data", train=False, download=True)
X = (tr.data[:20000].reshape(-1, 784).float() / 255).to(dev); y = tr.targets[:20000].to(dev)
Xt = (te.data.reshape(-1, 784).float() / 255).to(dev); yt = te.targets.to(dev)
results = {}
for name, kin, kh, lr0, mn in (("standard backprop", 1.0, 1.0, 3.0, None),
                               ("dropout (20% inputs, 50% hidden) + max-norm", 0.8, 0.5, 10.0, 15.0)):
    net = DropoutNet([784, 800, 800, 10], keep_input=kin, keep_hidden=kh, seed=0).to(dev)
    h = train(net, X, y, 15, dropout=kh < 1, max_norm=mn, lr0=lr0, lr_decay=0.9, mom_epochs=5,
              Xte=Xt, yte=yt)
    train_err = test_errors(net, X, y)
    results[name] = (train_err, h[-1])
    print(f"   {name:45s}: training errors {train_err:5d} / 20000,  test errors {h[-1]:4d} / 10000")
(sb_tr, sb_te), (do_tr, do_te) = results.values()
print(f"   Gap between test and training error rate: standard {sb_te / 100 - sb_tr / 200:.2f} points,"
      f" dropout {do_te / 100 - do_tr / 200:.2f} points.")
print("   Dropout keeps the network from memorizing the training set, so it generalizes better.")
print(f"   Averaging 20 sampled dropout nets instead of the mean net: {mc_average_errors(net, Xt, yt, 20)} test errors")
