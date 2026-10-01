"""A toy 'image' world for fast tests and the demo.

Each image holds one object with a size, a colour and a shape. Its 'CNN feature' is the one-hot
(size, colour, shape) code plus noise, pushed through a fixed random matrix (a stand-in for a frozen CNN).
Each image has 5 reference captions, like Flickr8k / MSCOCO, e.g. 'a big red circle', 'a red circle',
'a circle that is red', 'there is a big circle', 'a big red shape'.
"""

import itertools
import random

import torch

from nic import BOS, EOS, PAD

SIZES = ["small", "big"]
COLOURS = ["red", "green", "blue", "yellow"]
SHAPES = ["circle", "square", "triangle"]
WORDS = ["<pad>", "<s>", "</s>", "a", "that", "is", "there", "shape"] + SIZES + COLOURS + SHAPES
STOI = {w: i for i, w in enumerate(WORDS)}
FEAT = 32
ALL = list(itertools.product(SIZES, COLOURS, SHAPES))       # 24 kinds of image


def captions(size, colour, shape):
    return [f"a {size} {colour} {shape}", f"a {colour} {shape}", f"a {shape} that is {colour}",
            f"there is a {size} {shape}", f"a {size} {colour} shape"]


def encode(text):
    return [STOI[w] for w in text.split()]


def decode(toks):
    return " ".join(WORDS[t] for t in toks if t not in (PAD, BOS, EOS))


_proj = torch.randn(len(SIZES) + len(COLOURS) + len(SHAPES), FEAT, generator=torch.Generator().manual_seed(0))


def features(kinds, noise=0.3, generator=None):
    code = torch.zeros(len(kinds), _proj.shape[0])
    for k, (s, c, h) in enumerate(kinds):
        code[k, SIZES.index(s)] = 1
        code[k, len(SIZES) + COLOURS.index(c)] = 1
        code[k, len(SIZES) + len(COLOURS) + SHAPES.index(h)] = 1
    return (code + noise * torch.randn(code.shape, generator=generator)) @ _proj


def split(n_held=4, seed=0):
    """Hold out some (size, colour, shape) combinations: every word is seen in training, the combination is not."""
    kinds = ALL[:]
    random.Random(seed).shuffle(kinds)
    return kinds[n_held:], kinds[:n_held]


def batch(kinds, n, rng):
    """n random (image, one of its 5 captions) training pairs."""
    pick = [rng.choice(kinds) for _ in range(n)]
    caps = [encode(rng.choice(captions(*k))) for k in pick]
    return pick, caps
