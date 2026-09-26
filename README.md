# Research Papers Implementation

Implementing classic and modern research papers from scratch: reading the paper closely, rebuilding the method in code, and testing that the code reproduces the paper's claims.

## Papers

| # | Paper | Year | What's implemented |
|---|---|---|---|
| 1 | [A Logical Calculus of the Ideas Immanent in Nervous Activity](01-McCulloch-Pitts-1943/) (McCulloch & Pitts) | 1943 | The first artificial neuron: a network simulator, all of Figure 1 (including the heat/cold illusion), memory loops, and a compiler that turns logic formulas into networks (Theorem II). 21 tests. |

## Folder naming

Each paper gets its own folder named `NN-FirstAuthor-SecondAuthor-Year`, numbered in the order implemented (e.g. `01-McCulloch-Pitts-1943`, `02-Rosenblatt-1958`).

## Each paper folder contains

- `EXPLAINED.md`: the paper, explained simply, section by section
- `CODE_EXPLAINED.md`: how the code works and how it maps to the paper
- The code, a `demo.py` to watch it run, and tests that check the paper's claims

## Running

Plain Python 3.10+ with no dependencies; the tests need `pytest`.

```bash
cd 01-McCulloch-Pitts-1943
python3 demo.py
python3 -m pytest -q
```
