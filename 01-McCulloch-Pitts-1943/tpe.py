"""Temporal propositional expressions (TPEs) and a compiler from TPE to net.

Theorem II: every TPE can be built as a loop-free net. The proof is a recipe, and
compile_tpe follows it: the formula's parse tree becomes the wiring diagram.

TPE grammar (pages 6-7):  p  |  S(e)  |  e1 OR e2  |  e1 AND e2  |  e1 AND NOT e2
There is no plain NOT: Theorem III shows a net can't compute anything that is true
when all its inputs are silent.
"""

from dataclasses import dataclass
from itertools import count   # an endless counter 0, 1, 2, ... used for neuron names

from mp_neuron import Network


# ---------------------------------------------------------------------------
# 1. The formula pieces: one small class per rule of the TPE grammar.
#    A formula is a tree of these, e.g. Or(Var("p"), Delay(Var("q")))
#    means "p OR (q one tick ago)".
#    frozen=True makes each piece read-only, so a formula can't change after it's built.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Var:          # an input neuron, e.g. Var("p")
    name: str

@dataclass(frozen=True)
class Delay:        # the paper's S: "one tick earlier"
    e: "Expr"

@dataclass(frozen=True)
class Or:           # a OR b
    a: "Expr"
    b: "Expr"

@dataclass(frozen=True)
class And:          # a AND b
    a: "Expr"
    b: "Expr"

@dataclass(frozen=True)
class AndNot:       # a AND NOT b (the only negation allowed)
    a: "Expr"
    b: "Expr"

# "Expr" means any one of the five pieces above.
Expr = Var | Delay | Or | And | AndNot


# ---------------------------------------------------------------------------
# 2. Evaluate a formula directly, with no network.
#    The tests use this as the "right answer" to check the compiled nets against.
# ---------------------------------------------------------------------------

def evaluate(e: Expr, inputs: dict[str, list[int]], t: int) -> int:
    """Evaluate a TPE directly at tick t. Inputs before tick 0 count as 0."""
    # `match` picks the case that fits the piece and unpacks its fields.
    match e:
        case Var(name):
            # Read the input at tick t; before tick 0 or past the end it's silent.
            seq = inputs.get(name, [])
            return int(0 <= t < len(seq) and bool(seq[t]))
        case Delay(x):
            # "x one tick earlier" is x evaluated at t - 1.
            return evaluate(x, inputs, t - 1)
        case Or(a, b):
            return evaluate(a, inputs, t) | evaluate(b, inputs, t)        # | is OR on 0/1
        case And(a, b):
            return evaluate(a, inputs, t) & evaluate(b, inputs, t)        # & is AND on 0/1
        case AndNot(a, b):
            return evaluate(a, inputs, t) & (1 - evaluate(b, inputs, t))  # 1 - x is NOT x
    raise TypeError(f"not a TPE: {e!r}")


def variables(e: Expr) -> list[str]:
    """All input names used in a formula, sorted. These become the net's inputs."""
    match e:
        case Var(name):
            return [name]
        case Delay(x):
            return variables(x)
        case Or(a, b) | And(a, b) | AndNot(a, b):
            # Union of both sides; set() removes duplicates, sorted() fixes the order.
            return sorted(set(variables(a)) | set(variables(b)))
    raise TypeError(f"not a TPE: {e!r}")


# ---------------------------------------------------------------------------
# 3. The compiler: formula -> network (the proof of Theorem II, as code).
# ---------------------------------------------------------------------------

def compile_tpe(e: Expr) -> tuple[Network, str, int]:
    """Build a loop-free net realizing e (Theorem II).

    Returns (net, output neuron, lag): for every tick t >= lag,
        net output at t == evaluate(e, inputs, t - lag).
    A lag > 0 is "realizable in the extended sense": the answer arrives late.
    """
    net = Network(variables(e))
    ids = count()   # gives the hidden neurons unique names: n0, n1, n2, ...

    def new(**kw) -> str:
        """Add a hidden neuron with the next free name and return that name."""
        return net.add(f"n{next(ids)}", **kw)

    def pad(node: str, ticks: int) -> str:
        """Delay a signal by `ticks` using a chain of Fig 1a delay neurons."""
        for _ in range(ticks):                      # chain of Fig 1a delay neurons
            node = new(excite={node: 2})
        return node

    def build(x: Expr) -> tuple[str, int]:
        """Build the network for x. Returns (neuron computing x, how many ticks late).

        Each case adds the matching building block from Figure 1a–d, the same way the
        formula was built from its pieces (the proof works by induction).
        """
        match x:
            case Var(name):
                # An input is already a neuron, with no delay. (Base case of the proof.)
                return name, 0
            case Delay(inner):
                node, lag = build(inner)
                # A Fig 1a delay neuron. Every neuron naturally adds one tick, and here
                # that tick IS the S we were asked for, so the lag doesn't grow.
                return new(excite={node: 2}), lag   # the neuron's own delay IS the S
            case Or(a, b) | And(a, b) | AndNot(a, b):
                # First build both halves.
                (na, la), (nb, lb) = build(a), build(b)
                # A gate needs both signals at the SAME tick. If one half is faster,
                # delay it so both have the same lag (the paper's S^m / S^n step).
                lag = max(la, lb)                   # line both signals up in time
                na, nb = pad(na, lag - la), pad(nb, lag - lb)
                # Then add the matching gate neuron (Fig 1b, 1c or 1d).
                if isinstance(x, Or):
                    gate = new(excite={na: 2, nb: 2})
                elif isinstance(x, And):
                    # add endings up: And(p, p) must give p two endings, not one
                    ends = {na: 1}
                    ends[nb] = ends.get(nb, 0) + 1
                    gate = new(excite=ends)
                else:
                    gate = new(excite={na: 2}, inhibit={nb})
                # The gate itself is one more synaptic delay.
                return gate, lag + 1
        raise TypeError(f"not a TPE: {x!r}")

    out, lag = build(e)
    return net, out, lag
