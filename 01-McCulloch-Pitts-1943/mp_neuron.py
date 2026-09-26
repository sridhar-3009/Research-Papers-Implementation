"""McCulloch–Pitts neuron and network simulator (McCulloch & Pitts, 1943).

The five assumptions from page 4 of the paper, as code:
  1. All-or-none: every neuron is 0 or 1.
  2. Fixed threshold: a neuron fires when enough excitatory endings are active.
     An input neuron may make several endings on the same target (its "weight").
  3. Only synaptic delay: time moves in ticks; a neuron at tick t depends on tick t-1.
  4. Absolute inhibition: any active inhibitory input vetoes firing.
  5. Fixed structure: connections never change.

Threshold convention: a neuron fires when active endings >= threshold (default 2).
The paper says the sum must *exceed* θ, with θ = 1 in its figures. "> 1" and ">= 2"
are the same rule.
"""

# dataclass writes __init__ for us; field() lets each neuron get its own fresh dict/set.
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# One neuron
# ---------------------------------------------------------------------------

@dataclass
class Neuron:
    # How many active excitatory endings are needed to fire (rule 2).
    threshold: int = 2

    # Excitatory connections: {source neuron name: number of endings it makes here}.
    # Example: {"A": 2, "B": 1} means A connects twice (counts 2) and B once (counts 1).
    # This whole-number count is the only kind of "weight" the paper has.
    excite: dict[str, int] = field(default_factory=dict)

    # Inhibitory connections: the names of neurons that can veto this one (rule 4).
    inhibit: set[str] = field(default_factory=set)

    def fires(self, prev: dict[str, int]) -> int:
        """Decide whether this neuron fires NOW, given every neuron's state one tick ago.

        `prev` maps each neuron name to 0 or 1 at the previous tick. Looking only at
        the previous tick is the synaptic delay (rule 3).
        """
        # Rule 4 (absolute inhibition): if ANY inhibitor fired last tick, stay silent.
        # We check this first because no amount of excitation can overrule it.
        if any(prev[src] for src in self.inhibit):
            return 0

        # Rule 2 (threshold): add up the endings from every excitatory source that fired.
        total = sum(n for src, n in self.excite.items() if prev[src])

        # Rule 1 (all-or-none): the answer is only ever 0 or 1.
        return int(total >= self.threshold)


# ---------------------------------------------------------------------------
# A network of neurons
# ---------------------------------------------------------------------------

class Network:
    def __init__(self, inputs: list[str]):
        # Input neurons (the paper's "peripheral afferents"): nothing inside the
        # net feeds them. The outside world sets their values at every tick.
        self.inputs = list(inputs)

        # All other neurons, by name. Their values are computed by the simulator.
        self.neurons: dict[str, Neuron] = {}

    def add(self, name: str, excite: dict[str, int] | None = None,
            inhibit: set[str] | None = None, threshold: int = 2) -> str:
        """Add a neuron and return its name (so builders can chain calls)."""
        # Names must be unique, or two neurons would overwrite each other's state.
        if name in self.inputs or name in self.neurons:
            raise ValueError(f"duplicate neuron name: {name}")
        # Copy the dict/set so later changes by the caller can't alter the net (rule 5).
        self.neurons[name] = Neuron(threshold, dict(excite or {}), set(inhibit or set()))
        return name

    @property
    def names(self) -> list[str]:
        """Every neuron name: the inputs first, then the rest in the order they were added."""
        return self.inputs + list(self.neurons)

    def _check(self) -> None:
        """Fail early if a neuron connects from a neuron that doesn't exist (a typo)."""
        known = set(self.names)
        for name, n in self.neurons.items():
            for src in list(n.excite) + list(n.inhibit):
                if src not in known:
                    raise ValueError(f"{name} connects from unknown neuron {src}")

    def run(self, inputs: dict[str, list[int]], steps: int | None = None,
            initial: dict[str, int] | None = None) -> dict[str, list[int]]:
        """Simulate the net and return each neuron's firing history.

        inputs:  firing sequence for each input neuron (missing ticks count as 0).
        initial: state of non-input neurons at tick 0 (default all 0). Needed for
                 nets with loops that should start active, such as a clock.

        Returns {neuron name: [value at tick 0, tick 1, ...]}.
        """
        self._check()

        # By default, run for as long as the longest input sequence.
        if steps is None:
            steps = max((len(seq) for seq in inputs.values()), default=0)

        def input_at(name: str, t: int) -> int:
            """Value of an input neuron at tick t. Past the end of its list, or if no
            list was given, it's silent (0)."""
            seq = inputs.get(name, [])
            return int(bool(seq[t])) if t < len(seq) else 0

        # ---- Tick 0: the starting state ----
        # Inputs take their first value; other neurons start at 0 unless `initial` says
        # otherwise. (A loop needs a starting spike to get going, e.g. the clock.)
        state = {name: input_at(name, 0) for name in self.inputs}
        state.update({name: int(bool((initial or {}).get(name, 0))) for name in self.neurons})

        # history[name] is that neuron's list of 0/1 values, one per tick.
        history = {name: [state[name]] for name in self.names}

        # ---- Ticks 1, 2, 3, ... ----
        for t in range(1, steps):
            # Inputs: read the next value from the outside world.
            new = {name: input_at(name, t) for name in self.inputs}
            # Every other neuron: decide from the OLD state (last tick). We build `new`
            # separately, so every neuron updates at the same moment. Updating in place
            # would let later neurons see this tick's values, which would break rule 3.
            new.update({name: n.fires(state) for name, n in self.neurons.items()})
            state = new
            # Record this tick.
            for name in self.names:
                history[name].append(state[name])
        return history


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def show(history: dict[str, list[int]], names: list[str] | None = None) -> str:
    """Format a firing history as a tick-by-tick table ('#' = fires, '.' = silent)."""
    names = names or list(history)                        # which rows to show
    steps = len(next(iter(history.values())))             # number of ticks recorded
    width = max(len(n) for n in names)                    # widest name, for alignment
    header = " " * width + " | " + " ".join(f"{t:>2}" for t in range(steps))
    rows = [f"{n:>{width}} | " + " ".join(" #" if v else " ." for v in history[n]) for n in names]
    return "\n".join([header, "-" * len(header), *rows])
