"""The nets drawn in Figure 1 of the paper, plus the loops from Theorem X.

Each builder returns a Network. Every gate adds one tick of delay.

How to read the wiring below (every neuron has threshold 2 unless stated):
  excite={"x": 2}  ->  x connects twice: x ALONE is enough to fire this neuron.
  excite={"x": 1}  ->  x connects once: x needs a partner to reach 2.
  inhibit={"x"}    ->  x vetoes this neuron whenever x fires.
"""

from mp_neuron import Network


# ---- Figure 1a–d: the four building blocks (Theorem II) ----
# Every loop-free net in the paper is built from just these four.

def delay() -> Network:
    """Fig 1a: N2(t) = N1(t-1). Neuron 1 makes two endings on 2."""
    net = Network(["1"])
    # 2 endings reach the threshold alone, so neuron 2 simply copies neuron 1,
    # one tick late. This is the paper's S ("one tick earlier") as hardware.
    net.add("2", excite={"1": 2})
    return net


def or_gate() -> Network:
    """Fig 1b: N3(t) = N1(t-1) OR N2(t-1). Two endings each: either alone suffices."""
    net = Network(["1", "2"])
    # Each input brings 2 on its own, so ONE firing input is enough -> OR.
    net.add("3", excite={"1": 2, "2": 2})
    return net


def and_gate() -> Network:
    """Fig 1c: N3(t) = N1(t-1) AND N2(t-1). One ending each: both are needed."""
    net = Network(["1", "2"])
    # Each input brings only 1, so BOTH must fire to reach 2 -> AND.
    # Compare with or_gate: same threshold, only the ending counts differ.
    net.add("3", excite={"1": 1, "2": 1})
    return net


def and_not_gate() -> Network:
    """Fig 1d: N3(t) = N1(t-1) AND NOT N2(t-1). Neuron 2 inhibits."""
    net = Network(["1", "2"])
    # 1 can fire 3 alone (2 endings), but if 2 fires, it vetoes.
    # This is the ONLY kind of NOT the model allows (Theorem III).
    net.add("3", excite={"1": 2}, inhibit={"2"})
    return net


# ---- Figure 1e: the heat/cold illusion (pages 9-10) ----

def heat_cold() -> Network:
    """Fig 1e.

    Inputs:  1 = heat receptor, 2 = cold receptor.
    Outputs: 3 = feel heat, 4 = feel cold.
      N3(t) = N1(t-1) OR [N2(t-3) AND NOT N2(t-2)]
      N4(t) = N2(t-2) AND N2(t-1)
    """
    net = Network(["1", "2"])

    # Step 1 (a Fig 1a delay): a = "cold was on one tick ago".
    net.add("a", excite={"2": 2})

    # Step 2 (a Fig 1c AND): 4 = a AND cold = "cold two ticks in a row" -> FEEL COLD.
    net.add("4", excite={"a": 1, "2": 1})

    # Step 3 (a Fig 1d AND-NOT): b = a AND NOT cold = "cold was on, then it stopped".
    # This detects a BRIEF cold touch.
    net.add("b", excite={"a": 2}, inhibit={"2"})

    # Step 4 (a Fig 1b OR): 3 = real heat OR brief cold -> FEEL HEAT.
    # The "OR brief cold" part is the illusion.
    net.add("3", excite={"1": 2, "b": 2})
    return net


# ---- Figure 1h: temporal summation built from spatial summation (Theorem VI) ----

def two_in_a_row() -> Network:
    """Fig 1h: N2(t) = N1(t-1) AND N1(t-2), via a direct path and a delayed path."""
    net = Network(["1"])
    # d is a delay: it carries neuron 1's spike from one tick earlier.
    net.add("d", excite={"1": 2})
    # Neuron 2 gets "1 now" (direct path) and "1 a tick ago" (through d), 1 ending each.
    # Both must be on, so 2 fires only if neuron 1 fired twice in a row.
    # Spikes spread over TIME are turned into spikes arriving at the SAME time.
    net.add("2", excite={"1": 1, "d": 1})
    return net


# ---- Figure 1i: learning replaced by a loop (Theorem VII) ----

def learned_association() -> Network:
    """Fig 1i (bottom): 2 always fires 3; 1 fires 3 only after 1 and 2 fired together.

    L is a memory loop: it switches on when 1 and 2 coincide, then feeds itself forever.
    """
    net = Network(["1", "2"])

    # L (the memory):
    #   - 1 and 2 give 1 ending each -> both together turn L on  (the "pairing")
    #   - L gives ITSELF 2 endings    -> once on, it keeps itself on forever (a loop)
    net.add("L", excite={"1": 1, "2": 1, "L": 2})

    # 3 (the output):
    #   - 2 gives 2 endings           -> 2 alone always fires 3
    #   - 1 and L give 1 ending each  -> 1 fires 3 only when L is on, i.e. after pairing
    # This is like Pavlov's dog: 2 = food, 1 = bell. After pairing, the bell alone works.
    net.add("3", excite={"2": 2, "1": 1, "L": 1})
    return net


# ---- Theorem X: the three loops that give nets memory and rhythm ----

def latch(with_reset: bool = False) -> Network:
    """'Did P ever happen?'  M(t) = P(t-1) OR M(t-1).

    With a reset input R that inhibits M, this is the set/reset latch from the remark
    after Theorem VII (spontaneous activity started by one input, stopped by another).
    """
    net = Network(["P", "R"] if with_reset else ["P"])
    # P turns M on (2 endings). M feeds itself 2 endings, so it stays on forever.
    # If there's a reset input R, its veto breaks the loop and M goes off.
    net.add("M", excite={"P": 2, "M": 2}, inhibit={"R"} if with_reset else set())
    return net


def always_so_far() -> Network:
    """'Has P held at every tick?'  A(t) = P(t-1) AND A(t-1). Start it with A = 1."""
    net = Network(["P"])
    # A needs BOTH P and its own previous value (1 ending each).
    # The first time P fails, A goes off, and it can never come back:
    # it would need itself to be on already. Must start with A = 1 (initial={"A": 1}).
    net.add("A", excite={"P": 1, "A": 1})
    return net


def clock(n: int) -> Network:
    """A ring of n neurons passing one spike around. Start it with c0 = 1."""
    net = Network([])   # no inputs at all: it runs on its own once started
    for i in range(n):
        # Each neuron copies the one before it; (i - 1) % n makes the last one feed
        # the first, which closes the ring. One spike goes around forever, so each
        # neuron fires once every n ticks.
        net.add(f"c{i}", excite={f"c{(i - 1) % n}": 2})
    return net
