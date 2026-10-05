"""Mastering the Game of Go with Deep Neural Networks and Tree Search -- AlphaGo (Silver et al., Nature 2016).

The AlphaGo pipeline (Figure 1), rebuilt end to end on a game small enough to SOLVE exactly (Connect-K on a small
board; default 4 rows x 5 columns, 3 in a row), so every network can be checked against perfect play v*(s):

  1  SL policy p_sigma      a network trained to predict 'expert' moves (here: a noisy perfect player stands in for
                            the KGS 6-9 dan humans)
  2  rollout policy p_pi    a fast LINEAR softmax over a few hand-made move features (cheap, weaker)
  3  RL policy p_rho        p_sigma improved by policy-gradient self-play against randomly chosen PREVIOUS versions
                            (an opponent pool, to avoid overfitting to the current policy)
  4  value network v_theta  regression of the game outcome z on positions from p_rho self-play. Using all positions of
                            each game overfits (successive positions are strongly correlated); AlphaGo used ONE position
                            per game
  5  APV-MCTS               select a = argmax Q(s, a) + u(s, a),  u = c_puct P(s, a) sqrt(sum_b N(s, b)) / (1 + N(s, a)),
                            priors P from p_sigma, leaf value V = (1 - lambda) v_theta(s_L) + lambda z_L (z_L from a
                            p_pi rollout), back up mean values, play the MOST VISITED move
"""

import numpy as np
from functools import lru_cache

# ----------------------------------------------------------------------------------------------- the game

class ConnectK:
    """Gravity game: players drop stones into columns; K in a row (any direction) wins; full board = draw.
    Board: tuple of R*C ints (0 empty, 1, 2), row-major with row 0 at the bottom."""

    def __init__(self, rows=4, cols=5, k=3):
        self.R, self.C, self.K = rows, cols, k
        lines = []
        for r in range(rows):
            for c in range(cols):
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    cells = [(r + i * dr, c + i * dc) for i in range(k)]
                    if all(0 <= a < rows and 0 <= b < cols for a, b in cells):
                        lines.append(tuple(a * cols + b for a, b in cells))
        self.lines = lines
        self.lines_at = [[L for L in lines if i in L] for i in range(rows * cols)]
        self.solve = lru_cache(maxsize=None)(self._solve)

    def empty(self):
        return tuple([0] * (self.R * self.C))

    def legal(self, b):
        return [c for c in range(self.C) if b[(self.R - 1) * self.C + c] == 0]

    def drop_index(self, b, c):
        for r in range(self.R):
            if b[r * self.C + c] == 0:
                return r * self.C + c
        return None

    def play(self, b, c, player):
        i = self.drop_index(b, c)
        nb = b[:i] + (player,) + b[i + 1:]
        won = any(all(nb[j] == player for j in L) for L in self.lines_at[i])
        return nb, won, (not won and all(x != 0 for x in nb))

    def to_move(self, b):
        n = sum(1 for x in b if x)
        return 1 if n % 2 == 0 else 2

    def _solve(self, b):
        """Negamax value for the player to move: +1 win, 0 draw, -1 loss (perfect play)."""
        p = self.to_move(b)
        best = -1
        for c in self.legal(b):
            nb, won, draw = self.play(b, c, p)
            v = 1 if won else (0 if draw else -self.solve(nb))
            if v > best:
                best = v
                if best == 1:
                    break
        return best if self.legal(b) else 0

    def move_values(self, b):
        """Exact value (for the mover) of every legal move."""
        p = self.to_move(b)
        out = {}
        for c in self.legal(b):
            nb, won, draw = self.play(b, c, p)
            out[c] = 1 if won else (0 if draw else -self.solve(nb))
        return out

# ----------------------------------------------------------------------------------------------- features and networks

def board_features(g, b):
    """Planes from the mover's point of view: own stones, opponent stones, empty, and 'playable here' cells."""
    p = g.to_move(b)
    a = np.array(b)
    own, opp, emp = (a == p), (a == 3 - p), (a == 0)
    playable = np.zeros(len(b), bool)
    for c in g.legal(b):
        playable[g.drop_index(b, c)] = True
    return np.concatenate([own, opp, emp, playable]).astype(float)


def legal_mask(g, b):
    m = np.full(g.C, -1e9)
    m[g.legal(b)] = 0.0
    return m


class Net:
    """One hidden layer (tanh). kind='policy': softmax over columns (masked); kind='value': tanh scalar."""

    def __init__(self, d_in, d_out, hidden=64, seed=0, kind="policy"):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 1 / np.sqrt(d_in), (d_in, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, 0.1 / np.sqrt(hidden), (hidden, d_out)); self.b2 = np.zeros(d_out)
        self.kind = kind
        self.m = [np.zeros_like(x) for x in self.params()]
        self.v = [np.zeros_like(x) for x in self.params()]
        self.t = 0

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def copy(self):
        n = Net.__new__(Net)
        n.W1, n.b1, n.W2, n.b2, n.kind = self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(), self.kind
        n.m = [np.zeros_like(x) for x in n.params()]; n.v = [np.zeros_like(x) for x in n.params()]; n.t = 0
        return n

    def forward(self, X, mask=None):
        h = np.tanh(X @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        if self.kind == "value":
            return h, np.tanh(z[:, 0])
        z = z + (0 if mask is None else mask)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        return h, p / p.sum(1, keepdims=True)

    def step(self, X, dz, lr):
        """Adam step given dLoss/d(pre-activation output) dz."""
        h = np.tanh(X @ self.W1 + self.b1)
        dh = (dz @ self.W2.T) * (1 - h ** 2)
        grads = [X.T @ dh, dh.sum(0), h.T @ dz, dz.sum(0)]
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params(), grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            p -= lr * (self.m[i] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[i] / (1 - 0.999 ** self.t)) + 1e-8)

    def policy(self, g, b):
        return self.forward(board_features(g, b)[None], legal_mask(g, b)[None])[1][0]

    def value(self, g, b):
        return float(self.forward(board_features(g, b)[None])[1][0])

# ----------------------------------------------------------------------------------------------- 1. expert data, SL policy

def expert_move(g, b, rng, eps=0.25):
    """A strong but imperfect 'expert': a perfect move, except with probability eps a random legal move."""
    if rng.random() < eps:
        return int(rng.choice(g.legal(b)))
    mv = g.move_values(b)
    best = max(mv.values())
    return int(rng.choice([c for c, v in mv.items() if v == best]))


def expert_games(g, n_games, rng, eps=0.25):
    """(features, mask, expert move) for every position of n_games expert-vs-expert games."""
    X, M, Y, B = [], [], [], []
    for _ in range(n_games):
        b, p = g.empty(), 1
        while True:
            c = expert_move(g, b, rng, eps)
            X.append(board_features(g, b)); M.append(legal_mask(g, b)); Y.append(c); B.append(b)
            b, won, draw = g.play(b, c, p)
            if won or draw:
                break
            p = 3 - p
    return np.array(X), np.array(M), np.array(Y), B


def train_policy(net, X, M, Y, epochs=30, lr=3e-3, batch=128, rng=None):
    rng = rng or np.random.default_rng(0)
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(Y)), max(1, len(Y) // batch)):
            _, p = net.forward(X[idx], M[idx])
            dz = (p - np.eye(net.b2.shape[0])[Y[idx]]) / len(idx)
            net.step(X[idx], dz, lr)
    return net


def accuracy(net, X, M, Y):
    _, p = net.forward(X, M)
    return float((p.argmax(1) == Y).mean())


def optimal_move_rate(agent_or_net, g, boards, rng=None):
    """Share of DECISIVE positions (where moves differ in exact value) in which the top move (for a network) or the
    chosen move (for an agent function) is a perfect move."""
    rng = rng or np.random.default_rng(0)
    ok = n = 0
    for b in boards:
        mv = g.move_values(b)
        if len(set(mv.values())) < 2:
            continue
        best = max(mv.values())
        c = int(np.argmax(agent_or_net.policy(g, b))) if isinstance(agent_or_net, Net) else agent_or_net(g, b, rng)
        ok += mv[c] == best
        n += 1
    return ok / max(n, 1)

# ----------------------------------------------------------------------------------------------- 2. fast rollout policy

def rollout_features(g, b, c):
    """Small hand-made 'pattern' features of a move: wins now, blocks the opponent's win, centrality, height,
    number of own lines through the cell that are still open."""
    p = g.to_move(b)
    i = g.drop_index(b, c)
    _, won, _ = g.play(b, c, p)
    _, block, _ = g.play(b, c, 3 - p)
    centre = 1 - abs(c - (g.C - 1) / 2) / ((g.C - 1) / 2)
    height = (i // g.C) / (g.R - 1)
    open_lines = sum(1 for L in g.lines_at[i] if all(b[j] in (0, p) for j in L)) / 4
    return np.array([won, block, centre, height, open_lines], float)


class RolloutPolicy:
    """Linear softmax over rollout_features -- the analogue of AlphaGo's p_pi."""

    def __init__(self):
        self.w = np.zeros(5)

    def probs(self, g, b):
        cs = g.legal(b)
        F = np.array([rollout_features(g, b, c) for c in cs])
        z = F @ self.w
        e = np.exp(z - z.max())
        return cs, e / e.sum(), F

    def fit(self, g, boards, moves, epochs=100, lr=1.0):
        """Maximum likelihood by gradient ascent; each board's move features are computed once."""
        data = []
        for b, y in zip(boards, moves):
            cs = g.legal(b)
            data.append((np.array([rollout_features(g, b, c) for c in cs]), cs.index(y)))
        for _ in range(epochs):
            grad = np.zeros(5)
            for F, k in data:
                z = F @ self.w
                p = np.exp(z - z.max()); p /= p.sum()
                grad += F[k] - p @ F
            self.w += lr * grad / len(data)
        return self

    def act(self, g, b, rng):
        cs, p, _ = self.probs(g, b)
        return int(cs[rng.choice(len(cs), p=p)])

    def accuracy(self, g, boards, moves):
        return float(np.mean([self.probs(g, b)[0][int(np.argmax(self.probs(g, b)[1]))] == y
                              for b, y in zip(boards, moves)]))

# ----------------------------------------------------------------------------------------------- playing games

def play_game(g, agent1, agent2, rng, opening=0):
    """agent(g, b, rng) -> column. `opening` random moves first (for variety). Returns +1 if player 1 wins, -1 if
    player 2 wins, 0 for a draw, and the list of (board, move, player)."""
    b, p, hist = g.empty(), 1, []
    for t in range(g.R * g.C):
        c = int(rng.choice(g.legal(b))) if t < opening else (agent1 if p == 1 else agent2)(g, b, rng)
        hist.append((b, c, p))
        b, won, draw = g.play(b, c, p)
        if won:
            return (1 if p == 1 else -1), hist
        if draw:
            return 0, hist
        p = 3 - p
    return 0, hist


def policy_agent(net, greedy=False):
    def act(g, b, rng):
        pr = net.policy(g, b)
        return int(np.argmax(pr)) if greedy else int(rng.choice(g.C, p=pr))
    return act


def perfect_agent(g, b, rng):
    mv = g.move_values(b)
    best = max(mv.values())
    return int(rng.choice([c for c, v in mv.items() if v == best]))


def random_agent(g, b, rng):
    return int(rng.choice(g.legal(b)))


def match(g, a, b_agent, games, rng, opening=2):
    """a plays half the games as player 1 and half as player 2. Returns a's score (win 1, draw 0.5)."""
    score = 0.0
    for i in range(games):
        if i % 2 == 0:
            z, _ = play_game(g, a, b_agent, rng, opening)
            score += (z == 1) + 0.5 * (z == 0)
        else:
            z, _ = play_game(g, b_agent, a, rng, opening)
            score += (z == -1) + 0.5 * (z == 0)
    return score / games

# ----------------------------------------------------------------------------------------------- 3. RL policy (self-play)

def rl_self_play(g, sl_net, iters=40, games_per_iter=32, lr=1e-3, pool_every=5, seed=0):
    """REINFORCE: the current policy plays against a RANDOMLY CHOSEN earlier version from a pool (added every
    `pool_every` iterations); its moves get gradient z * grad log p(a|s) with z = +1 / -1 / 0 for win / loss / draw."""
    rng = np.random.default_rng(seed)
    net = sl_net.copy()
    pool = [sl_net.copy()]
    for it in range(iters):
        X, M, Y, Z = [], [], [], []
        for k in range(games_per_iter):
            opp = pool[rng.integers(len(pool))]
            me_first = k % 2 == 0
            a1, a2 = (policy_agent(net), policy_agent(opp)) if me_first else (policy_agent(opp), policy_agent(net))
            z, hist = play_game(g, a1, a2, rng)
            me = 1 if me_first else 2
            zme = z if me_first else -z
            for b, c, p in hist:
                if p == me:
                    X.append(board_features(g, b)); M.append(legal_mask(g, b)); Y.append(c); Z.append(zme)
        X, M, Y, Z = map(np.array, (X, M, Y, Z))
        _, pr = net.forward(X, M)
        dz = -Z[:, None] * (np.eye(g.C)[Y] - pr) / len(Y)            # ascent on z log p  ->  descent on -z log p
        net.step(X, dz, lr)
        if (it + 1) % pool_every == 0:
            pool.append(net.copy())
    return net

# ----------------------------------------------------------------------------------------------- 4. value network

def self_play_positions(g, net, n_games, rng, one_per_game):
    """Positions from p_rho self-play, labelled with the final outcome z from the mover's point of view.
    one_per_game=True keeps ONE random position per game (AlphaGo's fix for correlated positions)."""
    X, Z, B = [], [], []
    for _ in range(n_games):
        z, hist = play_game(g, policy_agent(net), policy_agent(net), rng, opening=1)
        picks = [hist[rng.integers(len(hist))]] if one_per_game else hist
        for b, c, p in picks:
            X.append(board_features(g, b)); B.append(b)
            Z.append(z if p == 1 else -z)
    return np.array(X), np.array(Z, float), B


def train_value(X, Z, epochs=60, lr=3e-3, seed=0, batch=128):
    net = Net(X.shape[1], 1, seed=seed, kind="value")
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(Z)), max(1, len(Z) // batch)):
            _, v = net.forward(X[idx])
            dz = (2 * (v - Z[idx]) * (1 - v ** 2) / len(idx))[:, None]
            net.step(X[idx], dz, lr)
    return net


def mse(net, X, Z):
    return float(((net.forward(X)[1] - Z) ** 2).mean())

# ----------------------------------------------------------------------------------------------- 5. APV-MCTS (single thread)

class Node:
    __slots__ = ("P", "N", "W", "children", "terminal")

    def __init__(self, P):
        self.P, self.N, self.W, self.children, self.terminal = P, 0, 0.0, None, None


def rollout_value(g, b, rp, rng):
    """Play to the end with the fast rollout policy; outcome for the player to move at b."""
    me = g.to_move(b)
    p = me
    while True:
        cs = g.legal(b)
        if not cs:
            return 0.0
        c = rp.act(g, b, rng)
        b, won, draw = g.play(b, c, p)
        if won:
            return 1.0 if p == me else -1.0
        if draw:
            return 0.0
        p = 3 - p


def mcts(g, b, prior_net, value_net, rp, sims=100, lam=0.5, c_puct=5.0, rng=None):
    """PUCT search from board b. Each node stores, per child move, N and W (total value from the point of view of the
    player choosing that move). Leaf value V = (1 - lam) v_theta + lam z_rollout. Returns the most visited move and
    the visit counts."""
    rng = rng or np.random.default_rng(0)
    root = Node(1.0)

    def expand(node, board):
        pr = prior_net.policy(g, board)
        node.children = {c: Node(float(pr[c])) for c in g.legal(board)}

    def evaluate(board):
        v = value_net.value(g, board) if lam < 1 else 0.0
        z = rollout_value(g, board, rp, rng) if lam > 0 else 0.0
        return (1 - lam) * v + lam * z

    expand(root, b)
    for _ in range(sims):
        node, board, path = root, b, []
        value = None
        while True:
            total = sum(ch.N for ch in node.children.values())
            best, best_s = None, -1e9
            for c, ch in node.children.items():
                q = ch.W / ch.N if ch.N else 0.0
                u = c_puct * ch.P * np.sqrt(total + 1) / (1 + ch.N)
                if q + u > best_s:
                    best, best_s = c, q + u
            ch = node.children[best]
            path.append(ch)
            board, won, draw = g.play(board, best, g.to_move(board))
            if won or draw:
                value = 1.0 if won else 0.0                           # from the mover's (chooser's) view
                break
            if ch.children is None:                                   # leaf: expand + evaluate
                expand(ch, board)
                value = -evaluate(board)                              # evaluate() is for the NEXT player
                break
            node = ch
        for ch in reversed(path):                                     # back up, flipping perspective each ply
            ch.N += 1
            ch.W += value
            value = -value
    visits = {c: ch.N for c, ch in root.children.items()}
    return max(visits, key=visits.get), visits


def mcts_agent(prior_net, value_net, rp, sims=100, lam=0.5, c_puct=5.0):
    def act(g, b, rng):
        return mcts(g, b, prior_net, value_net, rp, sims, lam, c_puct, rng)[0]
    return act


REPORTED = {
    "SL policy": "13-layer policy network trained on 30 million positions from the KGS Go Server (6-9 dan); 57.0% move "
                 "prediction accuracy with all features (55.7% with raw board and history), vs 44.4% state of the art",
    "rollout policy": "linear softmax of small pattern features: 24.2% accuracy, 2 microseconds per move vs 3 ms for "
                      "the policy network",
    "RL policy": "won more than 80% of games against the SL policy; without any search won 85% against Pachi "
                 "(100,000 simulations per move), where the previous SL-only state of the art won 11%",
    "value network": "trained on all positions of KGS games it memorised outcomes (MSE 0.19 train vs 0.37 test); on a "
                     "new self-play set of 30 million positions, each from a SEPARATE game: 0.226 train / 0.234 test; "
                     "approaches Monte Carlo rollouts with p_rho using 15,000 times less computation",
    "search": "a_t = argmax Q + u, u = c_puct P sqrt(sum N) / (1 + N); priors from the SL policy (it worked better "
              "than the RL policy); leaf value (1 - lambda) v_theta + lambda z_L; lambda = 0.5 best (>= 95% wins vs "
              "other variants); c_puct = 5, virtual loss 3, expansion threshold 40; plays the most visited move",
    "results": "99.8% wins against other Go programs (494 of 495); with 4 handicap stones 77% / 86% / 99% vs Crazy "
               "Stone / Zen / Pachi; distributed AlphaGo (40 search threads, 1,202 CPUs, 176 GPUs) won 77% vs "
               "single-machine AlphaGo (48 CPUs, 8 GPUs); beat European champion Fan Hui 5-0",
}
