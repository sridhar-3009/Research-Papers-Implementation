# AlphaGo, explained simply

**Paper:** David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, Demis Hassabis (Google DeepMind and Google), *Mastering the game of Go with deep neural networks and tree search*, Nature 529, 484–489 (2016). (Nature is paywalled; DeepMind hosts a copy of the paper.)

**In one sentence:** AlphaGo combined **neural networks** that suggest good moves (policy networks) and judge positions (a value network) with **Monte Carlo tree search**. It was the first program to beat a professional Go player on a full board without handicap, defeating European champion Fan Hui **5–0**.

---

## 1. Why Go was so hard

- **Perfect play** means computing the optimal value v*(s) of every position by searching the game tree. That tree has about **b^d** move sequences, where b is the number of legal moves (breadth) and d is the game length (depth).
  - **Chess:** b ≈ 35, d ≈ 80.
  - **Go:** b ≈ 250, d ≈ 150, so 250¹⁵⁰ ≈ 10³⁶⁰ sequences. Exhaustive search is hopeless.
- **Two ways to shrink the search:**
  1. **Reduce depth:** stop at some position s and replace the subtree below it with an **estimate** v(s) ≈ v*(s) (a value function).
  2. **Reduce breadth:** only consider moves that a **policy** p(a|s) rates as promising.
- **Before AlphaGo,** Go programs used Monte Carlo tree search with simple handcrafted policies and reached strong amateur level.

---

## 2. The pipeline (Figure 1)

| Step | Network | Trained on | Paper's result |
|---|---|---|---|
| 1 | **SL policy** p_σ (13-layer CNN) | 30 million positions from KGS 6–9 dan human games | **57.0%** move-prediction accuracy (55.7% from the raw board only) vs 44.4% previous best |
| 2 | **Rollout policy** p_π (linear softmax over small pattern features) | the same human moves | 24.2% accuracy, but **2 µs** per move vs 3 ms for the network |
| 3 | **RL policy** p_ρ (initialised from p_σ) | self-play by policy gradient, against **randomly chosen previous versions** | won **> 80%** vs p_σ; won **85%** vs Pachi with **no search** (the previous SL net won 11%) |
| 4 | **Value network** v_θ | 30 million self-play positions of p_ρ, **one per game** | MSE 0.226 train / 0.234 test; close to rollouts with p_ρ at **15,000×** less computation |
| 5 | **Search** (APV-MCTS) | combines all of the above | 99.8% wins vs other programs; 5–0 vs Fan Hui |

### Step 3 in detail: policy gradient from self-play
- **Play a game** with the current policy against a random earlier version, and get the outcome z = +1 (win) or −1 (loss).
- **Update:** Δρ ∝ z · ∇ log p_ρ(aₜ | sₜ) for every move the current policy made. Moves from won games become more likely, moves from lost games less likely.
- **The opponent pool** stops the policy from over-specialising against its current self.

### Step 4 in detail: the overfitting trap
- **Successive positions in one game are almost identical but share the same outcome z.** Training on all positions from KGS games, the value net **memorised** games: MSE 0.19 on training vs **0.37** on test.
- **The fix:** generate 30 million self-play games and take **one position from each game**: MSE 0.226 / 0.234, with almost no overfitting.

---

## 3. The search

Each edge (s, a) of the tree stores a prior P(s, a), a visit count N(s, a) and a mean value Q(s, a). Every simulation runs four steps:

1. **Selection:** from the root, repeatedly choose
   ```
   aₜ = argmax_a [ Q(s, a) + u(s, a) ],      u(s, a) = c_puct · P(s, a) · √(Σ_b N(s, b)) / (1 + N(s, a))
   ```
   - u is large for moves with a high prior and few visits (explore). As visits grow, Q dominates (exploit).
   - The paper uses **c_puct = 5**.
2. **Expansion:** at a leaf s_L, store priors P(s_L, ·) = **p_σ**(·|s_L). The SL policy worked **better** than the RL policy here, "presumably because humans select a diverse beam of promising moves".
3. **Evaluation:** combine the value network and one fast rollout:
   ```
   V(s_L) = (1 − λ) · v_θ(s_L) + λ · z_L        λ = 0.5 was best
   ```
4. **Backup:** add 1 to N and V to the total value of every edge on the path. Q is the mean.

After all simulations, **play the most-visited move**. That is more robust than the highest Q, which may rest on few visits.

**Worked example of the PUCT score.** The root has two moves. A has P = 0.6, N = 10, Q = 0.1; B has P = 0.3, N = 0. The total is ΣN = 10, √10 = 3.16, and c_puct = 5.
- u(A) = 5 · 0.6 · 3.16 / 11 = 0.86, so the score is 0.96.
- u(B) = 5 · 0.3 · 3.16 / 1 = 4.74, so the score is 4.74.
- **B is tried next.** Its prior is lower, but it has never been visited.

**Parallelism:**
- **Asynchronous search** with **virtual loss** (n_vl = 3): a thread temporarily counts its path as losses, so other threads explore elsewhere.
- **Expansion threshold:** nodes are expanded only after 40 visits.
- **Hardware:** single-machine AlphaGo used 40 search threads, 48 CPUs and 8 GPUs. The distributed version used 1,202 CPUs and 176 GPUs, and won 77% against single-machine AlphaGo.

---

## 4. Results

- **Against other programs:** **494 of 495 games won (99.8%)**. Even giving 4 handicap stones, it won 77% / 86% / 99% against Crazy Stone / Zen / Pachi.
- **Value net vs rollouts:**
  - the value network alone (λ = 0) already beat all other Go programs;
  - the mixture λ = 0.5 was best, winning **≥ 95%** against the other variants.
  - The two evaluations are complementary: the value net approximates strong-but-slow p_ρ play, while rollouts score positions precisely with a weaker policy.
- **Fan Hui:** defeated the European champion **5–0**, the first time a program beat a professional on a full 19×19 board without handicap.

---

## 5. How our code rebuilds it

- **Why a smaller game:** Go is far too big for a laptop. We use **Connect-3 on a 4 × 5 board** (gravity: stones drop to the lowest empty cell). It is small enough to **solve exactly**: 14,166 positions in 0.06 s, and the first player wins with perfect play.
- **What that buys:** every learned component can be scored against **perfect play**. The **perfect-move rate** is the share of decisive positions (where moves differ in exact value) in which a player chooses a best move.
- **The full pipeline is implemented:**
  - **"Expert" games:** a perfect player making 25% random moves, a stand-in for strong human players;
  - **SL policy:** an MLP on board planes (own, opponent, empty, playable);
  - **rollout policy:** a linear softmax over 5 move features (wins now, blocks a win, centrality, height, open lines);
  - **RL policy:** REINFORCE self-play against an opponent pool;
  - **value network:** two data regimes;
  - **PUCT search:** c_puct = 5, SL priors, λ-mixed leaf evaluation, most-visited move.

---

## 6. What our code found

| Component | Result |
|---|---|
| SL policy | move-prediction accuracy 0.419 train / 0.274 test (low because the expert chooses randomly among equally good moves); **perfect-move rate 0.857** |
| Rollout policy | weights (wins, blocks, centre, height, open lines) = (0.99, 0.38, 0.33, 0.17, −0.10); **perfect-move rate 0.944** |
| RL policy | scores **0.587** against the SL policy (both sampling; paper > 80%); greedy perfect-move rate 0.838 |
| Value net, all positions of 1,500 games | train MSE 0.755, test **0.785** |
| Value net, one position from each of 12,538 games | train MSE 0.697, test **0.763** (predicting the mean gives 0.994) |

**Search** (50 simulations per move):

| Player | Perfect-move rate | Score vs the SL policy (40 games) |
|---|---|---|
| SL policy alone (greedy) | 0.857 | 0.400 |
| RL policy alone (greedy) | 0.838 | 0.425 |
| rollout policy alone | 0.944 | 0.775 |
| MCTS, λ = 0 (value net only) | 0.981 | 0.800 |
| MCTS, λ = 0.5 (mixed) | 0.985 | 0.825 |
| MCTS, λ = 1 (rollouts only) | **0.989** | 0.850 |

- **Search adds what the networks miss:** from 86% perfect moves (network alone) to about 98–99%.
- **Honest differences from the paper:**
  1. **The rollout policy is stronger than the network here.** In this small tactical game, "win now" and "block a win" features capture most of the skill. In Go the rollout policy was much weaker (24% vs 57% accuracy). In numpy it isn't faster either (32 µs vs 12 µs per move); AlphaGo's pattern code was 1,500× faster than its network.
  2. **So rollouts-only search came out narrowly best** (0.989 vs 0.985 mixed vs 0.981 value-only). The paper found λ = 0.5 best; our three variants differ by under one point.
  3. **RL self-play helped much less than in the paper** (0.587 vs > 80%), and it didn't improve the greedy move.
  4. **The one-position-per-game advantage is real but small** (test 0.763 vs 0.785). Our games last about 8 moves, so positions within a game are less redundant than in 150-move Go games.

**`experiments.py`:**
- **E1:** simulations × λ;
- **E2:** SL vs RL priors and c_puct;
- **E3:** value-data regimes at 1–10× size;
- **E4:** RL iterations with and without the opponent pool;
- **E5:** Connect-4 on 4 × 5 (a draw; about 2.5M positions to solve).
- Only an E4 smoke run was done here: RL after 10 / 30 iterations scored 0.50 / 0.58 vs SL.

---

## 7. Why it matters

- **AlphaGo was a landmark:** in March 2016 its successor beat Lee Sedol 4–1.
- **Its descendants simplified the recipe:**
  - **AlphaGo Zero** dropped human data and rollouts: one network, pure self-play, with search as the policy-improvement operator;
  - **AlphaZero** generalised it to chess and shogi;
  - **MuZero** learned the rules too.
- **The PUCT selection rule and "network priors + search" now appear** in many planning systems, including recent work on search with language models.

---

## 8. Check yourself

1. Why can't Go be solved by exhaustive search?
<details><summary>Answer</summary>The tree has about b^d ≈ 250¹⁵⁰ sequences, far beyond any computer.</details>

2. What do the policy network and the value network each reduce?
<details><summary>Answer</summary>The policy network reduces breadth (it focuses the search on promising moves via priors). The value network reduces depth (it evaluates a position without playing to the end).</details>

3. Compute u(s, a) for P = 0.2, N(s, a) = 3, ΣN = 15, c_puct = 5.
<details><summary>Answer</summary>5 · 0.2 · √15 / (1 + 3) = 5 · 0.2 · 3.873 / 4 ≈ 0.97.</details>

4. Why did the value network overfit when trained on all positions of KGS games?
<details><summary>Answer</summary>Consecutive positions in one game are nearly identical and share the same outcome, so the network could memorise games rather than learn to evaluate positions (MSE 0.19 train vs 0.37 test). Taking one position per self-play game made the samples independent (0.226 / 0.234).</details>

5. With λ = 0.5, v_θ = 0.4 and a rollout outcome z = −1, what is the leaf value?
<details><summary>Answer</summary>0.5 · 0.4 + 0.5 · (−1) = −0.3.</details>

6. Why did AlphaGo use the SL policy, not the stronger RL policy, for priors?
<details><summary>Answer</summary>The SL policy spreads probability over a diverse set of plausible human moves, which is useful for exploration in search. The RL policy concentrates on its single favourite move.</details>

7. Why does AlphaGo play the most-visited move rather than the one with the highest Q?
<details><summary>Answer</summary>Visit counts are robust: a move with a high Q but few visits may be a noisy estimate, while the most-visited move is the one the search kept confirming.</details>

8. What is a virtual loss?
<details><summary>Answer</summary>In parallel search, a thread temporarily adds losses to the edges on its path, so other threads are steered to different variations instead of all exploring the same one.</details>

9. In our toy, why was the rollout policy stronger than the policy network?
<details><summary>Answer</summary>Connect-3 on a small board is mostly tactical (win now, block the opponent's win), and those exact features are hard-coded into the linear rollout policy, while the network must learn them from noisy expert data.</details>

10. What changed in AlphaGo Zero?
<details><summary>Answer</summary>No human data and no rollouts. A single network with policy and value heads was trained purely from self-play, using the search's visit counts as policy targets and game outcomes as value targets.</details>
