"""TensorFlow OSDI experiments (Abadi et al. 2016): sweeps of the replication simulator and the sparse / sampled
softmax mechanisms, plus real distributed TensorFlow when available.

  E1  Figure 8 sweep: n in {25, 50, 100, 200} workers x backups b in {0..8} x straggler models (rate, slowdown) x PS
      cost per message: which b minimises step time and which maximises normalised speedup.
  E2  Figure 7: throughput (gradients/s) and step-time distributions (median, 90th, 99th percentile) for async and sync
      with 25 ... 200 workers.
  E3  Staleness vs convergence: async / sync / sync+backups on the logistic-regression problem over learning rates
      {0.1, 0.5, 2.0} (large rates are where stale asynchronous updates should start to hurt).
  E4  Sampled softmax: vocabulary {10k, 100k} x samples {64, 512, 4096}: work per step and convergence of the
      full-softmax loss.
  E5  Real TensorFlow (needs `tensorflow`): a ParameterServerStrategy-free stand-in -- tf.distribute.MirroredStrategy
      on all local devices vs one device, examples/s for a small MLP (synchronous all-reduce replication).

!! E5 needs `tensorflow` (GPUs for a meaningful result); E1-E4 take minutes on a CPU.
       python3 experiments.py --quick
       python3 experiments.py --only e1
       python3 experiments.py --report-only
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import tfsys as S

HERE = Path(__file__).parent


def e1(a):
    out = []
    for n in a.ns:
        for p, slow in a.stragglers:
            for ps in (0.0, 0.005, 0.01):
                c = S.backup_worker_curve(n=n, max_b=a.max_b, steps=a.steps, p_straggle=p, slow=slow, ps_cost=ps)
                out.append({"n": n, "straggle p": p, "slow": slow, "ps cost": ps,
                            "best b (time)": min(c, key=lambda b: c[b]["median step"]),
                            "best b (normalised)": max(c, key=lambda b: c[b]["normalised speedup"]),
                            "curve": c})
    return out


def e2(a):
    rng = np.random.default_rng(0)
    out = {}
    for n in a.ns:
        sync = [S.sync_step_time(rng, n, 0) for _ in range(a.steps)]
        asyn = [S.worker_times(rng, 1)[0] for _ in range(a.steps)]
        out[n] = {"sync step p50/p90/p99": np.percentile(sync, [50, 90, 99]).tolist(),
                  "async step p50/p90/p99": np.percentile(asyn, [50, 90, 99]).tolist(),
                  "sync gradients/s": n / float(np.mean(sync)), "async gradients/s": n / float(np.mean(asyn))}
    return out


def e3(a):
    out = {}
    for lr in a.lrs:
        for label, mode, b in (("async", "async", 0), ("sync", "sync", 0), ("sync+3", "sync", 3)):
            r = S.train_replicated(mode, b=b, lr=lr, wall=a.wall)
            out[f"lr={lr}, {label}"] = {"final loss": r["final loss"], "updates": r["updates"],
                                        "staleness": r["mean staleness"]}
    return out


def e4(a):
    rng = np.random.default_rng(0)
    out = {}
    for V in a.vocabs:
        d = 32
        W_true = rng.standard_normal((V, d)) / np.sqrt(d)
        H = rng.standard_normal((2048, d))
        Y = (H @ W_true.T + 2 * rng.gumbel(size=(2048, V))).argmax(1)
        for k in a.samples:
            W = np.zeros((V, d))
            t = time.time()
            for _ in range(a.ss_steps):
                idx = rng.integers(0, len(Y), 128)
                _, classes, gW, _ = S.sampled_softmax_grad(H[idx], W, Y[idx], min(k, V - 1), rng)
                W[classes] -= 2.0 * gW
            out[f"V={V}, k={k}"] = {"seconds": time.time() - t, "loss": S.full_softmax_loss(H[:512], W, Y[:512]),
                                    "work ratio V/(k+128)": V / (k + 128)}
    return out


def e5(a):
    import tensorflow as tf
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20000, 64)).astype("float32")
    y = (X[:, 0] > 0).astype("int32")
    out = {}
    for label, strategy in (("one device", tf.distribute.OneDeviceStrategy("/cpu:0")),
                            ("mirrored (all local devices)", tf.distribute.MirroredStrategy())):
        with strategy.scope():
            m = tf.keras.Sequential([tf.keras.layers.Dense(256, activation="relu"), tf.keras.layers.Dense(2)])
            m.compile("sgd", tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        t = time.time()
        m.fit(X, y, batch_size=512, epochs=a.tf_epochs, verbose=0)
        out[label] = {"examples/s": len(X) * a.tf_epochs / (time.time() - t), "replicas": strategy.num_replicas_in_sync}
    return out


def report(Rs, a):
    Ls = ["# Results", "", "(QUICK run)" if a.quick else "", ""]
    for k in sorted(Rs, key=str):
        Ls += [f"## {str(k).upper()}", "", "```", json.dumps(Rs[k], indent=1, default=float)[:20000], "```", ""]
    (HERE / "results.md").write_text("\n".join(Ls) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=[f"e{i}" for i in range(1, 6)])
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    a.ns, a.stragglers, a.max_b, a.steps = (25, 50, 100, 200), ((0.02, 1.5), (0.06, 1.3), (0.1, 1.2)), 8, 3000
    a.lrs, a.wall, a.vocabs, a.samples, a.ss_steps, a.tf_epochs = (0.1, 0.5, 2.0), 60.0, (10000, 100000), (64, 512, 4096), 200, 3
    if a.quick:
        a.ns, a.stragglers, a.max_b, a.steps = (25,), ((0.06, 1.3),), 2, 50
        a.lrs, a.wall, a.vocabs, a.samples, a.ss_steps, a.tf_epochs = (0.5,), 3.0, (1000,), (64,), 3, 1
    path = HERE / "results.json"
    Rs = json.loads(path.read_text()) if path.exists() else {}
    if not a.report_only:
        t0 = time.time()
        for name, fn in (("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4), ("e5", e5)):
            if a.only in (None, name):
                try:
                    Rs[name] = fn(a)
                except ImportError as e:
                    Rs[name] = {"skipped": f"missing package: {e}"}
                path.write_text(json.dumps(Rs, default=float))
        print(f"done in {time.time() - t0:.0f}s")
    report(Rs, a)


if __name__ == "__main__":
    main()
