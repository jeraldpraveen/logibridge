"""Task F2 - five-metric benchmark of the three variants (Lab 2 methodology).

Per variant, single-sample inference (as deployed: one window every 10 s):
  1. mean latency (ms)   - 200 timed runs after 10 warm-up runs (excluded)
  2. p95 latency (ms)
  3. model file size (KiB = bytes / 1024)
  4. accuracy on the held-out validation set (%), plus Class 2 (Critical) recall + support
  5. energy per inference (mJ): E = P x t, P = TDP x CPU% / 100
     CPU% = psutil process CPU time over a separate 5 s loop of the same workload / wall time
     / logical cores (this process's share of the package; 200 runs alone are too short for
     Windows CPU accounting); TDP = laptop estimate (--tdp, default
     28 W for the i7-1165G7 used here). This is an estimate, not a power-meter reading.

Timing covers what the service does per window: quantise input (INT8 models),
set_tensor, invoke, read output. invoke() alone is also reported, because at ~2 us the
interpreter call is a small part of the Python-side cost.

Repetition: the 10 + 200 protocol is one TRIAL. Differences between variants are a few
microseconds, which a single trial in a fixed order cannot resolve (CPU frequency and cache
state drift between models). TRIALS trials are run with the variant order rotated, and the
MEDIAN across trials of each trial's mean / p95 is reported (min-max range kept in JSON).
Writes optimisation/results/benchmark_results.csv (+ benchmark_results.json with per-trial
values and model SHA-256).
"""
import hashlib
import json
import argparse
import csv
import os
import time
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import psutil
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "training" / "models"
OUT = ROOT / "optimisation" / "results"
VARIANTS = [
    ("M1 FP32 baseline", "m1_fp32.tflite"),
    ("M2 PTQ INT8", "m2_ptq_int8.tflite"),
    ("M3 Pruned + PTQ INT8", "m3_pruned_ptq_int8.tflite"),
]
WARMUP, RUNS = 10, 200
CPU_SAMPLE_S = 5.0
TRIALS = 5


class Runner:
    def __init__(self, path):
        self.it = tf.lite.Interpreter(model_path=str(path), num_threads=1)
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        self.o = self.it.get_output_details()[0]
        self.int8 = self.i["dtype"] == np.int8

    def __call__(self, x):
        x = x.reshape(1, -1).astype(np.float32)
        if self.int8:
            s, z = self.i["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        self.it.set_tensor(self.i["index"], x)
        self.it.invoke()
        return int(np.argmax(self.it.get_tensor(self.o["index"]).copy()))


def one_trial(fname, Xva):
    """One spec trial: fresh interpreter, 10 warm-up (excluded), 200 timed full calls,
    then 200 timed invoke()-only calls."""
    r = Runner(MODELS / fname)
    for k in range(WARMUP):
        r(Xva[k % len(Xva)])
    lat = np.empty(RUNS)
    for k in range(RUNS):
        x = Xva[k % len(Xva)]
        t0 = time.perf_counter_ns()
        r(x)
        lat[k] = (time.perf_counter_ns() - t0) / 1e6
    inv = np.empty(RUNS)
    for k in range(RUNS):
        t0 = time.perf_counter_ns()
        r.it.invoke()
        inv[k] = (time.perf_counter_ns() - t0) / 1e6
    return float(lat.mean()), float(np.percentile(lat, 95)), float(inv.mean())


def bench(name, fname, Xva, yva, tdp, trials):
    path = MODELS / fname
    r = Runner(path)
    means = [t[0] for t in trials]
    p95s = [t[1] for t in trials]
    invs = [t[2] for t in trials]
    # (b) CPU share: 200 runs finish in well under Windows' CPU-accounting resolution
    # (~15.6 ms ticks), so utilisation is sampled over a separate loop of the SAME workload
    # lasting CPU_SAMPLE_S seconds. Process CPU time / wall time / logical cores = this
    # process's share of the package (other processes excluded).
    proc = psutil.Process()
    c0, w0, n = proc.cpu_times(), time.perf_counter(), 0
    while time.perf_counter() - w0 < CPU_SAMPLE_S:
        r(Xva[n % len(Xva)])
        n += 1
    c1, wall = proc.cpu_times(), time.perf_counter() - w0
    cpu_s = (c1.user - c0.user) + (c1.system - c0.system)
    cpu = 100.0 * cpu_s / wall / psutil.cpu_count()
    pred = np.array([r(x) for x in Xva])
    crit = yva == 2
    mean_ms = float(np.median(means))
    power_w = tdp * cpu / 100.0
    return {
        "variant": name,
        "file": fname,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mean_latency_ms": round(mean_ms, 4),
        "p95_latency_ms": round(float(np.median(p95s)), 4),
        "invoke_only_mean_ms": round(float(np.median(invs)), 4),
        "mean_latency_trials_ms": [round(m, 4) for m in means],
        "p95_latency_trials_ms": [round(m, 4) for m in p95s],
        "size_KiB": round(path.stat().st_size / 1024, 2),
        "val_accuracy_pct": round(float((pred == yva).mean()) * 100, 2),
        "critical_recall_pct": round(float((pred[crit] == 2).mean()) * 100, 2),
        "critical_support": int(crit.sum()),
        "cpu_pct": round(cpu, 2),
        "cpu_sample_runs": n,
        "cpu_sample_s": round(wall, 2),
        "power_W_est": round(power_w, 2),
        "energy_mJ": round(power_w * mean_ms, 5),        # W x ms = mJ
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdp", type=float, default=28.0, help="laptop TDP estimate in W")
    a = ap.parse_args()
    d = np.load(ROOT / "training" / "dataset.npz")
    Xva, yva = d["X_val"], d["y_val"]
    files = [f for _, f in VARIANTS]
    trials = {f: [] for f in files}
    for t in range(TRIALS):                       # rotate the order each trial
        order = files[t % len(files):] + files[:t % len(files)]
        for f in order:
            trials[f].append(one_trial(f, Xva))
    rows = [bench(n, f, Xva, yva, a.tdp, trials[f]) for n, f in VARIANTS]
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"tdp_W": a.tdp, "warmup": WARMUP, "runs": RUNS, "trials": TRIALS,
               "cpu_sample_s": CPU_SAMPLE_S, "val_n": int(len(yva)),
               "dataset_sha256": hashlib.sha256((ROOT / "training" / "dataset.npz").read_bytes()).hexdigest(),
               "variants": rows}, open(OUT / "benchmark_results.json", "w"), indent=2)
    cols = ["variant", "file", "mean_latency_ms", "p95_latency_ms", "size_KiB", "val_accuracy_pct",
            "energy_mJ", "critical_recall_pct", "critical_support", "invoke_only_mean_ms",
            "cpu_pct", "power_W_est", "sha256"]
    with open(OUT / "benchmark_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"TDP estimate {a.tdp} W | {TRIALS} trials x ({WARMUP} warm-up + {RUNS} timed), "
          f"order rotated, median reported | val n={len(yva)}")
    for r in rows:
        print(f"{r['variant']:22s} mean {r['mean_latency_ms'] * 1000:6.2f} us  p95 {r['p95_latency_ms'] * 1000:6.2f} us  "
              f"invoke {r['invoke_only_mean_ms'] * 1000:5.2f} us  {r['size_KiB']:.2f} KiB  "
              f"acc {r['val_accuracy_pct']:.2f}%  crit {r['critical_recall_pct']:.2f}% (n={r['critical_support']})  "
              f"CPU {r['cpu_pct']}%  E {r['energy_mJ'] * 1000:.2f} uJ")
        print(f"{'':22s} per-trial means (us): {[round(m * 1000, 2) for m in r['mean_latency_trials_ms']]}")
