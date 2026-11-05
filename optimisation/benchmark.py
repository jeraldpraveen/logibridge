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
set_tensor, invoke, read output.
Writes optimisation/results/benchmark_results.csv
"""
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


def bench(name, fname, Xva, yva, tdp):
    path = MODELS / fname
    r = Runner(path)
    for k in range(WARMUP):                      # warm-up, excluded
        r(Xva[k % len(Xva)])
    # (a) latency: exactly RUNS timed single-sample inferences
    lat = np.empty(RUNS)
    for k in range(RUNS):
        x = Xva[k % len(Xva)]
        t0 = time.perf_counter_ns()
        r(x)
        lat[k] = (time.perf_counter_ns() - t0) / 1e6
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
    mean_ms = float(lat.mean())
    power_w = tdp * cpu / 100.0
    return {
        "variant": name,
        "file": fname,
        "mean_latency_ms": round(mean_ms, 4),
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 4),
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
    rows = [bench(n, f, Xva, yva, a.tdp) for n, f in VARIANTS]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "benchmark_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"TDP estimate {a.tdp} W | {WARMUP} warm-up + {RUNS} timed runs | val n={len(yva)}")
    for r in rows:
        print(f"{r['variant']:24s} mean {r['mean_latency_ms']:.4f} ms  p95 {r['p95_latency_ms']:.4f} ms  "
              f"{r['size_KiB']:.2f} KiB  acc {r['val_accuracy_pct']:.2f}%  "
              f"critRecall {r['critical_recall_pct']:.2f}% (n={r['critical_support']})  "
              f"CPU {r['cpu_pct']}%  E {r['energy_mJ']:.5f} mJ")
