"""Builds monitoring/reference_dist.json from 300 fresh clean Normal-class windows (Task E1).

Fresh data, not the training set: N_TRUCKS simulator instances (--anomaly none, real time) publish
to the local broker in parallel; each truck has its own StreamFeaturizer (same preprocessing as
the edge service) and every window goes through the committed model with the edge service's own
Model.predict. Parallel trucks only shorten the wall-clock time: each stream is an independent
clean stream with the brief's parameters, and the simulator is NOT sped up (--speedup would
compress wall-clock timestamps and distort the window features, e.g. the rate of change).

Run from the repo root:  .venv\\Scripts\\python monitoring\\build_reference.py
Writes monitoring/reference_dist.json and monitoring/results/reference_windows.json.
"""
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "monitoring"))
import inference_service as svc   # noqa: E402  (Model, preprocessing, runtime)
import psi as P                   # noqa: E402

pp = svc.pp
N_WINDOWS, N_TRUCKS = 300, 5
MODEL = ROOT / "inference" / "model.tflite"          # committed M2, the deployed model
DURATION = 30 + (N_WINDOWS // N_TRUCKS + 4) * 10     # first window at 30 s, then one per 10 s


def main():
    model = svc.Model(MODEL)
    stats = pp.load_training_stats()
    feats = {f"ref{i:02d}": pp.StreamFeaturizer() for i in range(1, N_TRUCKS + 1)}
    last_t = {t: None for t in feats}
    rows, lock, done = [], threading.Lock(), threading.Event()

    def on_message(c, u, m):
        truck, leaf = m.topic.split("/")[2], m.topic.split("/")[3]
        d = json.loads(m.payload)
        with lock:
            if done.is_set() or truck not in feats:
                return
            if leaf == "vibration":
                feats[truck].add_vibration(float(d["ts"]), float(d["value"]))
                return
            ts = float(d["ts"])
            if last_t[truck] is not None and ts <= last_t[truck]:
                return
            last_t[truck] = ts
            f = feats[truck].add_temperature(ts, float(d["value"]))
            if f is None:
                return
            probs, _ = model.predict(pp.normalise(f, stats))
            rows.append({"truck": truck, "window_end_ts": ts, "probs": [float(x) for x in probs],
                         "pred": int(np.argmax(probs))})
            if len(rows) >= N_WINDOWS:
                done.set()

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="logibridge-reference-builder")
    c.on_message = on_message
    c.connect("localhost", 1883, 30)
    c.subscribe([("logibridge/trucks/+/temperature", 1), ("logibridge/trucks/+/vibration", 0)])
    c.loop_start()
    sims = [subprocess.Popen([sys.executable, str(ROOT / "data_pipeline" / "simulator.py"), "--anomaly", "none",
                              "--truck-id", t, "--duration", str(DURATION)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for t in feats]
    print(f"{N_TRUCKS} clean simulators, real time, ~{DURATION / 60:.1f} min for {N_WINDOWS} windows")
    t0 = time.time()
    while not done.wait(30):
        print(f"  {time.time() - t0:5.0f} s  {len(rows)}/{N_WINDOWS} windows", flush=True)
        if all(s.poll() is not None for s in sims):
            break
    for s in sims:
        s.terminate()
    c.loop_stop(); c.disconnect()
    rows = rows[:N_WINDOWS]
    assert len(rows) == N_WINDOWS, f"only {len(rows)} windows collected"

    pn = np.array([r["probs"][0] for r in rows])
    counts, props = P.proportions(pn)
    ref = {
        "description": "PSI reference: distribution of the deployed model's Normal-class probability "
                       "on 300 fresh clean Normal windows (simulator --anomaly none, real time).",
        "score": "p_normal (model output probability of class 0)",
        "bins": P.EDGES, "bin_labels": P.BIN_LABELS, "counts": counts.tolist(),
        "proportions": [round(float(x), 6) for x in props], "n_windows": N_WINDOWS,
        "eps_for_empty_bins": P.EPS,
        "model": MODEL.name, "model_sha256": model.sha256, "runtime": svc.RUNTIME,
        "stats_sha256": hashlib.sha256(pp.STATS_PATH.read_bytes()).hexdigest(),
        "source": {"trucks": list(feats), "simulator": "--anomaly none, real time (no --speedup)",
                   "built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": "personal PC (MACHINES.md)"},
        "p_normal_summary": {"min": round(float(pn.min()), 4), "mean": round(float(pn.mean()), 4),
                             "predicted_normal": int(sum(r["pred"] == 0 for r in rows))},
    }
    json.dump(ref, open(ROOT / "monitoring" / "reference_dist.json", "w"), indent=2)
    (ROOT / "monitoring" / "results").mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(ROOT / "monitoring" / "results" / "reference_windows.json", "w"))
    print(f"reference: counts {counts.tolist()} | p_normal min {pn.min():.4f} mean {pn.mean():.4f} | "
          f"predicted Normal {ref['p_normal_summary']['predicted_normal']}/{N_WINDOWS}")


if __name__ == "__main__":
    main()
