"""Task D1 - generate the labelled dataset THROUGH the MQTT pipeline.

For each class a simulator subprocess is launched in the requested mode
(real time) and this script subscribes to the broker, runs the streaming
preprocessing pipeline, and records raw feature vectors with the class label.

  class 0 Normal    --anomaly none        20 min  (~120 windows)
  class 1 Warning   --anomaly temp_drift  15 min  (~90 windows)
  class 2 Critical  --anomaly combined    15 min  (~90 windows)

Each class uses its own truck_id, so the three can be collected in parallel
(use --only to run a single class). Also records 10 min of clean Normal
output -> training_stats.npy (normalisation constants).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
import preprocessing as pp  # noqa: E402

PLAN = {
    0: ("none", 20 * 60),
    1: ("temp_drift", 15 * 60),
    2: ("combined", 15 * 60),
}
RAW = ROOT / "training" / "raw"


def collect(label, host, port, speedup):
    mode, secs = PLAN[label]
    truck = f"train{label}"
    feat = pp.StreamFeaturizer()
    rows = []

    def on_msg(client, userdata, msg):
        d = json.loads(msg.payload)
        if msg.topic.endswith("/temperature"):
            f = feat.add_temperature(d["ts"], d["value"])
            if f is not None:
                rows.append(f)
                print(f"[class {label}] window {len(rows)}", flush=True)
        elif msg.topic.endswith("/vibration"):
            feat.add_vibration(d["ts"], d["value"])

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"collector-{label}")
    c.on_message = on_msg
    c.connect(host, port)
    c.subscribe([(f"logibridge/trucks/{truck}/temperature", 1),
                 (f"logibridge/trucks/{truck}/vibration", 0)])
    c.loop_start()
    sim = subprocess.Popen([sys.executable, str(ROOT / "data_pipeline" / "simulator.py"),
                            "--anomaly", mode, "--truck-id", truck,
                            "--host", host, "--port", str(port),
                            "--duration", str(secs), "--speedup", str(speedup)])
    sim.wait()
    time.sleep(1)
    c.loop_stop()
    c.disconnect()
    X = np.array(rows, dtype=np.float32).reshape(-1, 6)
    RAW.mkdir(parents=True, exist_ok=True)
    np.save(RAW / f"class{label}_raw.npy", X)
    print(f"[class {label}] saved {len(X)} windows -> {RAW / f'class{label}_raw.npy'}")


def build(stats_minutes=10):
    """Combine per-class recordings; fit normalisation stats on first 10 min of Normal."""
    X0 = np.load(RAW / "class0_raw.npy")
    n_stats = int(stats_minutes * 60 / pp.STEP_S) - 2    # windows in ~10 min
    stats = pp.save_training_stats(X0[:n_stats])
    print("training_stats.npy: mean", np.round(stats[0], 3), "std", np.round(stats[1], 3))
    Xs, ys = [], []
    for k in (0, 1, 2):
        X = np.load(RAW / f"class{k}_raw.npy")
        Xs.append(pp.normalise(X, (stats[0], stats[1])))
        ys.append(np.full(len(X), k))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    np.savez(ROOT / "training" / "dataset.npz", X=X, y=y)
    print("dataset:", X.shape, "class counts", np.bincount(y))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, choices=[0, 1, 2], help="collect one class only")
    ap.add_argument("--build", action="store_true", help="combine recordings into dataset.npz")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--speedup", type=float, default=1.0, help="test only; real collection = 1")
    a = ap.parse_args()
    if a.build:
        build()
    else:
        for k in ([a.only] if a.only is not None else [0, 1, 2]):
            collect(k, a.host, a.port, a.speedup)
