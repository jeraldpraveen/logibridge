"""Task D1 - generate the labelled dataset THROUGH the MQTT pipeline (v2).

All recordings run in parallel, each as its own simulated truck, in real time. One
collector subscribes to the broker, runs the streaming preprocessing pipeline per truck and
stores, per recording: raw feature vectors, window end times, and the simulator's truth
stream. Labels come from the truth stream via training/labeling.py (the same rule the
evaluation uses), NOT from the recording mode, and transition windows are excluded.

Why v2: the v1 dataset (kept in training/models/diag_frozen/) had Critical examples only
from `combined` (always with failure vibration) and Warning only with normal vibration.
The model learned "vibration separates Warning from Critical" and missed temperature-only
breaches and failure vibration at ~6 degC. v2 covers each way a class can occur.

Brief's D1 table -> recordings used here
  class 0  --anomaly none        20 min   -> N                (20 min, as specified)
  class 1  --anomaly temp_drift  15 min   -> W15 + W25        EXTENDED, SPLIT: 2 x 8 = 16 min,
                                                              caps 1.5 / 2.5 degC
  class 2  --anomaly combined    15 min   -> CB2 + CB5        EXTENDED, SPLIT: 2 x 8 = 16 min,
                                                              caps 2 / 5 degC
  class 2  supplementary coverage         -> CT4 + CT7 + CV   (3 x 8 min: temperature-only
                                             breach at ~8 / ~11 degC, vibration-only failure
                                             at 4 degC) - not in the brief's table
Labels come from ground truth, so a recording can contribute to more than one class
(e.g. the ramp of CT7 passes through Warning); counts per class are printed by --build.
Cooling-side deviations: the simulator only drifts upward, so none are recorded.

Drift caps: the brief's +0.08 degC/reading drift is uncapped (72 degC after 15 min), so a
cap is applied per recording to hold the temperature inside the intended band.

Split: per recording, time-block - first 80 % train, last 20 % validation, PURGE windows
dropped between them (40 s between nearest retained windows > 38 s input history).
Normalisation stats: first 10 min of N only (clean Normal), saved once to training_stats.npy.

  python generate_dataset.py            # record everything (~20 min)
  python generate_dataset.py --build    # labels, split, stats -> dataset.npz
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
sys.path.insert(0, str(ROOT / "training"))
import preprocessing as pp  # noqa: E402
import labeling as lb       # noqa: E402

# name -> (simulator mode, seconds, drift cap degC or None)
RECORDINGS = {
    "N":    ("none",       20 * 60, None),
    "W15":  ("temp_drift",  8 * 60, 1.5),
    "W25":  ("temp_drift",  8 * 60, 2.5),
    "CB2":  ("combined",    8 * 60, 2.0),
    "CB5":  ("combined",    8 * 60, 5.0),
    "CT4":  ("temp_drift",  8 * 60, 4.0),
    "CT7":  ("temp_drift",  8 * 60, 7.0),
    "CV":   ("vibration",   8 * 60, None),
}
RAW = ROOT / "training" / "raw"
PURGE = 3
VAL_FRAC = 0.2
STATS_WINDOWS = int(10 * 60 / pp.STEP_S) - 2          # windows inside the first ~10 min


def record(host, port):
    feats = {n: pp.StreamFeaturizer() for n in RECORDINGS}
    wins = {n: [] for n in RECORDINGS}
    truth = {n: [] for n in RECORDINGS}

    def on_msg(client, userdata, msg):
        _, _, truck, stream = msg.topic.split("/")
        name = truck[3:]                                  # strip "ds_"
        if name not in feats:
            return
        d = json.loads(msg.payload)
        if stream == "truth":
            truth[name].append((d["ts"], d["offset"], d["vib_fault"]))
        elif stream == "temperature":
            f = feats[name].add_temperature(d["ts"], d["value"])
            if f is not None:
                wins[name].append((d["ts"], f))
        elif stream == "vibration":
            feats[name].add_vibration(d["ts"], d["value"])

    c = mqtt.Client(CallbackAPIVersion.VERSION2, client_id="dataset-collector")
    c.on_message = on_msg
    c.connect(host, port)
    c.subscribe("logibridge/trucks/+/+", qos=1)
    c.loop_start()
    sims = []
    for name, (mode, secs, cap) in RECORDINGS.items():
        cmd = [sys.executable, str(ROOT / "data_pipeline" / "simulator.py"), "--anomaly", mode,
               "--truck-id", f"ds_{name}", "--host", host, "--port", str(port),
               "--duration", str(secs)]
        if cap is not None:
            cmd += ["--drift-cap", str(cap)]
        sims.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL))
    t0 = time.time()
    while any(p.poll() is None for p in sims):
        time.sleep(30)
        print(f"[{(time.time() - t0) / 60:4.1f} min] " +
              " ".join(f"{n}:{len(wins[n])}" for n in RECORDINGS), flush=True)
    time.sleep(1.5)
    c.loop_stop()
    c.disconnect()
    RAW.mkdir(parents=True, exist_ok=True)
    for n in RECORDINGS:
        tr = np.array(truth[n], dtype=float).reshape(-1, 3)
        np.savez(RAW / f"rec_{n}.npz",
                 win_ts=np.array([w[0] for w in wins[n]]),
                 X=np.array([w[1] for w in wins[n]], dtype=np.float32).reshape(-1, 6),
                 truth_ts=tr[:, 0], truth_offset=tr[:, 1], truth_vib=tr[:, 2].astype(bool))
        print(f"{n}: {len(wins[n])} windows, {len(tr)} truth samples saved")


def build():
    N = np.load(RAW / "rec_N.npz")
    stats = pp.save_training_stats(N["X"][:STATS_WINDOWS])
    print(f"training_stats.npy from first {STATS_WINDOWS} windows of N")
    print("  mean", np.round(stats[0], 3), "\n  std ", np.round(stats[1], 3))
    tr, va, rows = [], [], []
    for n in RECORDINGS:
        r = np.load(RAW / f"rec_{n}.npz")
        y, steady = lb.label_windows(r["win_ts"], r["truth_ts"], r["truth_offset"], r["truth_vib"])
        X = pp.normalise(r["X"][steady], (stats[0], stats[1]))   # time order preserved
        y = y[steady]
        cut = int(len(X) * (1 - VAL_FRAC))
        tr.append((X[:cut - PURGE], y[:cut - PURGE]))
        va.append((X[cut:], y[cut:]))
        rows.append((n, len(steady), int((~steady).sum()), len(X), np.bincount(y, minlength=3),
                     cut - PURGE, len(X) - cut))
    print(f"\n{'rec':5s} {'windows':>7s} {'transition':>10s} {'retained':>8s}  "
          f"{'labels 0/1/2':>14s} {'train':>5s} {'val':>4s}")
    for n, w, t, k, b, a, v in rows:
        print(f"{n:5s} {w:7d} {t:10d} {k:8d}  {str(b.tolist()):>14s} {a:5d} {v:4d}")
    Xtr = np.concatenate([a for a, _ in tr]); ytr = np.concatenate([b for _, b in tr])
    Xva = np.concatenate([a for a, _ in va]); yva = np.concatenate([b for _, b in va])
    np.savez(ROOT / "training" / "dataset.npz", X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva)
    print(f"\ntrain {Xtr.shape} per class {np.bincount(ytr, minlength=3)} | "
          f"val {Xva.shape} per class {np.bincount(yva, minlength=3)}")
    # provenance: which code revision built the dataset, from which recordings
    import hashlib
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    prov = {"built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "build_code_sha256": {f: h(ROOT / f) for f in
                                  ("training/generate_dataset.py", "training/labeling.py",
                                   "data_pipeline/preprocessing.py")},
            "recordings_sha256": {f"rec_{n}.npz": h(RAW / f"rec_{n}.npz") for n in RECORDINGS},
            "outputs_sha256": {"training/dataset.npz": h(ROOT / "training" / "dataset.npz"),
                               "data_pipeline/training_stats.npy": h(pp.STATS_PATH)},
            "per_recording": [{"rec": n, "windows": w, "transition": t, "retained": k,
                               "labels_0_1_2": b.tolist(), "train": a, "val": v}
                              for n, w, t, k, b, a, v in rows]}
    rp = RAW / "v2_recording_provenance.json"
    if rp.exists():
        prov["recording"] = json.load(open(rp))
    json.dump(prov, open(ROOT / "training" / "dataset_provenance.json", "w"), indent=2)
    print("provenance -> training/dataset_provenance.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    a = ap.parse_args()
    build() if a.build else record(a.host, a.port)
