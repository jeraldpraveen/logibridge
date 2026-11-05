"""Evolving-fault evaluation: clean -> fault -> clean, through the live MQTT pipeline.

Two steps, so results are reproducible from saved records:
  record   run simulator scenarios, run FROZEN models + stats on every window, save
           per-window records (truth stream, predictions, confidences, timestamps) together
           with the SHA-256 of each model file and of training_stats.npy.
  analyse  recompute every metric from a saved record file (no MQTT, no models needed).

Ground truth comes from the simulator's `truth` topic (noise-free state):
    true class = 2 if offset > 3 degC or failure vibration active
                 1 if offset >= 1 degC
                 0 otherwise

Labelling: a window is labelled with the true class at its END time. A window is 'steady'
only if the true class is constant over its full INPUT HISTORY: the 30 s window plus the
moving-average history (MA_N - 1 = 4 extra samples: 4 s for temperature at 1 Hz, 8 s for
vibration at 0.5 Hz), i.e. 38 s before the window end. Others are 'transition windows'.

Delays (per model), using two references for temperature scenarios:
  fault_onset_ts     the simulator starts the fault (drift begins / vibration steps)
  target_onset_ts    ground truth first reaches the scenario's target class
  detection          first prediction >= target class inside [target_onset, fault_end).
                     Predictions >= target while truth is still below target are 'early
                     escalations', reported separately and never counted as detection.
                     A prediction at/after fault_end does not count (missed).
  detection_delay    reported from BOTH fault onset and target onset (only if detected)
  recovery_delay     reported only if detected: first Normal prediction at/after fault_end.
                     If never detected, recovery is 'n/a (never alerted)'.

Provenance: record stores SHA-256 of the recording code + models + stats; analyse stores
SHA-256 of the analysis/labelling code it ran with, so updated logic stays traceable.

Usage:
  python evaluate_transitions.py record  --tag diag_frozen [--models-dir DIR] [--stats FILE]
  python evaluate_transitions.py analyse --tag diag_frozen
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
sys.path.insert(0, str(ROOT / "training"))
import preprocessing as pp  # noqa: E402
import labeling as lb       # noqa: E402  (same labelling rule as dataset generation)

MODELS = ROOT / "training" / "models"
OUT = ROOT / "optimisation" / "results"
CLEAN_BEFORE, FAULT_FOR, CLEAN_AFTER = 120, 180, 150
HISTORY_S = lb.HISTORY_S

# name -> (simulator mode, drift cap, target class the scenario must reach)
SCENARIOS = {
    "temp_to_critical": ("temp_drift", 5.0, 2),   # normal vibration, temperature crosses 7 degC
    "temp_warning":     ("temp_drift", 2.0, 1),   # drift capped inside the 1-3 degC band
    "vib_only":         ("vibration", None, 2),   # failure vibration, temperature normal
    "combined_capped":  ("combined", 2.0, 2),     # ~6 degC + failure vibration
}
DEFAULT_MODELS = ["m1_fp32.tflite", "m2_ptq_int8.tflite", "m3_pruned_ptq_int8.tflite"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


CODE_FILES = ["training/evaluate_transitions.py", "training/labeling.py",
              "data_pipeline/preprocessing.py", "data_pipeline/simulator.py"]


def code_hashes():
    return {f: sha256(ROOT / f) for f in CODE_FILES}


def true_class(t):
    return lb.true_class(t["offset"], t["vib_fault"])


# ------------------------------------------------------------------ record
class Tflite:
    def __init__(self, path):
        import tensorflow as tf
        self.it = tf.lite.Interpreter(model_path=str(path))
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        self.o = self.it.get_output_details()[0]

    def predict(self, x):
        x = x.reshape(1, -1).astype(np.float32)
        if self.i["dtype"] == np.int8:
            s, z = self.i["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        self.it.set_tensor(self.i["index"], x)
        self.it.invoke()
        y = self.it.get_tensor(self.o["index"]).copy().astype(np.float32).ravel()
        if self.o["dtype"] == np.int8:
            s, z = self.o["quantization"]
            y = (y - z) * s
        return int(np.argmax(y)), float(np.max(y))


def record_scenario(scen, host, port, model_files, stats):
    import paho.mqtt.client as mqtt
    mode, cap, _ = SCENARIOS[scen]
    truck = f"eval_{scen}"
    models = {Path(m).stem: Tflite(m) for m in model_files}   # own interpreters per thread
    feat, truth, wins = pp.StreamFeaturizer(), [], []

    def on_msg(c, u, msg):
        d = json.loads(msg.payload)
        if msg.topic.endswith("/truth"):
            truth.append(d)
        elif msg.topic.endswith("/temperature"):
            f = feat.add_temperature(d["ts"], d["value"])
            if f is not None:
                x = pp.normalise(f, stats)
                out = {k: m.predict(x) for k, m in models.items()}
                wins.append({"ts": d["ts"], "features": f.tolist(),
                             "pred": {k: v[0] for k, v in out.items()},
                             "conf": {k: v[1] for k, v in out.items()}})
        elif msg.topic.endswith("/vibration"):
            feat.add_vibration(d["ts"], d["value"])

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"eval-{scen}-{os.getpid()}")
    c.on_message = on_msg
    c.connect(host, port)
    c.subscribe([(f"logibridge/trucks/{truck}/{s}", 1) for s in ("temperature", "vibration", "truth")])
    c.loop_start()
    cmd = [sys.executable, str(ROOT / "data_pipeline" / "simulator.py"), "--anomaly", mode,
           "--truck-id", truck, "--host", host, "--port", str(port),
           "--anomaly-after", str(CLEAN_BEFORE), "--anomaly-for", str(FAULT_FOR),
           "--duration", str(CLEAN_BEFORE + FAULT_FOR + CLEAN_AFTER)]
    if cap is not None:
        cmd += ["--drift-cap", str(cap)]
    subprocess.Popen(cmd).wait()
    time.sleep(1.5)
    c.loop_stop()
    c.disconnect()
    return {"scenario": scen, "mode": mode, "drift_cap": cap, "truth": truth, "windows": wins}


def record(a):
    from concurrent.futures import ThreadPoolExecutor
    mdir = Path(a.models_dir).resolve()
    files = [mdir / m for m in a.models]
    stats_path = Path(a.stats).resolve()
    stats = pp.load_training_stats(stats_path)
    meta = {"tag": a.tag, "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "clean_before_s": CLEAN_BEFORE, "fault_for_s": FAULT_FOR, "clean_after_s": CLEAN_AFTER,
            "models": {f.stem: {"file": str(f.relative_to(ROOT)), "sha256": sha256(f)} for f in files},
            "stats": {"file": str(stats_path.relative_to(ROOT)), "sha256": sha256(stats_path)},
            "recording_code_sha256": code_hashes()}
    with ThreadPoolExecutor(len(a.scenarios)) as ex:
        recs = list(ex.map(lambda s: record_scenario(s, a.host, a.port, files, stats), a.scenarios))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"transitions_{a.tag}_records.json"
    json.dump({"meta": meta, "scenarios": recs}, open(path, "w"))
    print(f"saved records -> {path}")


# ------------------------------------------------------------------ analyse
def analyse_scenario(rec):
    scen, target = rec["scenario"], SCENARIOS[rec["scenario"]][2]
    truth, wins = rec["truth"], rec["windows"]
    onset = next(t["ts"] for t in truth if t["fault"])
    end = next(t["ts"] for t in truth if t["ts"] > onset and not t["fault"])
    target_onset = next((t["ts"] for t in truth if t["ts"] >= onset and true_class(t) >= target), None)
    ts = np.array([w["ts"] for w in wins])
    labels, steady = lb.label_windows(ts, [t["ts"] for t in truth],
                                      [t["offset"] for t in truth], [t["vib_fault"] for t in truth])
    out = {"scenario": scen, "target_class": target, "n_windows": len(wins),
           "n_steady": int(steady.sum()), "n_transition": int((~steady).sum()),
           "fault_onset_ts": onset, "target_onset_ts": target_onset, "fault_end_ts": end,
           "target_onset_after_fault_onset_s": None if target_onset is None else round(target_onset - onset, 1),
           "models": {}}
    for k in wins[0]["pred"]:
        pred = np.array([w["pred"][k] for w in wins])
        # detection = first prediction >= target class while the TARGET condition is active:
        # [target_onset, fault_end). A window at fault_end is already clean -> excluded.
        if target_onset is not None:
            hits = ts[(ts >= target_onset) & (ts < end) & (pred >= target)]
        else:
            hits = ts[:0]
        detected = len(hits) > 0
        # escalation to >= target while truth was still below target (reported separately,
        # never counted as detection)
        early = ts[(ts >= onset) & (ts < (target_onset if target_onset is not None else end))
                   & (pred >= target)]
        rec_after = ts[(ts >= end) & (pred == 0)]
        cm = np.zeros((3, 3), int)
        for t_, p_ in zip(labels[steady], pred[steady]):
            cm[t_, p_] += 1
        sc = steady & (labels == 2)
        # clean-state errors: steady windows whose truth is Normal (before fault onset or after
        # the fault has fully left the input history) but the model predicted Warning/Critical
        clean = steady & (labels == 0)
        clean_pre = clean & (ts < onset)
        clean_post = clean & (ts >= end)
        tc = steady & (labels == target)        # steady target-class windows (any target)
        out["models"][k] = {
            "acc_all": round(float((pred == labels).mean()), 4),
            "acc_steady": round(float((pred[steady] == labels[steady]).mean()), 4) if steady.any() else None,
            "critical_recall_steady": round(float((pred[sc] == 2).mean()), 4) if sc.any() else None,
            "n_critical_steady": int(sc.sum()),
            "critical_correct_steady": int((pred[sc] == 2).sum()),
            "target_correct_steady": int((pred[tc] == target).sum()),
            "n_target_steady": int(tc.sum()),
            "clean_errors_pre": int((pred[clean_pre] != 0).sum()),
            "n_clean_pre": int(clean_pre.sum()),
            "clean_errors_post": int((pred[clean_post] != 0).sum()),
            "n_clean_post": int(clean_post.sum()),
            "detected": detected,
            "early_escalations": int(len(early)),
            "first_early_escalation_before_target_s": (round(float(target_onset - early[0]), 1)
                                                       if len(early) and target_onset else None),
            "detection_delay_from_fault_onset_s": round(float(hits[0] - onset), 1) if detected else None,
            "detection_delay_from_target_onset_s": (round(float(hits[0] - target_onset), 1)
                                                    if detected and target_onset else None),
            "recovery_delay_s": (round(float(rec_after[0] - end), 1) if detected and len(rec_after)
                                 else ("n/a (never alerted)" if not detected else None)),
            "confusion_steady": cm.tolist(),
        }
    return out


def analyse(a):
    data = json.load(open(OUT / f"transitions_{a.tag}_records.json"))
    res = {"meta": data["meta"], "history_s_for_steady": HISTORY_S,
           "analysed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "analysis_code_sha256": code_hashes(),
           "results": [analyse_scenario(r) for r in data["scenarios"]]}
    json.dump(res, open(OUT / f"transitions_{a.tag}_analysis.json", "w"), indent=2)
    print(f"tag {a.tag} | stats sha256 {data['meta']['stats']['sha256'][:12]}")
    for r in res["results"]:
        print(f"\n== {r['scenario']}  target class {r['target_class']}  windows {r['n_windows']} "
              f"(steady {r['n_steady']}, transition {r['n_transition']})  "
              f"truth reaches target {r['target_onset_after_fault_onset_s']} s after fault onset")
        for k, m in r["models"].items():
            cr = m["critical_recall_steady"]
            print(f"  {k:20s} acc_steady {m['acc_steady'] * 100:5.1f}%  "
                  f"critRecall {'-' if cr is None else f'{cr * 100:.1f}%'} (n={m['n_critical_steady']})  "
                  f"detect {m['detection_delay_from_fault_onset_s']} s from fault / "
                  f"{m['detection_delay_from_target_onset_s']} s from target  "
                  f"recovery {m['recovery_delay_s']}  early-escalations {m['early_escalations']}  "
                  f"target {m['target_correct_steady']}/{m['n_target_steady']}  "
                  f"clean errors pre {m['clean_errors_pre']}/{m['n_clean_pre']} "
                  f"post {m['clean_errors_post']}/{m['n_clean_post']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["record", "analyse"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--models-dir", default=str(MODELS))
    ap.add_argument("--stats", default=str(pp.STATS_PATH))
    a = ap.parse_args()
    record(a) if a.step == "record" else analyse(a)
