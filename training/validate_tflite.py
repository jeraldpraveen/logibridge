"""Validation metrics of the CONVERTED .tflite files (not the Keras models).

Runs every validation window through each .tflite exactly as the edge service does
(INT8 input quantisation for M2/M3). Reports accuracy, confusion matrix, per-class recall
with support, and Critical recall per fault mechanism (which recording the window came
from). Validation windows are re-derived from the preserved recordings with the same rules
as the build and checked to equal dataset.npz, so the mechanism tag is exact.
Writes optimisation/results/validation_tflite.json (with model + dataset hashes).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
sys.path.insert(0, str(ROOT / "training"))
import preprocessing as pp      # noqa: E402
import labeling as lb           # noqa: E402
import generate_dataset as gd   # noqa: E402

MODELS = ROOT / "training" / "models"
VARIANTS = ["m1_fp32.tflite", "m2_ptq_int8.tflite", "m3_pruned_ptq_int8.tflite"]
MECHANISM = {"CB2": "combined", "CB5": "combined", "CT4": "temperature-only",
             "CT7": "temperature-only", "CV": "vibration-only"}
H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def val_with_source():
    stats = pp.load_training_stats()
    X, y, src = [], [], []
    for n in gd.RECORDINGS:
        r = np.load(gd.RAW / f"rec_{n}.npz")
        lab, steady = lb.label_windows(r["win_ts"], r["truth_ts"], r["truth_offset"], r["truth_vib"])
        Xk, yk = pp.normalise(r["X"][steady], stats), lab[steady]
        cut = int(len(Xk) * (1 - gd.VAL_FRAC))
        X.append(Xk[cut:]); y.append(yk[cut:]); src += [n] * (len(Xk) - cut)
    return np.concatenate(X), np.concatenate(y), np.array(src)


def run(path, X):
    it = tf.lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = []
    for x in X:
        x = x.reshape(1, -1).astype(np.float32)
        if i["dtype"] == np.int8:
            s, z = i["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        it.set_tensor(i["index"], x)
        it.invoke()
        out.append(int(np.argmax(it.get_tensor(o["index"]).copy())))
    return np.array(out), str(np.dtype(i["dtype"])), str(np.dtype(o["dtype"]))


if __name__ == "__main__":
    X, y, src = val_with_source()
    d = np.load(ROOT / "training" / "dataset.npz")
    assert np.array_equal(X, d["X_val"]) and np.array_equal(y, d["y_val"]), \
        "re-derived validation set differs from dataset.npz"
    res = {"dataset_sha256": H(ROOT / "training" / "dataset.npz"),
           "stats_sha256": H(pp.STATS_PATH), "n_val": int(len(y)),
           "val_support": np.bincount(y, minlength=3).tolist(), "variants": {}}
    for v in VARIANTS:
        pred, din, dout = run(MODELS / v, X)
        cm = np.zeros((3, 3), int)
        for t, p in zip(y, pred):
            cm[t, p] += 1
        rec = {c: {"correct": int(cm[c, c]), "support": int(cm[c].sum()),
                   "recall_pct": round(100 * cm[c, c] / cm[c].sum(), 2)} for c in range(3)}
        mech = {}
        for m in sorted(set(MECHANISM.values())):
            sel = (y == 2) & np.isin(src, [k for k, v_ in MECHANISM.items() if v_ == m])
            mech[m] = {"correct": int((pred[sel] == 2).sum()), "support": int(sel.sum()),
                       "recall_pct": round(100 * float((pred[sel] == 2).mean()), 2)}
        res["variants"][v] = {"sha256": H(MODELS / v), "size_bytes": (MODELS / v).stat().st_size,
                              "input_dtype": din, "output_dtype": dout,
                              "accuracy_pct": round(100 * float((pred == y).mean()), 2),
                              "confusion": cm.tolist(), "recall": rec,
                              "critical_recall_by_mechanism": mech}
    out = ROOT / "optimisation" / "results" / "validation_tflite.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(f"validation n={res['n_val']} support {res['val_support']} | dataset {res['dataset_sha256'][:12]}")
    for v, r in res["variants"].items():
        print(f"\n{v}  ({r['size_bytes']} B, in {r['input_dtype']} / out {r['output_dtype']}, "
              f"sha {r['sha256'][:12]})")
        print(f"  accuracy {r['accuracy_pct']:.2f} %   confusion {r['confusion']}")
        print("  recall: " + "  ".join(f"class {c} {x['correct']}/{x['support']} = {x['recall_pct']:.2f} %"
                                       for c, x in r["recall"].items()))
        print("  Critical by mechanism: " + "  ".join(
            f"{m} {x['correct']}/{x['support']} = {x['recall_pct']:.2f} %"
            for m, x in r["critical_recall_by_mechanism"].items()))
