"""Task C2 mandatory experiment: inference with correct vs 3-sigma-shifted normalisation stats.

The edge service z-scores every feature vector with the frozen training_stats.npy
([mean; std] from clean Normal data). This experiment shows what happens if those stats are
wrong, e.g. recomputed from live data or taken from another truck's sensor calibration.

Conditions (the std is never changed; only the mean is shifted, per feature, by k * its std):
  correct            mean, std as saved
  all +3 sigma       mean + 3*std on all 6 features  -> every z-score moves by -3
  all -3 sigma       mean - 3*std on all 6 features  -> every z-score moves by +3
  <feature> +/-3 sigma  one feature shifted at a time, the rest correct

Models: the committed M2 (PTQ INT8, recommended) and M1 (FP32, reference), run through the
TFLite interpreter exactly as validate_tflite.py does. No training or conversion happens here.
Raw validation features are re-derived from the preserved recordings and checked to reproduce
dataset.npz X_val/y_val with the correct stats.

Writes data_pipeline/results/normalisation_experiment.json.
Run:  .venv\\Scripts\\python data_pipeline\\normalisation_experiment.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
sys.path.insert(0, str(ROOT / "training"))
import preprocessing as pp        # noqa: E402
import labeling as lb             # noqa: E402
import generate_dataset as gd     # noqa: E402
from validate_tflite import run   # noqa: E402  same interpreter path + INT8 input quantisation

MODELS = ROOT / "training" / "models"
VARIANTS = {"M2 PTQ INT8": "m2_ptq_int8.tflite", "M1 FP32": "m1_fp32.tflite"}
FEATURES = pp.FEATURE_NAMES  # same names as preprocessing.py
CLASSES = ["Normal", "Warning", "Critical"]
K = 3.0
H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def raw_validation():
    """Un-normalised validation features, same split rule as the dataset build."""
    X, y = [], []
    for n in gd.RECORDINGS:
        r = np.load(gd.RAW / f"rec_{n}.npz")
        lab, steady = lb.label_windows(r["win_ts"], r["truth_ts"], r["truth_offset"], r["truth_vib"])
        Xk, yk = r["X"][steady], lab[steady]
        cut = int(len(Xk) * (1 - gd.VAL_FRAC))
        X.append(Xk[cut:]); y.append(yk[cut:])
    return np.concatenate(X), np.concatenate(y)


def metrics(y, pred):
    cm = np.zeros((3, 3), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return {"accuracy_pct": round(100 * float((pred == y).mean()), 2),
            "recall_pct": {CLASSES[c]: round(100 * cm[c, c] / cm[c].sum(), 2) for c in range(3)},
            "predicted_counts": {CLASSES[c]: int(cm[:, c].sum()) for c in range(3)},
            "confusion": cm.tolist()}


def conditions(mean, std):
    yield "correct", mean
    yield "all +3 sigma", mean + K * std
    yield "all -3 sigma", mean - K * std
    for j, f in enumerate(FEATURES):
        for sign, tag in ((1, "+"), (-1, "-")):
            m = mean.copy(); m[j] += sign * K * std[j]
            yield f"{f} {tag}3 sigma", m


if __name__ == "__main__":
    mean, std = pp.load_training_stats()
    Xraw, y = raw_validation()
    d = np.load(ROOT / "training" / "dataset.npz")
    assert np.array_equal(pp.normalise(Xraw, (mean, std)), d["X_val"]) and np.array_equal(y, d["y_val"]), \
        "re-derived validation set differs from dataset.npz"

    res = {"description": "Validation accuracy with correct vs mean-shifted (k*std) normalisation stats; "
                          "std unchanged. Shifting the mean by +k*std moves that feature's z-score by -k.",
           "k_sigma": K, "n_val": int(len(y)), "val_support": dict(zip(CLASSES, np.bincount(y, minlength=3).tolist())),
           "dataset_sha256": H(ROOT / "training" / "dataset.npz"), "stats_sha256": H(pp.STATS_PATH),
           "features": FEATURES,
           "shift_in_physical_units": {f: round(float(K * s), 4) for f, s in zip(FEATURES, std)},
           "units": {"temp_mean": "degC", "temp_std": "degC", "temp_rate_C_per_min": "degC/min",
                     "vib_rms": "g", "vib_peak": "g", "vib_kurtosis": "-"},
           "models": {}}
    for name, f in VARIANTS.items():
        rows = {}
        for cond, m in conditions(mean, std):
            pred, _, _ = run(MODELS / f, pp.normalise(Xraw, (m, std)))
            rows[cond] = metrics(y, pred)
        res["models"][name] = {"file": f, "sha256": H(MODELS / f), "conditions": rows}

    out = ROOT / "data_pipeline" / "results" / "normalisation_experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)

    print(f"validation n={res['n_val']} {res['val_support']} | stats {res['stats_sha256'][:8]} "
          f"| dataset {res['dataset_sha256'][:8]}")
    for name, r in res["models"].items():
        print(f"\n{name} ({r['sha256'][:8]})")
        print(f"  {'condition':<24}{'acc %':>7}{'Normal':>8}{'Warning':>9}{'Critical':>9}   predicted N/W/C")
        for cond, mt in r["conditions"].items():
            rc, pc = mt["recall_pct"], mt["predicted_counts"]
            print(f"  {cond:<24}{mt['accuracy_pct']:>7.1f}{rc['Normal']:>8.1f}{rc['Warning']:>9.1f}"
                  f"{rc['Critical']:>9.1f}   {pc['Normal']}/{pc['Warning']}/{pc['Critical']}")
    print(f"\nwrote {out.relative_to(ROOT)}")
