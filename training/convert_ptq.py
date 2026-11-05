"""Task F1 (M2) - full-INT8 post-training quantisation of the FP32 baseline.

TFLiteConverter, DEFAULT optimisation, representative dataset of >= 200 samples,
int8 input/output. Calibration uses TRAINING windows only; if fewer than 200 exist
they are re-sampled with replacement (reported below).
"""
import os
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "training" / "models"
N_CALIB = 256


MIN_CALIB = 200


def require_passed_baseline():
    """Downstream conversion only proceeds from a baseline that passed the 88 % gate."""
    import json
    mp = MODELS / "m1_metrics.json"
    if not mp.exists() or not json.load(open(mp)).get("passed_gate", False):
        raise SystemExit("baseline did not pass the accuracy gate - refusing to convert")


def calibration_set(X, n=N_CALIB, seed=0):
    """Distinct training windows only (no replacement); the spec needs >= 200."""
    X = np.unique(X, axis=0)
    if len(X) < MIN_CALIB:
        raise SystemExit(f"only {len(X)} distinct training windows (< {MIN_CALIB}); "
                         "collect more data - repeated copies add no calibration coverage")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    print(f"calibration: {len(idx)} distinct training windows")
    return X[idx].astype(np.float32)


def quantise_int8(keras_model, X_calib, out_path):
    def rep():
        for x in X_calib:
            yield [x.reshape(1, -1)]
    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()
    Path(out_path).write_bytes(blob)
    return len(blob)


def convert_fp32(keras_model, out_path):
    blob = tf.lite.TFLiteConverter.from_keras_model(keras_model).convert()
    Path(out_path).write_bytes(blob)
    return len(blob)


if __name__ == "__main__":
    require_passed_baseline()
    d = np.load(ROOT / "training" / "dataset.npz")
    model = tf.keras.models.load_model(MODELS / "m1_fp32.h5")
    n32 = convert_fp32(model, MODELS / "m1_fp32.tflite")
    n8 = quantise_int8(model, calibration_set(d["X_train"]), MODELS / "m2_ptq_int8.tflite")
    print(f"m1_fp32.tflite   {n32 / 1024:.2f} KiB")
    print(f"m2_ptq_int8.tflite {n8 / 1024:.2f} KiB")
