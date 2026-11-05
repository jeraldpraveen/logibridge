"""Task F1 (M2) - full-INT8 post-training quantisation of the FP32 baseline.

TFLiteConverter, DEFAULT optimisation, representative dataset of >= 200 samples,
int8 input/output. Calibration uses DISTINCT TRAINING windows only (never validation);
fewer than 200 distinct windows -> abort. The exact calibration inputs are saved
(models/calibration_<model>.npz) with their class support and hash.
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


def calibration_set(X, n=N_CALIB, seed=0, y=None, save_as=None):
    """Distinct training windows only (no replacement); the spec needs >= 200.
    If save_as is given, the selected inputs (and their labels, if y given) are saved."""
    Xu, first = np.unique(X, axis=0, return_index=True)
    if len(Xu) < MIN_CALIB:
        raise SystemExit(f"only {len(Xu)} distinct training windows (< {MIN_CALIB}); "
                         "collect more data - repeated copies add no calibration coverage")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xu), size=min(n, len(Xu)), replace=False)
    Xc = Xu[idx].astype(np.float32)
    msg = f"calibration: {len(idx)} distinct training windows (of {len(Xu)} available)"
    if y is not None:
        yc = np.asarray(y)[first[idx]]
        msg += f", class support {np.bincount(yc, minlength=3).tolist()}"
    if save_as is not None:
        import hashlib
        np.savez(save_as, X=Xc, **({"y": yc} if y is not None else {}))
        msg += (f" -> {Path(save_as).name} | sha256 of X array bytes "
                f"{hashlib.sha256(Xc.tobytes()).hexdigest()[:12]} | sha256 of .npz file "
                f"{hashlib.sha256(Path(save_as).read_bytes()).hexdigest()[:12]}")
    print(msg)
    return Xc


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
    n8 = quantise_int8(model, calibration_set(d["X_train"], y=d["y_train"],
                                              save_as=MODELS / "calibration_m2.npz"),
                       MODELS / "m2_ptq_int8.tflite")
    print(f"m1_fp32.tflite   {n32 / 1024:.2f} KiB")
    print(f"m2_ptq_int8.tflite {n8 / 1024:.2f} KiB")
