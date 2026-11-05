"""Task D1 / F1 (M1) - train the FP32 baseline MLP 6-32-16-3.

Reads training/dataset.npz (time-block train/val split made by generate_dataset.py).
Gate: validation accuracy must exceed 88 % - the script exits non-zero otherwise.
"""
import json
import os
import sys
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"   # tfmot pruning needs legacy Keras (tf_keras)
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "training" / "models"
SEED = 42
GATE = 0.88


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])


def per_class_report(y_true, y_pred):
    cm = np.zeros((3, 3), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    recall = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return cm, recall


def load():
    d = np.load(ROOT / "training" / "dataset.npz")
    return d["X_train"], d["y_train"], d["X_val"], d["y_val"]


if __name__ == "__main__":
    tf.keras.utils.set_random_seed(SEED)
    Xtr, ytr, Xva, yva = load()
    print("train", Xtr.shape, np.bincount(ytr), "| val", Xva.shape, np.bincount(yva))

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                          restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=300,
                     batch_size=16, callbacks=[es], verbose=0)

    pred = model.predict(Xva, verbose=0).argmax(1)
    acc = float((pred == yva).mean())
    cm, recall = per_class_report(yva, pred)
    print(f"\nepochs run: {len(hist.history['loss'])}")
    print(f"validation accuracy: {acc * 100:.2f} %")
    print("confusion matrix (rows = true 0/1/2):\n", cm)
    print("recall per class:", np.round(recall * 100, 2))

    MODELS.mkdir(parents=True, exist_ok=True)
    passed = acc > GATE
    # metrics are saved for every run (including failures) so the evidence is kept
    json.dump({"passed_gate": passed, "val_accuracy": acc, "recall": recall.tolist(),
               "confusion": cm.tolist(),
               "n_train": int(len(ytr)), "n_val": int(len(yva)),
               "train_support": np.bincount(ytr, minlength=3).tolist(),
               "val_support": np.bincount(yva, minlength=3).tolist()},
              open(MODELS / "m1_metrics.json", "w"), indent=2)

    if not passed:
        # a failed run must not leave usable-looking model files behind
        import shutil
        for stale in ("m1_fp32.h5", "m1_fp32.tflite", "m2_ptq_int8.tflite",
                      "m3_pruned_fp32.h5", "m3_pruned_ptq_int8.tflite"):
            (MODELS / stale).unlink(missing_ok=True)
        shutil.rmtree(MODELS / "m1_fp32_savedmodel", ignore_errors=True)
        print(f"\nFAIL: {acc * 100:.2f} % <= {GATE * 100:.0f} % gate - no model exported. "
              "Revisit features/architecture.")
        sys.exit(1)

    model.save(MODELS / "m1_fp32.h5")                  # M1 (also the pruning starting point)
    model.save(MODELS / "m1_fp32_savedmodel")          # Keras SavedModel
    print(f"\nPASS: above the {GATE * 100:.0f} % gate - model exported")
