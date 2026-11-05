"""Task F1 (M3) - 35 % structured pruning (PolynomialDecay) + full-INT8 PTQ.

Structured = whole neurons (a neuron's weight column + bias), ranked by L2 norm.
Schedule: tensorflow_model_optimization PolynomialDecay (0 -> 35 %, power 3).

Why a custom mask callback: tfmot's prune_low_magnitude(block_size=(n_in, 1)) pools each
weight column to one row, tf.squeeze then makes it rank-1 and tfmot raises
"Input tensor must be rank 2". So tfmot supplies the schedule; the masking is done here.

Starts from the trained M1 weights and fine-tunes while the sparsity ramps up. After
training, zeroed neurons are PHYSICALLY removed (32 -> 21, 16 -> 10 units), otherwise the
.tflite would be no smaller. The 3-unit output layer is never pruned.
"""
import os
from pathlib import Path

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

import convert_ptq as cp
from train_model import per_class_report

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "training" / "models"
SPARSITY = 0.35


class NeuronPruner(tf.keras.callbacks.Callback):
    """Zero the lowest-L2-norm neurons of each hidden layer at the scheduled sparsity."""

    def __init__(self, layers, schedule):
        super().__init__()
        self.layers, self.schedule, self.step = layers, schedule, 0

    def _apply(self):
        _, sp = self.schedule(self.step)
        sp = float(sp)
        for layer in self.layers:
            k, b = layer.get_weights()
            n_prune = int(round(sp * k.shape[1]))
            if n_prune == 0:
                continue
            norms = np.sqrt((k ** 2).sum(axis=0) + b ** 2)
            dead = np.argsort(norms)[:n_prune]
            k[:, dead], b[dead] = 0.0, 0.0
            layer.set_weights([k, b])

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        self._apply()

    def on_train_end(self, logs=None):
        self.step = 10 ** 9            # make sure the final sparsity is applied
        self._apply()


def shrink(model):
    """Remove all-zero hidden neurons; return a smaller dense model."""
    dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    keep_prev = np.arange(6)
    new = []
    for i, layer in enumerate(dense):
        k, b = layer.get_weights()
        k = k[keep_prev, :]
        if i < len(dense) - 1:
            alive = np.where((np.abs(k).sum(axis=0) + np.abs(b)) > 0)[0]
            new.append((k[:, alive], b[alive]))
            keep_prev = alive
        else:
            new.append((k, b))
    m = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(6,))] +
        [tf.keras.layers.Dense(k.shape[1], activation="relu" if i < len(new) - 1 else "softmax")
         for i, (k, _) in enumerate(new)])
    for layer, (k, b) in zip(m.layers, new):
        layer.set_weights([k, b])
    return m


if __name__ == "__main__":
    cp.require_passed_baseline()
    tf.keras.utils.set_random_seed(42)
    d = np.load(ROOT / "training" / "dataset.npz")
    Xtr, ytr, Xva, yva = d["X_train"], d["y_train"], d["X_val"], d["y_val"]
    epochs, bs = 120, 16
    steps = int(np.ceil(len(Xtr) / bs)) * epochs

    model = tf.keras.models.load_model(MODELS / "m1_fp32.h5")      # start from trained M1
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=SPARSITY,
        begin_step=0, end_step=int(steps * 0.8), power=3)
    hidden = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)][:-1]
    model.fit(Xtr, ytr, epochs=epochs, batch_size=bs, verbose=0,
              callbacks=[NeuronPruner(hidden, schedule)])

    small = shrink(model)
    small.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    units = [l.units for l in small.layers]
    pred = small.predict(Xva, verbose=0).argmax(1)
    cm, rec = per_class_report(yva, pred)
    print("pruned architecture (units per Dense):", units)
    print(f"neurons removed: hidden1 {32 - units[0]}/32, hidden2 {16 - units[1]}/16")
    print(f"pruned FP32 val acc {(pred == yva).mean() * 100:.2f} %, recall {np.round(rec * 100, 2)}")
    small.save(MODELS / "m3_pruned_fp32.h5")

    n8 = cp.quantise_int8(small, cp.calibration_set(Xtr), MODELS / "m3_pruned_ptq_int8.tflite")
    print(f"m3_pruned_ptq_int8.tflite {n8 / 1024:.2f} KiB")
