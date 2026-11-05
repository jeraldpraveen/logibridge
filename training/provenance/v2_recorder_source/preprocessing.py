"""LogiEdge preprocessing pipeline (Tasks C2 + C3).

Order of operations:
  1. Filtering      5-sample moving average per stream (temperature, vibration)
  2. Windowing      30 s windows, 10 s step, selected by message timestamp
  3. Features       6-value vector (feature-level fusion, Task C3):
                      [temp_mean, temp_std, temp_rate_C_per_min,
                       vib_rms, vib_peak, vib_kurtosis]
  4. Normalisation  z-score with training_stats.npy (computed ONCE offline from
                    clean Normal output; never recomputed from live data)

Only numpy is used, so the same file runs inside the slim Docker image.
"""
from collections import deque
from pathlib import Path

import numpy as np

WINDOW_S = 30.0
STEP_S = 10.0
MA_N = 5
FEATURE_NAMES = ["temp_mean", "temp_std", "temp_rate_C_per_min",
                 "vib_rms", "vib_peak", "vib_kurtosis"]
HERE = Path(__file__).resolve().parent
STATS_PATH = HERE / "training_stats.npy"


# ---------- step 1: filtering ----------
class MovingAverage:
    """Causal 5-sample moving average (streaming)."""

    def __init__(self, n=MA_N):
        self.buf = deque(maxlen=n)

    def update(self, x):
        self.buf.append(float(x))
        return sum(self.buf) / len(self.buf)


# ---------- step 3: feature extraction ----------
def extract_features(t_ts, t_val, v_ts, v_val):
    """6-value feature vector from one window of already-filtered samples."""
    t_ts, t_val = np.asarray(t_ts, float), np.asarray(t_val, float)
    v_val = np.asarray(v_val, float)
    t_mean = t_val.mean()
    t_std = t_val.std()
    # least-squares slope, seconds -> per minute
    if len(t_val) >= 2 and np.ptp(t_ts) > 0:
        slope = np.polyfit(t_ts - t_ts[0], t_val, 1)[0] * 60.0
    else:
        slope = 0.0
    v_rms = float(np.sqrt(np.mean(v_val ** 2)))
    v_peak = float(np.max(np.abs(v_val)))
    centred = v_val - v_val.mean()
    m2 = np.mean(centred ** 2)
    v_kurt = float(np.mean(centred ** 4) / (m2 ** 2) - 3.0) if m2 > 1e-12 else 0.0
    return np.array([t_mean, t_std, slope, v_rms, v_peak, v_kurt], dtype=np.float32)


class StreamFeaturizer:
    """Streaming pipeline: feed filtered samples, get a feature vector every STEP_S."""

    def __init__(self):
        self.t_ma, self.v_ma = MovingAverage(), MovingAverage()
        self.temp = deque()   # (ts, filtered value)
        self.vib = deque()
        self.next_emit = None

    def add_temperature(self, ts, value):
        self.temp.append((ts, self.t_ma.update(value)))
        return self._maybe_emit(ts)

    def add_vibration(self, ts, value):
        self.vib.append((ts, self.v_ma.update(value)))
        return None

    def _maybe_emit(self, now):
        if self.next_emit is None:
            self.next_emit = now + WINDOW_S      # first full window
            return None
        if now < self.next_emit:
            return None
        self.next_emit += STEP_S
        lo = now - WINDOW_S
        while self.temp and self.temp[0][0] < lo:
            self.temp.popleft()
        while self.vib and self.vib[0][0] < lo:
            self.vib.popleft()
        if len(self.temp) < 10 or len(self.vib) < 5:
            return None
        t = np.array(self.temp)
        v = np.array(self.vib)
        return extract_features(t[:, 0], t[:, 1], v[:, 0], v[:, 1])


# ---------- offline version (same maths) for recorded streams ----------
def featurize_recording(t_ts, t_val, v_ts, v_val):
    """Run the streaming pipeline over recorded arrays; returns (n, 6) features."""
    sf = StreamFeaturizer()
    events = sorted([(t, 0, x) for t, x in zip(t_ts, t_val)] +
                    [(t, 1, x) for t, x in zip(v_ts, v_val)])
    out = []
    for ts, kind, x in events:
        f = sf.add_temperature(ts, x) if kind == 0 else sf.add_vibration(ts, x)
        if f is not None:
            out.append(f)
    return np.array(out, dtype=np.float32).reshape(-1, 6)


# ---------- step 4: normalisation ----------
def save_training_stats(features, path=STATS_PATH):
    """features: (n, 6) from >= 10 min of clean Normal data. Saved as [mean; std]."""
    stats = np.stack([features.mean(axis=0), features.std(axis=0) + 1e-6]).astype(np.float32)
    np.save(path, stats)
    return stats


def load_training_stats(path=STATS_PATH):
    stats = np.load(path)
    return stats[0], stats[1]


def normalise(features, stats=None):
    mean, std = stats if stats is not None else load_training_stats()
    return ((features - mean) / std).astype(np.float32)
