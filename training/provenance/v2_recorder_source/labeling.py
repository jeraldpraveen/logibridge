"""Single labelling rule shared by dataset generation and evaluation.

Ground truth = the simulator's noise-free `truth` topic (offset from setpoint, failure
vibration flag), mapped onto the brief's class definitions:

    2 Critical  offset > 3 degC  OR  failure vibration active
    1 Warning   1 degC <= offset <= 3 degC
    0 Normal    otherwise (within +/-1 degC, normal vibration)

A window is labelled with the true class at its END time. It is 'steady' only if the true
class is constant over its full input history: the 30 s window plus the moving-average
history (MA_N - 1 extra samples: 4 s for temperature at 1 Hz, 8 s for vibration at 0.5 Hz),
i.e. HISTORY_S = 38 s before the window end. Non-steady windows are transition windows.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
import preprocessing as pp  # noqa: E402

HISTORY_S = pp.WINDOW_S + (pp.MA_N - 1) * 2.0


def true_class(offset, vib_fault):
    if offset > 3.0 or vib_fault:
        return 2
    return 1 if offset >= 1.0 else 0


def label_windows(win_ts, truth_ts, truth_offset, truth_vib):
    """Return (labels, steady) arrays for windows ending at win_ts."""
    truth_ts = np.asarray(truth_ts)
    cls = np.array([true_class(o, v) for o, v in zip(truth_offset, truth_vib)])
    labels, steady = [], []
    for t in win_ts:
        span = cls[(truth_ts >= t - HISTORY_S) & (truth_ts <= t)]
        labels.append(int(span[-1]) if len(span) else 0)
        steady.append(len(span) > 0 and len(set(span.tolist())) == 1)
    return np.array(labels, int), np.array(steady, bool)
