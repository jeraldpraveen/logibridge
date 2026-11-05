"""Population Stability Index on the model's Normal-class probability (Task E1).

Why P(Normal) and not the top-class confidence: a confident Critical prediction and a confident
Normal prediction both land in [0.75, 1.0] by max-confidence, so a fault would be invisible.
P(Normal) moves from ~1 (clean) to ~0 (Warning/Critical), which is the shift we need to see.

PSI = sum_i (a_i - r_i) * ln(a_i / r_i) over the 4 bins [0,.25) [.25,.5) [.5,.75) [.75,1.0],
with a_i, r_i the current and reference proportions. Empty bins are floored at EPS (not
renormalised), otherwise ln(0) is undefined. EPS is a design choice that sets how strongly an
empty reference bin reacts when it starts to fill; see psi_sensitivity.py.
"""
import numpy as np

EDGES = [0.0, 0.25, 0.50, 0.75, 1.0]          # np.histogram closes the last bin: [0.75, 1.0]
BIN_LABELS = ["[0,0.25)", "[0.25,0.50)", "[0.50,0.75)", "[0.75,1.0]"]
EPS = 1e-4


def proportions(p_normal):
    counts, _ = np.histogram(np.clip(np.asarray(p_normal, float), 0.0, 1.0), bins=EDGES)
    return counts, counts / max(counts.sum(), 1)


def psi(ref_props, cur_props, eps=EPS):
    r = np.maximum(np.asarray(ref_props, float), eps)
    a = np.maximum(np.asarray(cur_props, float), eps)
    return float(np.sum((a - r) * np.log(a / r)))
