"""Can PSI cross 0.25 within 5 minutes at one inference per 10 s? (Task E1 design check)

5 minutes of fault = at most 30 of the 100 rolling inferences. This builds windows of
(100 - k) clean + k fault outputs for k = 0..30 and reports the PSI and the smallest k that
crosses 0.25, for several choices of the empty-bin floor EPS.
  clean outputs : P(Normal) of the 300 reference windows (monitoring/results/reference_windows.json)
  fault outputs : P(Normal) from the deployed model on the 62 Warning + Critical validation windows
Uses the edge service's Model.predict on the committed model; no training.
Writes monitoring/results/psi_sensitivity.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "monitoring"))
import inference_service as svc  # noqa: E402
import psi as P                  # noqa: E402

EPS_CHOICES = [1e-6, 1e-4, 1e-3, 1e-2]
rng = np.random.default_rng(0)


def main():
    ref = json.load(open(ROOT / "monitoring" / "reference_dist.json"))
    clean = np.array([r["probs"][0] for r in json.load(open(ROOT / "monitoring" / "results" / "reference_windows.json"))])
    m = svc.Model(ROOT / "inference" / "model.tflite")
    d = np.load(ROOT / "training" / "dataset.npz")
    fault = np.array([m.predict(x)[0][0] for x, y in zip(d["X_val"], d["y_val"]) if y > 0])
    res = {"reference_proportions": ref["proportions"], "fault_p_normal_bins": P.proportions(fault)[0].tolist(),
           "n_fault_windows": int(len(fault)), "by_eps": {}}
    for eps in EPS_CHOICES:
        curve = []
        for k in range(31):
            vals = np.concatenate([rng.choice(clean, 100 - k), rng.choice(fault, k)])
            curve.append(round(P.psi(ref["proportions"], P.proportions(vals)[1], eps), 4))
        first = next((k for k, v in enumerate(curve) if v > 0.25), None)
        res["by_eps"][str(eps)] = {"psi_by_k_fault_samples": curve, "first_k_over_0.25": first,
                                   "seconds_at_10s_per_inference": None if first is None else first * 10,
                                   "psi_at_k30": curve[30],
                                   "recovery_k_below_0.10": max((k for k, v in enumerate(curve) if v < 0.10), default=None)}
    out = ROOT / "monitoring" / "results" / "psi_sensitivity.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"reference props {ref['proportions']} | fault P(Normal) bins {res['fault_p_normal_bins']}")
    for eps, r in res["by_eps"].items():
        print(f"EPS={eps:<7} PSI at k=30: {r['psi_at_k30']:.3f} | crosses 0.25 at k={r['first_k_over_0.25']} "
              f"(~{r['seconds_at_10s_per_inference']} s) | PSI < 0.10 only while k <= {r['recovery_k_below_0.10']}")


if __name__ == "__main__":
    main()
