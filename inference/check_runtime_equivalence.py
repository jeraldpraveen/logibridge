"""Checks that the container runtime (ai-edge-litert) gives the same results as the TensorFlow
tf.lite interpreter used for every result in F2/F3, on the 86 validation windows.

Uses inference_service.Model (the exact predict code that runs on the truck).
  host:       .venv\\Scripts\\python inference\\check_runtime_equivalence.py host
  container:  docker run --rm -v <repo>:/repo -w /repo logibridge-inference:latest \\
                  python /repo/inference/check_runtime_equivalence.py container
  compare:    .venv\\Scripts\\python inference\\check_runtime_equivalence.py compare
Writes inference/results/runtime_<tag>.json and runtime_equivalence.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "inference" / "results"
VARIANTS = ["m1_fp32.tflite", "m2_ptq_int8.tflite", "m3_pruned_ptq_int8.tflite"]


def collect(tag):
    sys.path.insert(0, str(ROOT / "inference"))
    import inference_service as svc
    d = np.load(ROOT / "training" / "dataset.npz")
    X, y = d["X_val"], d["y_val"]
    res = {"tag": tag, "runtime": svc.RUNTIME, "n": int(len(y)), "variants": {}}
    for v in VARIANTS:
        m = svc.Model(ROOT / "training" / "models" / v)
        P = np.array([m.predict(x)[0] for x in X])
        pred = P.argmax(1)
        res["variants"][v] = {"sha256": m.sha256, "pred": pred.tolist(), "probs": P.round(6).tolist(),
                              "accuracy_pct": round(100 * float((pred == y).mean()), 2)}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"runtime_{tag}.json", "w"))
    print(f"{tag} ({res['runtime']}): " + ", ".join(f"{v} acc {r['accuracy_pct']}%" for v, r in res["variants"].items()))


def compare():
    a, b = (json.load(open(OUT / f"runtime_{t}.json")) for t in ("host", "container"))
    out = {"host_runtime": a["runtime"], "container_runtime": b["runtime"], "n": a["n"], "variants": {}}
    for v in VARIANTS:
        ra, rb = a["variants"][v], b["variants"][v]
        diff = np.abs(np.array(ra["probs"]) - np.array(rb["probs"]))
        out["variants"][v] = {"sha256_equal": ra["sha256"] == rb["sha256"],
                              "identical_predictions": ra["pred"] == rb["pred"],
                              "n_prediction_differences": int(sum(p != q for p, q in zip(ra["pred"], rb["pred"]))),
                              "max_abs_prob_difference": float(diff.max()),
                              "accuracy_host_pct": ra["accuracy_pct"], "accuracy_container_pct": rb["accuracy_pct"]}
    json.dump(out, open(OUT / "runtime_equivalence.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    compare() if sys.argv[1] == "compare" else collect(sys.argv[1])
