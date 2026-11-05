"""Evidence report for the built v2 dataset (read-only: never modifies the dataset).

Recomputes, from the saved recordings and the same rules as generate_dataset.build():
  per recording   recorded, transition-excluded, retained, train, purged, val counts,
                  label support, and the actual time gap between the last training window
                  and the first validation window (must exceed the 38 s input history)
  per split       class support
  calibration     number of DISTINCT training feature vectors (must be >= 200)
  integrity       dataset.npz equals what the rules produce USING THE SAVED STATS
                  (construction consistency only - shares the builder's rules)
  stats origin    independently recomputes mean/std from the first 10 min of rec_N and
                  compares with the saved training_stats.npy; checks those windows are
                  true-Normal and steady
  criteria        the acceptance checklist, each PASS/FAIL
Writes training/split_report.json, plus training/split_report_manifest.json holding the
SHA-256 of the report itself and of the code that produced it.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_pipeline"))
sys.path.insert(0, str(ROOT / "training"))
import preprocessing as pp      # noqa: E402
import labeling as lb           # noqa: E402
import generate_dataset as gd   # noqa: E402

H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

if __name__ == "__main__":
    stats = pp.load_training_stats()
    rows, Xtr_all, Xva_all = [], [], []
    for n in gd.RECORDINGS:
        r = np.load(gd.RAW / f"rec_{n}.npz")
        ts = r["win_ts"]
        y, steady = lb.label_windows(ts, r["truth_ts"], r["truth_offset"], r["truth_vib"])
        ts_k, y_k = ts[steady], y[steady]
        X_k = pp.normalise(r["X"][steady], stats)
        cut = int(len(X_k) * (1 - gd.VAL_FRAC))
        tr_i, pu_i, va_i = np.arange(cut - gd.PURGE), np.arange(cut - gd.PURGE, cut), np.arange(cut, len(X_k))
        Xtr_all.append((X_k[tr_i], y_k[tr_i]))
        Xva_all.append((X_k[va_i], y_k[va_i]))
        gap = float(ts_k[va_i[0]] - ts_k[tr_i[-1]]) if len(tr_i) and len(va_i) else None
        rows.append({
            "rec": n, "mode": gd.RECORDINGS[n][0], "cap": gd.RECORDINGS[n][2],
            "recorded": int(len(ts)), "transition_excluded": int((~steady).sum()),
            "retained": int(len(X_k)), "train": int(len(tr_i)), "purged": int(len(pu_i)),
            "val": int(len(va_i)),
            "train_support": np.bincount(y_k[tr_i], minlength=3).tolist(),
            "val_support": np.bincount(y_k[va_i], minlength=3).tolist(),
            "train_end_to_val_start_s": None if gap is None else round(gap, 2),
            "gap_exceeds_history": None if gap is None else gap > lb.HISTORY_S,
        })
    Xtr = np.concatenate([a for a, _ in Xtr_all]); ytr = np.concatenate([b for _, b in Xtr_all])
    Xva = np.concatenate([a for a, _ in Xva_all]); yva = np.concatenate([b for _, b in Xva_all])
    d = np.load(ROOT / "training" / "dataset.npz")
    same = all(np.array_equal(a, d[k]) for a, k in
               ((Xtr, "X_train"), (ytr, "y_train"), (Xva, "X_val"), (yva, "y_val")))
    distinct = int(len(np.unique(d["X_train"], axis=0)))
    rep = {
        "history_s": lb.HISTORY_S, "purge_windows": gd.PURGE, "val_frac": gd.VAL_FRAC,
        "per_recording": rows,
        "totals": {"recorded": sum(r["recorded"] for r in rows),
                   "transition_excluded": sum(r["transition_excluded"] for r in rows),
                   "retained": sum(r["retained"] for r in rows),
                   "train": int(len(ytr)), "purged": sum(r["purged"] for r in rows),
                   "val": int(len(yva))},
        "train_support": np.bincount(ytr, minlength=3).tolist(),
        "val_support": np.bincount(yva, minlength=3).tolist(),
        "distinct_train_vectors": distinct, "calibration_eligible": distinct >= 200,
        "dataset_matches_rules": bool(same),
        "min_split_gap_s": min(r["train_end_to_val_start_s"] for r in rows
                               if r["train_end_to_val_start_s"] is not None),
        "sha256": {"training/dataset.npz": H(ROOT / "training" / "dataset.npz"),
                   "data_pipeline/training_stats.npy": H(pp.STATS_PATH),
                   "training/dataset_provenance.json": H(ROOT / "training" / "dataset_provenance.json")},
    }
    # --- statistics origin: recompute from the first STATS_WINDOWS windows of N and compare
    N = np.load(gd.RAW / "rec_N.npz")
    calib = N["X"][:gd.STATS_WINDOWS]
    expect = np.stack([calib.mean(axis=0), calib.std(axis=0) + 1e-6]).astype(np.float32)
    saved = np.load(pp.STATS_PATH)
    yN, sN = lb.label_windows(N["win_ts"], N["truth_ts"], N["truth_offset"], N["truth_vib"])
    span_s = float(N["win_ts"][gd.STATS_WINDOWS - 1] - N["win_ts"][0] + pp.WINDOW_S)
    rep["stats_origin"] = {
        "source": "rec_N.npz first STATS_WINDOWS windows", "windows": int(gd.STATS_WINDOWS),
        "covers_s": round(span_s, 1),
        "all_windows_true_normal_and_steady": bool((yN[:gd.STATS_WINDOWS] == 0).all()
                                                  and sN[:gd.STATS_WINDOWS].all()),
        "recomputed_equals_saved": bool(np.array_equal(expect, saved)),
        "max_abs_diff": float(np.abs(expect - saved).max()),
    }
    # --- acceptance criteria (evaluator's list), each pass/fail
    t = rep["totals"]
    rep["criteria"] = {
        "recorded = excluded + retained (every recording)":
            all(r["recorded"] == r["transition_excluded"] + r["retained"] for r in rows),
        "retained = train + purged + val (every recording)":
            all(r["retained"] == r["train"] + r["purged"] + r["val"] for r in rows),
        "every split gap > 38 s":            # missing gap = missing evidence = FAIL
            all(r["gap_exceeds_history"] is True for r in rows),
        "train and val contain all three classes":
            min(rep["train_support"]) > 0 and min(rep["val_support"]) > 0,
        ">= 200 distinct training vectors": distinct >= 200,
        "dataset_matches_rules": bool(same),
        "stats recomputed from first 58 windows of N == saved stats":
            rep["stats_origin"]["recomputed_equals_saved"]
            and rep["stats_origin"]["all_windows_true_normal_and_steady"],
    }
    rep["all_criteria_pass"] = all(rep["criteria"].values())
    out = ROOT / "training" / "split_report.json"
    json.dump(rep, open(out, "w"), indent=2)
    # --- separate manifest: hash of the report RESULT and of the code that produced it
    code = ["training/split_report.py", "training/generate_dataset.py", "training/labeling.py",
            "data_pipeline/preprocessing.py"]
    json.dump({"report": {"file": "training/split_report.json", "sha256": H(out)},
               "report_code_sha256": {f: H(ROOT / f) for f in code},
               "inputs_sha256": rep["sha256"]},
              open(ROOT / "training" / "split_report_manifest.json", "w"), indent=2)
    print(f"{'rec':4s} {'mode':10s} {'cap':>4s} {'rec':>4s} {'excl':>4s} {'kept':>4s} {'trn':>4s} "
          f"{'prg':>3s} {'val':>3s}  {'train 0/1/2':>12s}  {'val 0/1/2':>10s}  {'gap s':>6s}")
    for r in rows:
        print(f"{r['rec']:4s} {r['mode']:10s} {str(r['cap']):>4s} {r['recorded']:4d} "
              f"{r['transition_excluded']:4d} {r['retained']:4d} {r['train']:4d} {r['purged']:3d} "
              f"{r['val']:3d}  {str(r['train_support']):>12s}  {str(r['val_support']):>10s}  "
              f"{r['train_end_to_val_start_s']:6.1f}")
    t = rep["totals"]
    print(f"\nTOTAL recorded {t['recorded']}, transition-excluded {t['transition_excluded']}, "
          f"retained {t['retained']}, train {t['train']}, purged {t['purged']}, val {t['val']}")
    print(f"train support {rep['train_support']} | val support {rep['val_support']}")
    print(f"min train->val gap {rep['min_split_gap_s']} s (history {lb.HISTORY_S} s)")
    print(f"distinct training vectors {distinct} -> calibration eligible: {rep['calibration_eligible']}")
    print(f"dataset.npz matches rules: {rep['dataset_matches_rules']}")
    so = rep["stats_origin"]
    print(f"stats origin: first {so['windows']} windows of N, covering {so['covers_s']} s; all true Normal & "
          f"steady: {so['all_windows_true_normal_and_steady']}; recomputed == saved: "
          f"{so['recomputed_equals_saved']} (max diff {so['max_abs_diff']:.2e})")
    print("\nacceptance criteria:")
    for k, v in rep["criteria"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"ALL PASS: {rep['all_criteria_pass']}")
    print("\ninput hashes:")
    for k, v in rep["sha256"].items():
        print(f"  {v[:16]}  {k}")
    print("report + report-code hashes -> training/split_report_manifest.json")
