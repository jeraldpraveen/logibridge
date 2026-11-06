# LogiEdge — Cold-Chain Edge AI Platform

AIML ZG535 Machine Learning on Edge, mini-project (EC-I). An edge AI pipeline for FreightBridge's
85 refrigerated trucks: a sensor simulator publishes temperature, vibration and door events to a
local Mosquitto broker; the truck's edge node extracts a 6-value feature vector every 10 s and
classifies the cargo state as Normal / Warning / Critical with a small TFLite MLP, offline.

New machine? Read [SETUP_PERSONAL_PC.md](SETUP_PERSONAL_PC.md) first and run `setup_check.py`
until it prints **ALL OK**.

## Repository map

| Folder | Contents | Tasks |
|---|---|---|
| `scenario_architecture/` | Constraint analysis, architecture diagram (+ script that draws it) | A1, A2 |
| `hardware/` | Constraint Triangle, Arithmetic Intensity / Roofline | B1, B2 |
| `data_pipeline/` | `simulator.py`, `preprocessing.py`, `training_stats.npy` | C1, C2, C3 |
| `training/` | Dataset generation, training, PTQ, pruning, TFLite validation, transition evaluation, frozen v1 archive, v2 provenance | D1, F1 |
| `optimisation/` | Five-metric benchmark, Pareto chart, deployment recommendation | F2, F3 |
| `deployment/` | OTA strategy comparison | E3 |

## Frozen artefacts — do not change

Every result in the repository refers to these exact files. **Do not retrain.**
`setup_check.py` fails if any hash changes.

| File | SHA-256 (prefix) |
|---|---|
| `training/models/m1_fp32.tflite` (M1 FP32) | `986ebc19` |
| `training/models/m2_ptq_int8.tflite` (M2 PTQ INT8, the recommended candidate) | `c331664a` |
| `training/models/m3_pruned_ptq_int8.tflite` (M3 pruned + INT8) | `f220ec06` |
| `data_pipeline/training_stats.npy` | `20f01b2d` |
| `training/dataset.npz` (v2 dataset) | `d2ec8995` |

Key results so far: validation accuracy 100 % for all three variants (n = 86, Critical n = 44);
fresh-stream Critical recall including transitions 86 % / 86 % / 82 % (M1/M2/M3), below the
brief's 95 % target — disclosed in F3, which recommends **M2 in shadow mode** first.

## Benchmark machine attribution

The figures in `optimisation/results/benchmark_results.*` were produced on the **office PC**:
`<FILL IN: model, CPU, RAM, OS>`, TDP estimate 28 W, Python 3.11.4, TensorFlow 2.21.0.
All work from task D2 onward is done on the **personal PC**: Samsung 750XHD, Intel Core Ultra 5
225U, 15.5 GB RAM, Windows 11 Home (moved because Docker Desktop could not be installed on the
office PC). `benchmark_results.json` does not record the machine, so this line is the
record. If the benchmark is re-run on another PC, save it to a **separate** file (e.g.
`benchmark_results_pc2.csv`), record that machine's CPU/RAM/OS here, and make the report say
which machine produced which table. Do not overwrite the original results.

## Status

| Task | Status |
|---|---|
| A1 Constraint analysis, A2 architecture diagram | Done |
| B1 Constraint Triangle, B2 Roofline | Done |
| C1 Simulator (`--anomaly`, `--speedup`, `--anomaly-after/-for`, truth topic) | Done |
| C2 Preprocessing (MA-5, 30 s / 10 s windows, 6 features, frozen stats) | Done — 3σ experiment in `data_pipeline/results/normalisation_experiment.json` |
| C3 Feature-level fusion | Done — justification + MQTT topic tree/QoS in `data_pipeline/mqtt_architecture.md` |
| D1 Dataset + training (> 88 %) | Done |
| D2 Docker + OTA layer-cache demo | Done — `inference/` (see `inference/README.md`) |
| D3 10-stage pipeline mapping | Done — `deployment/pipeline_mapping.md` |
| E1 PSI drift monitor | Done — `monitoring/` (see `monitoring/README.md`) |
| E2 Ansible playbook | Done — `deployment/` (run 2: `changed=0`; see `deployment/README.md`) |
| E3 OTA strategy | Done |
| F1 Three variants, F2 benchmark, F3 recommendation | Done |
| Reports (phase 1, phase 2, final), demo video, submission | **Not started** |

## Remaining plan (work through in order)

Rules: no retraining; reuse the committed models and `training_stats.npy`; commit each task
separately with the `Task Xn - ...` message style used so far.

1. **Verify the new PC.** `setup_check.py` → ALL OK. Record the new PC's CPU / RAM / OS in a
   short `MACHINES.md` (needed for any timing measured there). Run the simulator against
   Mosquitto for 30 s and confirm messages arrive (`& "C:\Program Files\mosquitto\mosquitto_sub.exe" -t "logibridge/#" -v`; it is not on PATH).

2. **C2 mandatory normalisation experiment.** New `data_pipeline/normalisation_experiment.py`:
   evaluate M2 (and M1 for reference) on the validation split with the correct stats, then with
   the mean shifted by +3σ per feature (and optionally −3σ); write
   `data_pipeline/results/normalisation_experiment.json` with accuracy and per-class recall.
   Feeds the Phase 2 report table.

3. **MQTT architecture + C3 fusion — `data_pipeline/mqtt_architecture.md`.** Topic tree as a
   figure: `logibridge/trucks/{truck_id}/{temperature, vibration, door, truth, inference, alerts}`
   plus a drift/status topic. QoS per topic as actually coded in `simulator.py`: temperature 1,
   vibration 0, door 2, truth 0 (evaluation only, never read by the node); choose and justify
   inference / alert QoS (alerts: QoS 1 + persistent session + local log). Justify feature-level
   fusion vs data-level (500 Hz vs 1 Hz rates, different units) and decision-level (loses the
   cross-sensor "combined" signature).

4. **D2 inference service + Docker.** `inference/inference_service.py`: subscribe to the truck's
   sensor topics, run the existing `preprocessing.py` logic, load the model from `MODEL_PATH`
   with the TFLite interpreter, publish to `logibridge/trucks/{truck_id}/inference`; append
   every inference result, alert, door event and drift value to a local SQLite log (sequence
   number, written before publishing); drop temperature readings whose timestamp is not newer
   than the last; publish retained `status` with a Last Will. A sync process replays unsynced
   entries (alerts first) with QoS 1 and marks them synced on PUBACK. Must match
   `data_pipeline/mqtt_architecture.md`. Copy M2 to
   `inference/model.tflite`. `Dockerfile`: `python:3.11-slim`, `pip install` (use
   `tflite-runtime` or `ai-edge-litert`, not full TensorFlow, to keep the image small) **before**
   `COPY model.tflite .`; broker host via env (`host.docker.internal` on Docker Desktop).
   OTA demo: build, swap `model.tflite` for M3 (`MODEL_PATH` unchanged), rebuild, capture that
   only the final layer is rebuilt; compute layer size vs full image × 85 trucks × ₹0.10/MB.

5. **E1 PSI drift monitor.** `monitoring/drift_monitor.py` + `reference_dist.json` from 300
   clean Normal windows, 4 bins, rolling 100, PSI printed every 60 s, alert text exactly
   `[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}`. Two things to settle first:
   - Monitor the **probability of the Normal class**, not the max-class confidence: a confident
     Critical prediction also lands in [0.75, 1.0] and would hide the drift.
   - At one inference per 10 s, 5 minutes is only 30 of the 100 rolling inferences. Either
     check that 30 shifted samples already push PSI > 0.25 (use a small epsilon for empty bins),
     or run the simulator with `--speedup` and state the scaling in the report.
   Run: clean → `--anomaly combined` → clean; save the PSI log (before/during/after values) to
   `monitoring/results/`.

6. **E2 Ansible.** Local registry (`docker run -d -p 127.0.0.1:5000:5000 --name registry registry:2`, localhost only),
   push the image as `localhost:5000/logibridge-inference`. `deployment/logibridge_deploy.yml`
   with exactly the 7 tasks in the brief, run from WSL against localhost
   (`community.docker` collection). Make tasks 4 and 6 idempotent (e.g. only stop/restart when
   the model, reference file or image changed) so run 2 shows `changed=0`. Save both run
   outputs to `deployment/results/`.

7. **D3 10-stage pipeline mapping** — 1–2 LogiEdge-specific sentences per stage of the Module 4
   pipeline (`deployment/pipeline_mapping.md`).

8. **Reports.** `reports/phase1_report.pdf` (A, B), `reports/phase2_report.pdf` (C, D incl.
   normalisation table), `reports/final_report.pdf` — 2,500–3,000 words in the 5 sections and
   word ranges of the brief, built from the existing markdown with `reportlab`; include the
   architecture diagram, MQTT tree, 15-cell benchmark table, Pareto chart, PSI values, Ansible
   result, one technical difficulty, one architectural change, repo URL and video link. Upload
   copy named `GROUPNO_LogiEdge_Final.pdf`. State which machine produced each timing figure.

9. **Demo video (15–20 min)** — script in `demo/demo_script.md`: simulator + broker, inference
   container, OTA layer-cache rebuild with 85-truck saving, PSI injection and recovery, Ansible
   twice (`changed=0`), benchmark/Pareto. Link in `demo/demo_video_link.txt`, tested in
   incognito.

10. **Submission.** Repo private, instructor added as collaborator, LMS: repo URL, video link,
    report PDF.
