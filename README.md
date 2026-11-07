# LogiEdge — Cold-Chain Edge AI Platform

AIML ZG535 Machine Learning on Edge, mini-project (EC-I). An edge AI pipeline for FreightBridge's
85 refrigerated trucks: a sensor simulator publishes temperature, vibration and door events to a
local Mosquitto broker; the truck's edge node extracts a 6-value feature vector every 10 s and
classifies the cargo state as Normal / Warning / Critical with a small TFLite MLP, offline.

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
Dell Latitude 3420, Intel Core i7-1165G7, 32 GB RAM, Windows 11 Enterprise 23H2, TDP estimate 28 W (the CPU's upper configurable TDP; range 12–28 W), Python 3.11.4, TensorFlow 2.21.0.
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
