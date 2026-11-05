# Task D2 — Containerised inference service and OTA layer-cache demo

All figures below were produced on the **personal PC** (see `../MACHINES.md`), with Docker Desktop
29.9.2. No model was retrained: `model.tflite` is the committed M2 (`c331664a…`).

## Files

| File | Purpose |
|---|---|
| `inference_service.py` | Edge service: subscribe → preprocess → TFLite → publish; local durable log; uplink sync |
| `Dockerfile` | `python:3.11-slim`; all pip layers before `COPY model.tflite .` (the last layer) |
| `requirements.txt` | `ai-edge-litert` (slim TFLite runtime), `numpy`, `paho-mqtt`, all pinned |
| `model.tflite` | Copy of `training/models/m2_ptq_int8.tflite` (recommended variant, F3) |
| `check_runtime_equivalence.py` | Proves the container runtime gives the same outputs as `tf.lite` |
| `e2e_offline_sync_test.py` | Live test: fault during an uplink outage, then replay |
| `ota_layer_demo.py` | Model-only rebuild, registry push, 85-truck bandwidth calculation |
| `results/` | JSON evidence and logs from the three scripts |

## Build and run

From the repository root (BuildKit is the default in Docker Desktop):

```
docker build --build-context pipeline=data_pipeline -t logibridge-inference:latest inference
docker run -d --name logibridge-inference -v logibridge-data:/data -e TRUCK_ID=truck01 logibridge-inference:latest
.venv\Scripts\python data_pipeline\simulator.py --anomaly combined --anomaly-after 60
docker logs -f logibridge-inference
```

`--build-context pipeline=data_pipeline` lets the image take `preprocessing.py` and
`training_stats.npy` from `data_pipeline/` (one copy of each in the repository), while
`model.tflite` is copied from the `inference/` context exactly as the brief specifies.

**Switching variants without a rebuild (`MODEL_PATH`):**

```
docker run -d --name logibridge-inference -v logibridge-data:/data ^
  -v %cd%\training\models:/models:ro -e MODEL_PATH=/models/m3_pruned_ptq_int8.tflite logibridge-inference:latest
```

The service logs the loaded model's SHA-256 at start-up and puts it in every inference,
alert and status message. In Task E2, Ansible bind-mounts `/opt/logibridge/model.tflite` and
sets `MODEL_PATH` the same way.

**Environment variables:** `MODEL_PATH`, `TRUCK_ID`, `MQTT_HOST` / `MQTT_PORT` (local broker,
default `host.docker.internal:1883`), `UPLINK_HOST` / `UPLINK_PORT` (operations-centre broker;
empty means offline only), `LOG_DB` (default `/data/logibridge_log.db`), `STATS_PATH`.

## What the service does

It implements the design in `data_pipeline/mqtt_architecture.md`:

- Subscribes to `temperature` (QoS 1), `vibration` (QoS 0), `door` (QoS 2) and `drift` (QoS 1).
  Temperature readings whose timestamp is not newer than the last one are dropped.
- Every 10 s it builds the 6-feature window with `preprocessing.py`, normalises it with the
  frozen `training_stats.npy`, runs the model and publishes
  `logibridge/trucks/{truck_id}/inference` (QoS 1). INT8 outputs are converted back to
  probabilities (scale 1/256), so `probs` are real class probabilities, which the drift
  monitor uses.
- On a Warning/Critical state change it publishes `alerts` (QoS 1) with the event `raised`,
  `escalated`, `downgraded` or `cleared`, and prints a `[CAB ALERT]` line.
- Every inference, alert, door event and drift message is appended to a SQLite log *before*
  it is published. Each entry has a sequence number and a SHA-256 hash chained to the previous
  entry, which makes the chain-of-custody record tamper-evident.
- A sync thread replays unsynced entries to `UPLINK_HOST`, alerts first, QoS 1, on a persistent
  session. An entry is marked synced only on PUBACK.
- `status` is retained and has a Last Will (`offline`), on both the local broker and the
  uplink.

## Results

### Container runtime gives identical results — `results/runtime_equivalence.json`

The container uses `ai-edge-litert` 2.3.0 instead of full TensorFlow. On the 86 validation
windows, all three committed models give **identical predictions and identical probabilities
(maximum difference 0.0)** to the `tf.lite` interpreter used for every F2/F3 result. The
benchmark and recall results therefore carry over to the container unchanged.

### OTA layer-cache demo — `results/ota_layer_cache.json`, `results/ota_build_m3.txt`

`ota_layer_demo.py` builds the image with M2, replaces **only** `model.tflite` with M3 and
rebuilds:

| Step | Build 2 (only the model changed) |
|---|---|
| `[1/7] FROM python:3.11-slim` | base (not rebuilt) |
| `[2/7] WORKDIR /app` | CACHED |
| `[3/7] COPY requirements.txt .` | CACHED |
| `[4/7] RUN pip install -r requirements.txt` (157 MB uncompressed) | CACHED |
| `[5/7] COPY preprocessing.py training_stats.npy` | CACHED |
| `[6/7] COPY inference_service.py .` | CACHED |
| `[7/7] COPY model.tflite .` | **REBUILT** |

After both tags are pushed to the local registry (`registry:2`, bound to `127.0.0.1:5000` only;
also used by Ansible in E2), the manifests show **1 of 10 layers changed**. Registry (compressed) sizes, which is what
a truck downloads:

| | Per truck | 85 trucks | Cost at ₹0.10/MB |
|---|---|---|---|
| Full image | 95.30 MB | 8,100.65 MB | ₹810.07 |
| Model-only update (changed layer + image config) | 9.7 KB | 0.83 MB | ₹0.08 |
| **Saving per update** | | **8,099.8 MB (99.99 %)** | **₹809.98** |
| Saving per year (6-weekly, 8.67 cycles) | | 70,198 MB | ₹7,019.85 |

These are *our* image sizes, measured. Task E3 used the brief's 280 KB model figure; the
committed M2 is 4.5 KB. The demo removes earlier `:m3` tags and the cached `COPY model.tflite`
results first, so each run shows the rebuild that a genuinely new model would cause.

### Live test with an uplink outage — `results/e2e_offline_sync.json`

`e2e_offline_sync_test.py` starts the container with the operations-centre broker **down**,
runs the simulator (clean, combined fault from 60 s to 150 s, 240 s in total), and starts the
operations-centre broker (a second Mosquitto container on port 1884) at 130 s. At the end the node is
killed abruptly (`docker kill`) to test the Last Will.

| Check | Result |
|---|---|
| Inferences published locally | 21 (first full window at 30 s, then every 10 s) |
| Critical alert raised **while the uplink was down** | Yes, `[CAB ALERT] RAISED: Critical`, 21.2 s after fault onset (window end time). An earlier run of the same test gave 10.1 s. The spread is how the fixed 10 s window grid falls against the fault onset. |
| Alert cleared after the fault | Yes, `CLEARED: Normal` |
| Backlog when the sync connection came up | 19 entries; acknowledged within 2 s |
| Replay order | The alert first, then inference and door entries oldest first |
| Local log → operations centre | 33 of 33 entries (21 inference, 2 alerts, 9 door, 1 drift), **0 missing, 0 duplicates** |
| Drift record (synthetic, published at 100 s during the outage; the real monitor is Task E1) | Logged and replayed |
| Last Will after an abrupt `docker kill` (no DISCONNECT) | `offline` Will delivered by **both** the local broker and the operations-centre broker |

The detection times are PC measurements on a local broker, from the simulator's truth onset to
the end of the window that produced the alert. They fall inside F3's observed range and the
90 s SLA. They exclude cab and uplink delivery.

One observation for the report: the sync connection came up about 20 s after the operations-centre
broker did, because the MQTT client's reconnect back-off is capped at 30 s. After a dead zone,
alerts can therefore reach the operations centre up to about 30 s after coverage returns.
Lowering the cap shortens that delay but costs more connection attempts while a truck is still
in a dead zone.
