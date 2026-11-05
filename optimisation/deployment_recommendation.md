# Task F3 — Deployment Recommendation

**Question from the operations director:** *Which model should we deploy to the 85 trucks?*

**Answer: select M2 (PTQ INT8) as the candidate, but do not yet put any variant in sole charge of
alerts on a live truck.** The brief's >95 % Critical-recall target is met on validation and in
steady conditions, but **not across the fresh fault stream, where M2 reaches 86 %**. M2 should go
next to **shadow evaluation** on trucks (logging only, existing manual procedures stay in force).
Operational canary use on 10 trucks should begin only after independent alert protection for
both temperature and vibration, and timing on the target device, have been demonstrated
(section 5).

All figures refer to the same three model files (SHA-256 `986ebc19…` M1, `c331664a…` M2,
`f220ec06…` M3), the v2 dataset (`d2ec8995…`) and its statistics (`20f01b2d…`).

## 1. The 90-second SLA as a per-inference budget

One inference runs every 10 s, on a 30 s feature window. The SLA is spent almost entirely on
waiting for evidence, not on computing:

| Component | Time |
|---|---|
| Design estimate: 30 s window + 10 s step + inference | ~41 s |
| **Observed** longest decision delay, fresh recordings (temperature-only fault) | **69.8 s from fault onset** (37.4 s of it before the temperature had crossed the Critical threshold) |
| Remaining margin inside 90 s, for the local cab alert (and operations-centre receipt *while connected*) | ~20 s (delivery not measured). During a dead zone the operations centre is reached only after reconnection — see E3 |
| Measured inference, laptop, median of 5 trials | M1 8.4 µs · M2 17.4 µs · M3 17.5 µs |

**Per-inference budget:** even if only 1 % of the remaining ~20 s margin were allowed for
inference, the budget would be 200 ms. Every variant uses less than 0.01 % of that. Inference
latency therefore does not discriminate between the variants. These are laptop timings, not Pi
measurements; the Pi 5 CPU is slower, but a 200 ms budget is about 11,000 times the slowest
laptop figure, so a several-fold slowdown would still leave more than three orders of magnitude
of headroom. That is an extrapolation, which is why Pi timing is a precondition in section 5.

## 2. Flash and SRAM on the chosen device (Raspberry Pi 5, 8 GB)

| | M1 FP32 | M2 PTQ INT8 | M3 Pruned + INT8 |
|---|---|---|---|
| Model file size (storage footprint) | 5.20 KiB | 4.51 KiB | 3.67 KiB |
| Parameters | 803 | 803 | 400 |
| Weight storage | float32 | int8 | int8 |

**Model storage** is not a constraint: a few KiB on an SD card of gigabytes. **Runtime memory**
is a different quantity and was not measured: it includes the TFLite interpreter, its tensor
buffers, the Python inference service, the MQTT client and the 38 s sensor history, all inside a
Docker container. For a model this small the interpreter and service dominate, so the choice of
variant barely changes runtime memory — but the absolute figure must be measured on the Pi
(section 5).

On the STM32H7 alternative (about 1–2 MB Flash, about 1 MB SRAM), the **weights** of all three
would fit, but that alone does not show that complete firmware — runtime, tensor arena, sensor
buffers, networking and OTA support — fits. That was not assessed, since B1 rejects the MCU.

The INT8 variants' practical benefit is **smaller OTA updates**. In absolute terms the saving is
small: M2 is 708 B (13.3 %) smaller than M1.

## 3. Class 2 (Critical) recall — the deciding metric

| Evaluation | M1 | M2 | M3 |
|---|---|---|---|
| Validation, converted `.tflite` (n = 44 Critical) | 44/44 = 100 % | 44/44 = 100 % | 44/44 = 100 % |
| Fresh stream, steady windows (n = 38) | 38/38 = 100 % | 38/38 = 100 % | 38/38 = 100 % |
| **Fresh stream, transitions included (n = 50)** | **43/50 = 86 %** | **43/50 = 86 %** | **41/50 = 82 %** |
| — temperature-only fault (n = 14) | 11/14 | 11/14 | 11/14 |
| — vibration-only fault (n = 18) | 16/18 | 16/18 | 15/18 |
| — combined fault, ~6 °C (n = 18) | 16/18 | 16/18 | 15/18 |
| Under-classified windows (prediction below truth) | 11 | 11 | 13 |
| Over-classified windows (prediction above truth) | 8 | 8 | 6 |
| Fault episodes detected within 90 s | 4/4 | 4/4 | 4/4 |
| False alarms in steady Normal periods | 0/80 | 0/80 | 0/80 |

**Against the brief's >95 % target:** pass on validation and steady windows; **fail on the whole
fresh stream** (82–86 %). The failures are all at class changes. On the temperature ramp the
models predict Normal, then Warning, for about 30 s after the truth has already become Critical
(under-classification of an active fault). At the start of a vibration fault, the first two or
three windows are Normal or Warning before Critical is reached. After a vibration fault ends, a
Critical alert lingers for 10–20 s (over-classification). This lag is the cost of the 30 s
smoothing window, and it is a real operational consequence, not a measurement artefact.

## 4. Recommendation and reasoning

**Select M2 as the candidate model.**

- **Against M3:** it is the smallest file, but in this run it under-classified two more Critical
  windows than M1 and M2 (escalation 10 s later on vibration and combined faults). In cold-chain
  monitoring, late escalation is the costly error. M3 clears alerts faster, but that does not
  offset a later alarm. The scenario count is small, so this is an observation from one run, not
  a proven general difference.
- **M2 rather than M1:** the two produced identical predictions in all 168 fresh-stream windows.
  M1 has lower measured laptop latency (8.4 vs 17.4 µs) and lower estimated energy; M2 has the
  smaller file (708 B, 13.3 %). Because both are far inside the latency budget (section 1), size
  breaks the tie — a small advantage, chosen mainly because it shrinks every future OTA update.
  (The model runs on the Pi's CPU through the TFLite interpreter; the Hailo accelerator is not
  used for it — see B1.)
- **Pareto analysis** ([pareto_chart.png](results/pareto_chart.png)): on size, latency and
  fresh-stream Critical misses, all three variants are non-dominated. This is an
  **unconstrained** comparison: none of the three meets the >95 % whole-stream recall target, so
  the frontier ranks candidates; it does not identify a deployable one. The line joins the three
  measured models as a visual guide only; it says nothing about configurations between them.

## 5. Path to operational use

**Stage 1 — shadow evaluation (now).** M2 runs on trucks with its alerts logged, not acted on;
the existing manual temperature log and driver procedures remain the control. This gathers
real-route evidence without relying on a model that misses the target. (E3 previously argued
against shadow mode on compute grounds; for a model needing microseconds per 10 s window that
argument is weak and has been revised there.)

**Stage 2 — preconditions for an operational canary.** All must be demonstrated first:

1. **Independent alert protection for both mechanisms.** A temperature guard alone is not
   enough. Even if it caught every temperature-only miss and nothing else changed, M2's
   whole-stream recall would be only (14 + 16 + 16) / 50 = **92 %**, because the vibration-only
   and combined misses are vibration-driven. The guard therefore needs a **vibration path** too
   (for example, an alert on short-window RMS above a failure threshold) as well as a
   temperature path. Both rules need checking near the Warning/Critical boundary, where a single
   noisy 1 Hz temperature reading (σ = 0.3 °C) or one vibration sample could cause false alarms;
   a short persistence requirement (for example, 3 consecutive readings) is one option.
2. **Fresh evaluation of model + guards together**, on new recordings, reporting **model-only**
   and **alert-system** (model OR rules) results separately: whole-stream Critical recall per
   mechanism with support, detection delay from both onsets, and false alarms in clean periods.
   Ground truth is the simulator's truth stream, never the rules being evaluated. If the alert
   system passes 95 % but the model alone does not, the report must say so.
3. **Target-device timing**, measured on the Pi 5: inference and runtime memory; fault onset →
   local cab alert; and fault onset → operations-centre receipt while connected. Buffered
   delivery after a dead zone is measured from reconnection (see the E3 gate table).
4. **Wider scenarios:** more fault runs, and cooling-side faults (temperature falling below
   setpoint), which were neither trained nor tested.

**Stage 3 — canary of 10 trucks, then fleet**, using the promotion gate in E3.
