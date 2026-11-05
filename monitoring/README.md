# Task E1 — PSI drift monitoring

All runs were on the **personal PC** (`../MACHINES.md`), in real time and without `--speedup`.
The model is the committed M2 (`c331664a…`), the same file as `inference/model.tflite`.

## Files

| File | Purpose |
|---|---|
| `psi.py` | Bins and the PSI formula (shared by every script) |
| `build_reference.py` | Builds `reference_dist.json` from 300 fresh clean Normal windows |
| `reference_dist.json` | The reference distribution (copied to trucks by Ansible in E2) |
| `drift_monitor.py` | Live monitor: PSI every 60 s over the last 100 inferences |
| `psi_sensitivity.py` | Design check: can 30 fault inferences (5 minutes) push PSI over 0.25? |
| `run_drift_demo.py` | Live run: clean → `--anomaly combined` → clean recovery |
| `results/` | Evidence: reference windows, sensitivity curves, PSI log, monitor console, summary |

## Design

**Monitored score: the Normal-class probability, P(Normal), not the top-class confidence.**
A confident Critical prediction and a confident Normal prediction both have a top-class
confidence near 1, so a monitor on max-confidence would not see a fault at all. P(Normal) is
about 1 on clean data and about 0 during a Warning or Critical state, so a shift shows up
directly.

**Bins:** `[0, 0.25)`, `[0.25, 0.50)`, `[0.50, 0.75)`, `[0.75, 1.0]`.
**PSI** = Σ (aᵢ − rᵢ) · ln(aᵢ / rᵢ). Empty bins are floored at ε = 10⁻⁴, because ln(0) is
undefined.
**Monitoring:** a rolling window of the last 100 inferences; PSI is computed every 60 s, printed,
and published to `logibridge/trucks/{truck_id}/drift` (QoS 1). The inference service writes it
to the local durable log, so drift values are also synced after a dead zone. If PSI > 0.25 the
monitor prints `[LOGIBRIDGE DRIFT ALERT] PSI=<value>`.

## Reference distribution — `reference_dist.json`

300 **fresh** windows, not the training data. Five clean simulators (`--anomaly none`, real
time) ran in parallel for about 11 minutes, each with its own featuriser, so the windows took
11 minutes instead of 50. Each stream is an independent clean stream with the brief's
parameters. Every window went through the deployed model using the edge service's own
`Model.predict`.

| Bin | `[0,0.25)` | `[0.25,0.50)` | `[0.50,0.75)` | `[0.75,1.0]` |
|---|---|---|---|---|
| Count (n = 300) | 0 | 0 | 0 | **300** |

All 300 windows were predicted Normal, with P(Normal) = 0.9961, the INT8 output ceiling of
255/256. A reference concentrated in one bin makes the monitor very sensitive. That is the
desired behaviour here, but it is also why ε matters (next section).

## Can PSI cross 0.25 within 5 minutes? — `results/psi_sensitivity.json`

At one inference per 10 s, a 5-minute fault adds at most **30** fault outputs to the
100-inference window. The test mixes (100 − k) clean reference outputs with k fault outputs
(P(Normal) of the 62 Warning/Critical validation windows, all of which fall in `[0, 0.25)`):

| ε (empty-bin floor) | PSI with 30 fault outputs | Crosses 0.25 at | PSI < 0.10 needs |
|---|---|---|---|
| 10⁻⁶ | 3.890 | k = 3 (~30 s) | k ≤ 1 |
| **10⁻⁴ (used)** | **2.508** | **k = 5 (~50 s)** | **k ≤ 1** |
| 10⁻³ | 1.812 | k = 7 (~70 s) | k ≤ 3 |
| 10⁻² | 1.093 | k = 11 (~110 s) | k ≤ 6 |

**Conclusion:** 30 fault outputs push PSI far above 0.25 for any reasonable ε, so **no
`--speedup` is needed**. That matters, because the simulator stamps readings with wall-clock
time. Speeding it up would pack more readings into each 30 s window and multiply the
rate-of-change feature by the speedup factor, so the model would be seeing a different stream
from the one it was trained on.

**ε choice and its consequence:** ε sets how fast an empty reference bin raises PSI as it
starts to fill. With 10⁻⁴, five fault inferences in a row (about 50 s) trigger the alert. The
other side is **recovery**: PSI stays above 0.10 until at most one non-Normal output remains in
the window, which takes about 100 clean inferences (about 17 minutes) after the fault ends.

## Live run: clean → fault → recovery — `results/drift_run_*`

`run_drift_demo.py` ran the inference container (`truck01`), `drift_monitor.py` and the
simulator in real time: **18 min clean → 5 min `--anomaly combined` → clean until PSI < 0.10**.
Fault onset and end times come from the simulator's truth topic. 36 PSI checks were made, one
every 60 s. Raw data is in `results/drift_run_psi.csv`, the monitor's console output in
`results/drift_run_monitor_log.txt`, and the summary in `results/drift_run_summary.json`.

| Phase | Time from start | P(Normal) bins in the window `[0,.25) … [.75,1]` | PSI |
|---|---|---|---|
| Before injection (13 checks; the window fills from 33 to 100) | 0–18 min | 0 / 0 / 0 / 100 | **0.000** at every check |
| Fault, +60 s | 19 min | 4 / 0 / 0 / 96 | 0.241 (just under the threshold) |
| Fault, +120 s | 20 min | 10 / 0 / 0 / 90 | **0.701 → `[LOGIBRIDGE DRIFT ALERT] PSI=0.701`** |
| Fault, +180 s … +300 s | 21–23 min | 16 → 28 fault outputs | 1.208 → 1.747 → 2.314 |
| After the fault (fault outputs still in the window) | 23–35 min | 29 / 0 / 0 / 71 | 2.411 (flat) |
| Fault outputs roll out of the window | 35–39 min | 23 → 5 fault outputs | 1.840 → 1.296 → 0.783 → 0.313 |
| **Recovered** | 40 min | 0 / 0 / 0 / 100 | **0.000** (and at the next check) |

- **Drift alert 119.9 s after fault onset**, inside the 5-minute requirement. PSI then peaked at
  **2.411**, with 29 fault outputs in the window. `[LOGIBRIDGE DRIFT ALERT]` was printed at
  20 consecutive checks.
- **Recovery below 0.10 came 1,019 s (17 min) after the fault ended.** That is the time for
  100 new inferences to replace the window, which matches the prediction above.
- The first check after onset had only 4 fault outputs, because the 30 s window and the
  model's detection lag delay the first non-Normal outputs. That check gave 0.241, consistent
  with the k = 5 threshold in the sensitivity test. The alert therefore came at the second
  check. The worst-case alert delay is about 50 s of fault outputs plus the detection lag,
  plus up to 60 s of waiting for the next check.

## Drift type

The injected fault changes the **input data**: the temperature slope and mean rise, and
vibration RMS steps from 0.45 g to 1.2 g. The relationship between inputs and labels does not
change; a breach is still Critical. In the usual terms this is **data (covariate) drift that
shows up as prediction drift**, not concept drift. PSI is measured on the *output* score, so it
detects prediction drift directly and covariate drift only through the model.

This has a practical consequence. Here the "drift" is a real fault, which the classifier
already alerts on within 10–30 s. PSI adds value for the drift the classifier does *not* flag
as a fault: for example, a slowly miscalibrating temperature sensor (the normalisation
experiment showed that a 3σ shift lowers P(Normal) on some windows), or a new truck or route
whose clean data no longer looks like the reference. Operations should therefore read
classifier alerts and drift alerts together: a drift alert **without** matching classifier
alerts suggests a sensor or data problem, not a cargo problem.
