# Task E3 — OTA Strategy Selection

Inputs: model file 280 KB (INT8 TFLite), M2M SIM at ₹0.10/MB, 85 trucks, update every 6 weeks (about 8.7 cycles/year).

## Bandwidth cost per update cycle

280 KB = 0.28 MB per truck. 85 trucks x 0.28 MB = 23.8 MB.

| Strategy | Trucks receiving the file | Data | Cost per cycle |
|---|---|---|---|
| Full replacement | 85 at once | 23.8 MB | **₹2.38** |
| Canary (10 trucks first, then 75) | 10, then 75 if healthy | 2.8 MB, then 21.0 MB | **₹0.28 if rolled back, ₹2.38 if promoted** |
| Shadow mode | 85 (new model runs beside the old one, output logged not acted on) | 23.8 MB, plus shadow-result logs | **₹2.38 + log upload** |

Annual cost at 8.7 cycles is about ₹20.6 for full or canary-promoted updates. Bandwidth is negligible in every case, so the choice rests on **safety and connectivity**, not money.

## Recommendation: Canary (10 trucks first) for routine updates — shadow mode for the first model

**Scope.** This recommends the strategy for the routine six-weekly update cycle. The *first*
model is a separate case: Task F3 found that no current variant meets the >95 % Critical-recall
target across the fresh fault stream (M2: 86 %), so it goes to **shadow mode first**, and to a
canary only once independent temperature and vibration alert rules are in place (see
`optimisation/deployment_recommendation.md`).

**Why canary:**
- A bad model on a cold-chain truck can miss a refrigeration failure and cause a ₹28 lakh loss like the vaccine-spoilage incident. A canary limits the exposure of a faulty model to 10 of 85 trucks (12%).
- If the canary fails, rollback costs only ₹0.28 in data. The independent rule-based temperature and vibration alerts keep running on every truck regardless of which model is installed, so a regressed model never leaves a truck unprotected.
- The 10 canary trucks should be chosen to cover different routes and both signal conditions.

**Promotion gate (absolute thresholds, not "unchanged"):** an unchanged 86 % would still miss
the target, so every condition below is a fixed bar.

*Two subjects are measured and reported separately.* **Model-only** = the classifier's own
predictions. **Alert system** = what actually reaches the cab: the model OR the independent
temperature/vibration rules. The rules can raise the alert system's recall while the model's own
recall stays where it is, so passing an alert-system gate never establishes that the model
itself meets > 95 %.

*Ground truth* is never the guard being evaluated. In simulation it is the simulator's noise-free
truth stream; on canary trucks it is the independent cargo data logger, plus incidents
adjudicated afterwards by operations staff.

| Gate | Subject | Threshold | Scope / evidence |
|---|---|---|---|
| Critical recall before the canary | Alert system | **> 95 %** | Fresh simulator evaluation, all windows incl. transitions, end-time labels, per fault mechanism with support |
| Critical recall before the canary | Model only | **Reported, not gated** | Same evaluation. Currently 86 % for M2; if it stays below 95 %, the report must say the target is met by the alert system, not by the model |
| Missed Critical episodes during the canary | Alert system | **0**, reported with the number of episodes that occurred | Logger-confirmed or adjudicated episodes. If none occurred, the gate is recorded as **untested**, not passed |
| Local cab alert | Alert system | **≤ 90 s** from fault onset | Measured on canary trucks, both connected and inside dead zones |
| Operations-centre receipt while connected | Alert system | **≤ 90 s** from fault onset to acknowledged receipt | Measured only for faults that start and are alerted while coverage is up |
| Operations-centre receipt after a dead zone | Alert system | **≤ 120 s** from reconnection to acknowledged receipt; total alert age also reported | Buffered alerts. Gaps of 35–90 min mean alert age will exceed 90 s, and no store-and-forward design can avoid that |
| False alarms in clean running | Alert system and model, separately | **No higher** than the outgoing configuration's rate | Alerts the logger does not confirm |
| PSI on confidence scores | Model | **≤ 0.25** throughout | Drift monitor (Task E1) on every canary truck |
| Observation period | — | **2 weeks** minimum, including at least one trip through each of the seven documented dead zones | Fleet log |

**Connectivity limitation, stated plainly.** If the brief's 90-second requirement applies to
*remote* receipt at the operations centre even during outages, it is **not met** by this
architecture. During a 35–90 minute gap the only alert inside 90 s is the local one at the cab;
the operations centre learns after reconnection. Meeting a remote 90 s deadline everywhere would
need a second communication path, for example a satellite SMS fallback. That is out of scope
here, and its cost per truck was not assessed.

**Against full replacement:**
- All 85 trucks switch at once, with no chance to see a regression first. A bad model would fail everywhere at the same time, which is unacceptable when safety is the priority.
- Updates over weak rural links can be interrupted mid-download, so a fleet-wide push also risks a fleet-wide partial update.

**Against shadow mode (as the routine strategy):**
- Running two models is **not** a real power problem: one inference takes about 17 µs on the laptop (Task F2), once every 10 s, so even a doubled load is negligible within the 10 W budget. (That is a laptop measurement; Pi timing is still to be done.) The objection is elsewhere.
- Shadow outputs must be uploaded for comparison. With 35–90 minute coverage gaps, results arrive late and out of order, so each comparison takes days, which is slow for a six-weekly cycle.
- A shadow model never acts, so it cannot reveal problems in the alert path itself (delivery, escalation, operator response). The canary tests the whole chain.
- **Where shadow mode is the right choice:** for a model that has not yet met the safety target, such as the current first model. There, exposure must be zero, not 12 %, and the slower comparison is acceptable.

The files are small enough that the 10-truck stage can be repeated quickly if needed. Interrupted downloads are handled by resumable transfer plus a checksum check before the swap, so a truck keeps its old model until the new one is verified.
