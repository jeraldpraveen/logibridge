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

## Recommendation: Canary (10 trucks first)

**Why canary:**
- A bad model on a cold-chain truck can miss a refrigeration failure and cause a ₹28 lakh loss like the vaccine-spoilage incident. A canary limits the exposure of a faulty model to 10 of 85 trucks (12%).
- If the canary fails, rollback costs only ₹0.28 in data. Each canary truck also still runs the old model's safety rules as a fallback.
- The 10 canary trucks should be chosen to cover different routes and both signal conditions, with a promotion gate: Critical recall unchanged, no PSI alert, and no rise in alert rate for a set period.

**Against full replacement:**
- All 85 trucks switch at once, with no chance to see a regression first. A bad model would fail everywhere at the same time, which is unacceptable when safety is the priority.
- Updates over weak rural links can be interrupted mid-download, so a fleet-wide push also risks a fleet-wide partial update.

**Against shadow mode:**
- It is the safest on paper, but it needs the new and old models to run on every truck at once, doubling inference compute within a 10 W budget.
- It needs the shadow outputs uploaded for comparison. With 35–90 minute coverage gaps, results arrive late and out of order, so the comparison would take many days to complete.
- Its extra safety over canary is small for a 280 KB model that can be rolled back cheaply.

The files are small enough that the 10-truck stage can be repeated quickly if needed. Interrupted downloads are handled by resumable transfer plus a checksum check before the swap, so a truck keeps its old model until the new one is verified.
