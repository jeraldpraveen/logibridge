# Hardware Selection and Roofline Analysis (Tasks B1, B2)

## B1 — Constraint Triangle

Vertices used: **Performance** (latency/accuracy headroom), **Power** (energy and thermal budget), **Cost** (unit and fleet cost).

| | Pi 5 + AI HAT+ | Jetson Orin Nano Super | STM32H7 custom MCU |
|---|---|---|---|
| Price per truck | ₹15,000 | ₹45,000 | ₹3,500 |
| 85-truck pilot | ₹12.75 lakh | ₹38.25 lakh | ₹2.98 lakh |
| 265-truck fleet | ₹39.75 lakh | ₹119.25 lakh | ₹9.28 lakh |
| Power (TDP) | 7.5 W | 15 W | 0.4 W |
| Within 10 W AI budget? | Yes (2.5 W spare) | **No (1.5x over)** | Yes |
| Performance | Plenty | Far more than needed | Enough for a ~1k-parameter MLP |
| Runs Docker/Mosquitto/Ansible? | Yes (Linux) | Yes | **No** |

**Dominant vertex: Power**, closely followed by Cost. The truck supplies 12 V through a DC-DC converter with a 10 W budget for AI. Performance is not the constraint, because our model is a 32-16 MLP over 6 features and needs milliseconds on any of these devices. The 90-second SLA gives a latency budget of about 49 s after the 41 s windowing delay (see A1), and every option meets it by orders of magnitude.

**For Option 1 (Raspberry Pi 5):**
- At 7.5 W it fits the 10 W budget with margin.
- It runs full Linux, so the Mosquitto broker, Docker container, store-and-forward alert log and Ansible-managed OTA from the brief all run unchanged. These are requirements of this project.
- Its fleet cost is ₹12.75 lakh for the pilot and ₹39.75 lakh at scale. It is 3x cheaper than the Jetson.

**Against Option 2 (Jetson Orin Nano Super):**
- 15 W exceeds the 10 W budget, so it would need a bigger converter and thermal design.
- It costs ₹119.25 lakh at 265 trucks, which is ₹79.5 lakh more than the Pi for compute we would not use.

**Against Option 3 (STM32H7):**
- It is the cheapest and lowest power, and it could technically run our tiny model.
- But a bare-metal MCU cannot host Docker, an MQTT broker, TLS uplink with persistent sessions, or Ansible-driven OTA. We would write and maintain all of that by hand, and a bad OTA flash can brick a truck unit in the field.
- The saving is ₹30 lakh at fleet scale, but it removes the operational capability that this pilot is meant to prove.

**Honest caveat:** the 13 TOPS Hailo accelerator is not needed for a ~1,000-parameter MLP, which runs on the Pi's CPU. A Pi 5 without the HAT would be cheaper still. The HAT leaves headroom for scaling to a larger model or camera-based checks later.

## B2 — Arithmetic Intensity and Roofline (Raspberry Pi 5 CPU)

Given: 45 MFLOPs and 18 MB moved per inference; peak 16 GFLOP/s; bandwidth 12 GB/s.

- **Arithmetic intensity** AI = 45×10⁶ FLOP / 18×10⁶ B = **2.5 FLOP/byte**
- **Ridge point** = 16 GFLOP/s / 12 GB/s = **1.33 FLOP/byte**
- AI (2.5) > ridge (1.33), so the model is **compute-bound**.
- Compute time = 45 MFLOP / 16 GFLOP/s = **2.81 ms**. Memory time = 18 MB / 12 GB/s = **1.50 ms**. The slower of the two, 2.81 ms, sets the latency.
- Attainable performance = min(16, 2.5 × 12) = 16 GFLOP/s (on the flat compute roof).

**Optimisation implied by the Roofline:** because the model is compute-bound, latency improves by reducing FLOPs or raising effective compute throughput: structured pruning (fewer MACs) and INT8 quantisation (NEON runs more INT8 operations per cycle). Reducing memory traffic alone (for example, weight compression) would not help until the model is moved left of the ridge point. INT8 also cuts bytes moved about 4x, which would lift AI to roughly 10 FLOP/byte and push the model even further into the compute-bound region.

**Note on the given figures:** 45 MFLOPs and 18 MB are the brief's assumed values. A 6-32-16-3 MLP has only about 800 parameters, so our real model is far smaller and is measured directly in the Task F benchmarks.
