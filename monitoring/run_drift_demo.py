"""Task E1 live run: clean -> drift injection (--anomaly combined) -> recovery, all real time.

  inference container (logibridge-inference:latest, truck01)  ->  .../inference
  drift_monitor.py (PSI every 60 s over the last 100 inferences)  ->  console + .../drift
  simulator.py: clean for T_CLEAN s (fills the 100-inference window with clean outputs),
                combined fault for T_FAULT s, then clean until PSI < 0.10 (or T_RECOVER_MAX)
No --speedup: psi_sensitivity.py showed that the 30 fault inferences available in 5 minutes are
enough, and speeding up the simulator would distort the window features.

Run from the repo root:  .venv\\Scripts\\python monitoring\\run_drift_demo.py
Writes monitoring/results/drift_run_psi.csv, drift_run_monitor_log.txt, drift_run_summary.json.
"""
import csv
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "monitoring" / "results"
TRUCK, NODE, IMAGE = "truck01", "logibridge-drift-node", "logibridge-inference:latest"
T_CLEAN, T_FAULT, T_RECOVER_MAX = 1080, 300, 1500


def main():
    RES.mkdir(parents=True, exist_ok=True)
    subprocess.run(["docker", "rm", "-f", NODE], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", NODE, "-e", f"TRUCK_ID={TRUCK}", IMAGE],
                   capture_output=True, check=True)

    truth = []                                         # (ts, fault) from the simulator's truth topic
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="drift-demo-truth")
    c.on_message = lambda c_, u, m: truth.append((json.loads(m.payload)["ts"], json.loads(m.payload)["fault"]))
    c.connect("localhost", 1883, 30)
    c.subscribe(f"logibridge/trucks/{TRUCK}/truth", 0)
    c.loop_start()

    csv_path, log_path = RES / "drift_run_psi.csv", RES / "drift_run_monitor_log.txt"
    mon = subprocess.Popen([sys.executable, "-u", str(ROOT / "monitoring" / "drift_monitor.py"),
                            "--truck-id", TRUCK, "--log", str(csv_path)],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_lines = []

    def pump():
        for line in mon.stdout:
            log_lines.append(line)
            print("  monitor | " + line, end="", flush=True)
    threading.Thread(target=pump, daemon=True).start()

    t0 = time.time()
    sim = subprocess.Popen([sys.executable, str(ROOT / "data_pipeline" / "simulator.py"), "--truck-id", TRUCK,
                            "--anomaly", "combined", "--anomaly-after", str(T_CLEAN), "--anomaly-for", str(T_FAULT),
                            "--duration", str(T_CLEAN + T_FAULT + T_RECOVER_MAX)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"clean {T_CLEAN} s -> combined fault {T_FAULT} s -> clean until PSI < 0.10 (max {T_RECOVER_MAX} s)")

    def rows():
        if not csv_path.exists():
            return []
        with open(csv_path) as fh:
            return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]

    while sim.poll() is None:                          # stop early once recovered after the fault
        time.sleep(5)
        el = time.time() - t0
        if el > T_CLEAN + T_FAULT and any(r["psi"] > 0.25 for r in rows()):
            post = [r for r in rows() if r["ts"] > t0 + T_CLEAN + T_FAULT]
            if post and post[-1]["psi"] < 0.10:
                time.sleep(65)                         # one more check below 0.10
                break
    sim.terminate(); time.sleep(2)
    mon.terminate(); time.sleep(1)
    subprocess.run(["docker", "stop", NODE], capture_output=True)
    c.loop_stop(); c.disconnect()
    log_path.write_text("".join(log_lines), encoding="utf-8")

    r = rows()
    onset = next(ts for ts, f in truth if f)
    end = next(ts for ts, f in truth if not f and ts > onset)
    before = [x for x in r if x["ts"] < onset]
    during = [x for x in r if onset <= x["ts"] <= end]
    after = [x for x in r if x["ts"] > end]
    first_alert = next((x for x in r if x["ts"] >= onset and x["psi"] > 0.25), None)
    first_rec = next((x for x in after if x["psi"] < 0.10), None)
    rel = lambda x: round(x["ts"] - t0, 1)
    summary = {
        "machine": "personal PC (MACHINES.md); real time, no --speedup",
        "timeline_s_from_start": {"fault_onset": round(onset - t0, 1), "fault_end": round(end - t0, 1)},
        "checks": len(r),
        "before_injection": {"psi_last": before[-1]["psi"] if before else None,
                             "psi_max": max((x["psi"] for x in before), default=None),
                             "n_last": before[-1]["n"] if before else None},
        "during_injection": {"psi_values": [x["psi"] for x in during],
                             "psi_max": max((x["psi"] for x in during), default=None)},
        "first_drift_alert": None if not first_alert else {
            "t_s": rel(first_alert), "psi": first_alert["psi"],
            "seconds_after_fault_onset": round(first_alert["ts"] - onset, 1),
            "within_5_min": first_alert["ts"] - onset <= 300},
        "after_fault": {"psi_values": [x["psi"] for x in after],
                        "first_below_0.10": None if not first_rec else {
                            "t_s": rel(first_rec), "psi": first_rec["psi"],
                            "seconds_after_fault_end": round(first_rec["ts"] - end, 1)}},
        "psi_final": r[-1]["psi"] if r else None,
        "alert_lines_printed": sum("[LOGIBRIDGE DRIFT ALERT]" in l for l in log_lines),
    }
    json.dump(summary, open(RES / "drift_run_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
