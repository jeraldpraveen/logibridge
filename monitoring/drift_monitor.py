"""LogiEdge PSI drift monitor (Task E1). Runs on the truck next to the inference container.

Subscribes to logibridge/trucks/{truck_id}/inference (QoS 1), keeps the Normal-class probability
of the last 100 inferences, and every 60 s computes PSI against reference_dist.json (4 bins).
  * prints the current PSI value every check
  * PSI > 0.25  ->  prints  [LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}
  * publishes logibridge/trucks/{truck_id}/drift (QoS 1); the inference service writes it to the
    local durable log, so drift values reach the operations centre after a dead zone too.

Run:  .venv\\Scripts\\python monitoring\\drift_monitor.py --truck-id truck01
      [--log monitoring/results/psi_log.csv]
"""
import argparse
import csv
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import paho.mqtt.client as mqtt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import psi as P  # noqa: E402

ALERT, RECOVERED = 0.25, 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truck-id", default="truck01")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--reference", default=str(HERE / "reference_dist.json"))
    ap.add_argument("--window", type=int, default=100, help="rolling window, inferences")
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between PSI checks")
    ap.add_argument("--min-n", type=int, default=30, help="first PSI once this many inferences are held")
    ap.add_argument("--log", default=None, help="optional CSV of every check")
    a = ap.parse_args()

    ref = json.load(open(a.reference))
    ref_props = ref["proportions"]
    base = f"logibridge/trucks/{a.truck_id}"
    win, seen, lock = deque(maxlen=a.window), deque(maxlen=1000), threading.Lock()

    def on_message(c, u, m):
        d = json.loads(m.payload)
        key = d.get("seq", d.get("window_end_ts"))
        with lock:
            if key in seen:                                   # QoS 1 duplicate
                return
            seen.append(key)
            win.append(float(d["probs"][0]))

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"logibridge-{a.truck_id}-drift")
    c.on_connect = lambda c_, u, f, rc, p: c_.subscribe(f"{base}/inference", qos=1)
    c.on_message = on_message
    c.reconnect_delay_set(1, 10)
    c.connect(a.host, a.port, 30)
    c.loop_start()

    writer = None
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        fh = open(a.log, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["ts", "n", "psi", "bin0", "bin1", "bin2", "bin3", "drift_alert"])
    print(f"PSI monitor | truck {a.truck_id} | reference {Path(a.reference).name} "
          f"(n={ref['n_windows']}, props {ref_props}, model {ref['model_sha256'][:8]}) | "
          f"window {a.window}, every {a.interval:.0f} s, alert > {ALERT}", flush=True)
    state = "ok"
    try:
        while True:
            time.sleep(a.interval)
            with lock:
                vals = list(win)
            if len(vals) < a.min_n:
                print(f"{time.strftime('%H:%M:%S')} warming up: {len(vals)}/{a.window} inferences", flush=True)
                continue
            counts, props = P.proportions(vals)
            value = P.psi(ref_props, props)
            alert = value > ALERT
            print(f"{time.strftime('%H:%M:%S')} PSI={value:.3f}  n={len(vals)}  P(Normal) bins={counts.tolist()}",
                  flush=True)
            if alert:
                print(f"[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}", flush=True)
                state = "drift"
            elif state == "drift" and value < RECOVERED:
                print(f"{time.strftime('%H:%M:%S')} recovered: PSI={value:.3f} < {RECOVERED}", flush=True)
                state = "ok"
            c.publish(f"{base}/drift", json.dumps({
                "truck_id": a.truck_id, "ts": time.time(), "psi": round(value, 4), "n": len(vals),
                "bin_counts": counts.tolist(), "drift_alert": alert, "threshold": ALERT}), qos=1)
            if writer:
                writer.writerow([round(time.time(), 1), len(vals), round(value, 4), *counts.tolist(), int(alert)])
                fh.flush()
    except KeyboardInterrupt:
        pass
    finally:
        c.loop_stop(); c.disconnect()


if __name__ == "__main__":
    main()
