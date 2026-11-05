"""End-to-end test of the containerised edge node with an uplink outage (Task D2 + A1 connectivity).

Timeline (seconds from start):
  0      inference container starts (UPLINK = operations-centre broker on host port 1884, which is DOWN)
  0      simulator starts on the local broker: clean, combined fault from FAULT_AT for FAULT_FOR s
  UP_AT  operations-centre broker container starts -> the sync thread replays the backlog
  END    simulator stops; wait for the backlog to drain

Checks: alerts raised offline during the outage; a (synthetic) drift record is logged and replayed;
the Last Will is delivered on both brokers after an abrupt `docker kill`; backlog replayed with alerts first; every log entry
reaches the operations centre exactly once after de-duplication by sequence number; detection delay
from fault onset (simulator truth topic) to the first alert.
Writes inference/results/e2e_offline_sync.json and e2e_container_log.txt.

Run from the repo root:  .venv\\Scripts\\python inference\\e2e_offline_sync_test.py
Requires: Mosquitto on localhost:1883, Docker, image logibridge-inference:latest.
"""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import paho.mqtt.client as mqtt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "inference" / "results"
TRUCK = "truck01"
FAULT_AT, FAULT_FOR, UP_AT, END = 60, 90, 130, 240
DRIFT_AT = 100        # one synthetic drift message during the outage (the real monitor is Task E1)
NODE, OPS, VOL = "logibridge-e2e-node", "logibridge-e2e-ops", "logibridge-e2e-data"
IMAGE = "logibridge-inference:latest"


def sh(*cmd, check=False):
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def wait_port(port, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.2)
    return False


def recorder(port, cid, topics):
    rec = []
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)
    c.on_connect = lambda c_, u, f, rc, p: c_.subscribe([(t, 1) for t in topics])
    c.on_message = lambda c_, u, m: rec.append((time.time(), m.topic.rsplit("/", 1)[-1], json.loads(m.payload)))
    c.connect("127.0.0.1", port, 30)
    c.loop_start()
    return c, rec


def main():
    for n in (NODE, OPS):
        sh("docker", "rm", "-f", n)
    sh("docker", "volume", "rm", "-f", VOL)

    base = f"logibridge/trucks/{TRUCK}"
    local, lrec = recorder(1883, "e2e-local-recorder", [f"{base}/inference", f"{base}/alerts",
                                                        f"{base}/truth", f"{base}/status"])
    t0 = time.time()
    sh("docker", "run", "-d", "--name", NODE, "-v", f"{VOL}:/data",
       "-e", f"TRUCK_ID={TRUCK}", "-e", "UPLINK_HOST=host.docker.internal", "-e", "UPLINK_PORT=1884",
       IMAGE, check=True)
    sim = subprocess.Popen([sys.executable, str(ROOT / "data_pipeline" / "simulator.py"),
                            "--anomaly", "combined", "--anomaly-after", str(FAULT_AT),
                            "--anomaly-for", str(FAULT_FOR), "--duration", str(END), "--truck-id", TRUCK],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"node + simulator started; uplink DOWN until t={UP_AT}s; fault {FAULT_AT}-{FAULT_AT + FAULT_FOR}s")

    while time.time() - t0 < DRIFT_AT:
        time.sleep(0.5)
    local.publish(f"{base}/drift", json.dumps({"ts": time.time(), "psi": 0.0, "drift_alert": False,
                                               "synthetic": True, "note": "e2e test record"}), qos=1)
    print(f"t={DRIFT_AT}s synthetic drift message published (uplink still down)")
    while time.time() - t0 < UP_AT:
        time.sleep(0.5)
    sh("docker", "run", "-d", "--name", OPS, "-p", "1884:1883", "eclipse-mosquitto:2",
       "mosquitto", "-c", "/mosquitto-no-auth.conf", check=True)
    assert wait_port(1884), "operations-centre broker did not start"
    ops, orec = recorder(1884, "e2e-ops-centre", [f"{base}/#"])
    t_up = time.time()
    print(f"t={t_up - t0:.0f}s uplink UP")

    sim.wait()
    time.sleep(15)                                         # let the backlog drain
    t_kill = time.time()
    sh("docker", "kill", NODE)                             # abrupt: no DISCONNECT -> brokers send the Last Will
    time.sleep(3)
    for c in (local, ops):
        c.loop_stop(); c.disconnect()
    logs = sh("docker", "logs", NODE).stdout
    db = sh("docker", "run", "--rm", "-v", f"{VOL}:/data", IMAGE, "python", "-c",
            "import sqlite3,json;d=sqlite3.connect('/data/logibridge_log.db');"
            "print(json.dumps(d.execute('select kind,count(*),sum(synced) from log group by kind').fetchall()));"
            "print(json.dumps(d.execute('select max(seq) from log').fetchone()))").stdout.splitlines()
    by_kind, (max_seq,) = json.loads(db[0]), json.loads(db[1])

    # ---- analysis
    truth = [(p["ts"], p["fault"]) for _, k, p in lrec if k == "truth"]
    onset = next(ts for ts, f in truth if f)
    alerts_local = [(t, p) for t, k, p in lrec if k == "alerts"]
    first_alert = alerts_local[0][1] if alerts_local else None
    first_critical = next((p for _, p in alerts_local if p["class"] == 2), None)
    up_msgs = [(t, k, p) for t, k, p in orec if k != "status"]
    seqs = [p["seq"] for _, _, p in up_msgs]
    backlog_at_up = [(k, p) for t, k, p in up_msgs if p["ts"] < t_up]
    n_backlog_alerts = sum(1 for k, _ in backlog_at_up if k == "alerts")
    first_kinds = [k for _, k, _ in up_msgs[:max(n_backlog_alerts, 1) + 3]]
    res = {
        "timeline_s": {"fault_at": FAULT_AT, "fault_for": FAULT_FOR, "uplink_up_at": UP_AT, "end": END},
        "local_inferences": sum(1 for _, k, _ in lrec if k == "inference"),
        "local_alerts": [{"t_s": round(t - t0, 1), "event": p["event"], "label": p["label"],
                          "offline": t < t_up} for t, p in alerts_local],
        "detection_delay_s": {
            "first_alert_from_fault_onset": round(first_alert["window_end_ts"] - onset, 1) if first_alert else None,
            "first_critical_from_fault_onset": round(first_critical["window_end_ts"] - onset, 1) if first_critical else None,
            "note": "simulator truth onset to the end of the window that produced the alert; PC timing, "
                    "local broker; excludes cab/uplink delivery"},
        "local_log": {"entries": max_seq, "by_kind_count_synced": by_kind},
        "uplink": {"messages_received": len(up_msgs), "unique_seq": len(set(seqs)),
                   "duplicates": len(seqs) - len(set(seqs)),
                   "missing_seq": sorted(set(range(1, max_seq + 1)) - set(seqs)),
                   "backlog_entries_created_while_down": len(backlog_at_up),
                   "backlog_alerts": n_backlog_alerts,
                   "first_replayed_kinds": first_kinds,
                   "alerts_replayed_first": all(k == "alerts" for k in first_kinds[:n_backlog_alerts])
                   and n_backlog_alerts > 0},
        "uplink_status_messages": [p["state"] for _, k, p in orec if k == "status"],
        "last_will_after_kill": {
            # the Will payload has no model fields; a graceful shutdown message would include them
            "local": any(k == "status" and t >= t_kill and p["state"] == "offline" and "model" not in p
                         for t, k, p in lrec),
            "uplink": any(k == "status" and t >= t_kill and p["state"] == "offline" and "model" not in p
                          for t, k, p in orec)},
        "drift_record_replayed": any(k == "drift" and p.get("synthetic") for _, k, p in up_msgs),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "e2e_offline_sync.json", "w"), indent=2)
    (OUT / "e2e_container_log.txt").write_text(logs, encoding="utf-8")
    sh("docker", "rm", "-f", OPS)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
