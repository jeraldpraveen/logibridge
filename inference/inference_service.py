"""LogiEdge edge inference service (Task D2). Runs on the truck, inside Docker.

  local broker  --temperature (QoS1) / vibration (QoS0)-->  preprocessing.py (MA-5, 30 s/10 s
  windows, 6 features, frozen training_stats.npy)  -->  TFLite model (MODEL_PATH)  -->
  logibridge/trucks/{truck_id}/inference  (QoS1, every 10 s)
  logibridge/trucks/{truck_id}/alerts     (QoS1, on Warning/Critical state change) + cab alert

Durable store-and-forward (data_pipeline/mqtt_architecture.md, section 2):
  * every inference result, alert, door event and drift message is appended to a local SQLite
    log BEFORE it is published. Each entry has a sequence number and is hash-chained to the
    previous one (tamper-evident chain of custody).
  * a sync thread replays unsynced entries to the operations-centre broker (UPLINK_HOST),
    alerts first, QoS 1, persistent session; an entry is marked synced only on PUBACK. If
    UPLINK_HOST is empty the log simply accumulates (fully offline operation).
  * status is not logged: retained message + Last Will ('offline').
  * temperature readings whose timestamp is not newer than the last one are dropped
    (QoS 1 may redeliver after a reconnect).

Configuration (environment variables):
  MODEL_PATH   model file (default /app/model.tflite); switches variant without rebuild
  TRUCK_ID     truck01
  MQTT_HOST / MQTT_PORT       local broker (host.docker.internal / 1883)
  UPLINK_HOST / UPLINK_PORT   operations-centre broker; empty = offline only
  LOG_DB       /data/logibridge_log.db
  STATS_PATH   normalisation statistics (default: training_stats.npy next to preprocessing.py)
"""
import hashlib
import json
import os
import signal
import sqlite3
import sys
import threading
import time
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "data_pipeline"):      # container: /app ; repo: ../data_pipeline
    if (p / "preprocessing.py").exists():
        sys.path.insert(0, str(p))
        break
import preprocessing as pp  # noqa: E402

try:                                                  # slim runtime in the container
    from ai_edge_litert.interpreter import Interpreter
    RUNTIME = "ai-edge-litert"
except ImportError:                                   # full TensorFlow on the dev PC
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    RUNTIME = "tensorflow.lite"

LABELS = ["Normal", "Warning", "Critical"]
ALERT_PRIORITY, NORMAL_PRIORITY = 0, 1


def env(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else v


TRUCK = env("TRUCK_ID", "truck01")
MODEL_PATH = Path(env("MODEL_PATH", str(HERE / "model.tflite")))
STATS_PATH = Path(env("STATS_PATH", str(pp.STATS_PATH)))
LOG_DB = Path(env("LOG_DB", "/data/logibridge_log.db"))
MQTT_HOST, MQTT_PORT = env("MQTT_HOST", "host.docker.internal"), int(env("MQTT_PORT", "1883"))
UPLINK_HOST, UPLINK_PORT = env("UPLINK_HOST", ""), int(env("UPLINK_PORT", "1883"))
BASE = f"logibridge/trucks/{TRUCK}"
sha256 = lambda b: hashlib.sha256(b).hexdigest()


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- model
class Model:
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = sha256(self.path.read_bytes())
        self.it = Interpreter(model_path=str(self.path))
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]

    def predict(self, z):
        """z: normalised (6,) feature vector -> (class probabilities float32 (3,), invoke ms)."""
        x = np.asarray(z, np.float32).reshape(1, -1)
        if self.inp["dtype"] == np.int8:                      # same quantisation as validate_tflite.py
            s, zp = self.inp["quantization"]
            x = np.clip(np.round(x / s + zp), -128, 127).astype(np.int8)
        t0 = time.perf_counter()
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        ms = (time.perf_counter() - t0) * 1e3
        y = self.it.get_tensor(self.out["index"]).copy().reshape(-1)
        if self.out["dtype"] == np.int8:                      # INT8 softmax -> probabilities
            s, zp = self.out["quantization"]
            y = (y.astype(np.float32) - zp) * s
        return y.astype(np.float32), ms


# ---------------------------------------------------------------- local durable log
class LocalLog:
    """Append-only SQLite log; hash chain hash_n = sha256(hash_{n-1} + payload_n)."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS log (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, kind TEXT, priority INTEGER,
            payload TEXT, prev_hash TEXT, hash TEXT, synced INTEGER DEFAULT 0)""")
        self.lock = threading.Lock()

    def append(self, kind, payload, priority=NORMAL_PRIORITY):
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self.lock:
            row = self.db.execute("SELECT hash FROM log ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else "0" * 64
            h = sha256((prev + body).encode())
            cur = self.db.execute(
                "INSERT INTO log (ts, kind, priority, payload, prev_hash, hash) VALUES (?,?,?,?,?,?)",
                (time.time(), kind, priority, body, prev, h))
            return cur.lastrowid, h

    def unsynced(self, limit=50):
        with self.lock:
            return self.db.execute(
                "SELECT seq, kind, payload, hash FROM log WHERE synced=0 "
                "ORDER BY priority, seq LIMIT ?", (limit,)).fetchall()

    def mark_synced(self, seq):
        with self.lock:
            self.db.execute("UPDATE log SET synced=1 WHERE seq=?", (seq,))

    def backlog(self):
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM log WHERE synced=0").fetchone()[0]


# ---------------------------------------------------------------- uplink sync
class Syncer(threading.Thread):
    """Replays unsynced log entries to the operations-centre broker (alerts first)."""

    def __init__(self, store, status_fn):
        super().__init__(daemon=True)
        self.store, self.stop, self.status_fn = store, threading.Event(), status_fn
        self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"logibridge-{TRUCK}-sync",
                             clean_session=False)                 # persistent session
        self.c.reconnect_delay_set(1, 30)
        # status is not logged: retained on the uplink, with a Last Will on this connection
        self.c.will_set(f"{BASE}/status", json.dumps({"truck_id": TRUCK, "state": "offline"}),
                        qos=1, retain=True)
        self.c.on_connect = self.on_connect
        self.c.on_disconnect = lambda c, u, f, rc, p: log(f"[SYNC] uplink lost ({rc}); buffering locally")

    def on_connect(self, c, u, f, rc, p):
        log(f"[SYNC] uplink connected ({rc}); backlog {self.store.backlog()} entries")
        c.publish(f"{BASE}/status", self.status_fn("online"), qos=1, retain=True)

    def run(self):
        self.c.connect_async(UPLINK_HOST, UPLINK_PORT, keepalive=30)
        self.c.loop_start()
        while not self.stop.wait(1.0):
            if not self.c.is_connected():
                continue
            sent = 0
            for seq, kind, body, h in self.store.unsynced():
                msg = dict(json.loads(body), seq=seq, hash=h)
                try:
                    info = self.c.publish(f"{BASE}/{kind}", json.dumps(msg), qos=1)
                    info.wait_for_publish(timeout=5)
                except (RuntimeError, ValueError):
                    break
                if not info.is_published():                   # no PUBACK: retry later
                    break
                self.store.mark_synced(seq)
                sent += 1
            if sent:
                log(f"[SYNC] {sent} entries acknowledged by uplink; backlog {self.store.backlog()}")
        self.c.loop_stop()


# ---------------------------------------------------------------- service
class InferenceService:
    def __init__(self):
        self.model = Model(MODEL_PATH)
        self.stats = pp.load_training_stats(STATS_PATH)
        self.store = LocalLog(LOG_DB)
        self.feat = pp.StreamFeaturizer()
        self.last_temp_ts = None
        self.prev_class = None
        self.n_inf = self.n_dup = 0
        self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"logibridge-{TRUCK}-inference")
        self.c.will_set(f"{BASE}/status", json.dumps({"truck_id": TRUCK, "state": "offline"}),
                        qos=1, retain=True)
        self.c.on_connect = self.on_connect
        self.c.on_message = self.on_message
        self.c.reconnect_delay_set(1, 10)

    def status(self, state):
        return json.dumps({"truck_id": TRUCK, "state": state, "ts": time.time(),
                           "model": MODEL_PATH.name, "model_sha256": self.model.sha256,
                           "stats_sha256": sha256(STATS_PATH.read_bytes()), "runtime": RUNTIME})

    def on_connect(self, c, userdata, flags, rc, props):
        log(f"local broker {MQTT_HOST}:{MQTT_PORT} connected ({rc})")
        c.subscribe([(f"{BASE}/temperature", 1), (f"{BASE}/vibration", 0),
                     (f"{BASE}/door", 2), (f"{BASE}/drift", 1)])
        c.publish(f"{BASE}/status", self.status("online"), qos=1, retain=True)

    def on_message(self, c, userdata, msg):
        try:
            d = json.loads(msg.payload)
            leaf = msg.topic.rsplit("/", 1)[-1]
            if leaf == "temperature":
                ts = float(d["ts"])
                if self.last_temp_ts is not None and ts <= self.last_temp_ts:
                    self.n_dup += 1                            # QoS 1 redelivery / out of order
                    return
                self.last_temp_ts = ts
                f = self.feat.add_temperature(ts, float(d["value"]))
                if f is not None:
                    self.on_window(ts, f)
            elif leaf == "vibration":
                self.feat.add_vibration(float(d["ts"]), float(d["value"]))
            elif leaf == "door":
                self.store.append("door", dict(d, truck_id=TRUCK))
            elif leaf == "drift":
                self.store.append("drift", dict(d, truck_id=TRUCK))
        except Exception as e:                                 # never let one bad message kill the node
            log(f"[WARN] bad message on {msg.topic}: {e!r}")

    def on_window(self, window_end_ts, features):
        probs, ms = self.model.predict(pp.normalise(features, self.stats))
        cls = int(np.argmax(probs))
        self.n_inf += 1
        res = {"truck_id": TRUCK, "ts": time.time(), "window_end_ts": window_end_ts,
               "class": cls, "label": LABELS[cls], "probs": [round(float(p), 4) for p in probs],
               "confidence": round(float(probs[cls]), 4), "infer_ms": round(ms, 4),
               "model_sha256": self.model.sha256[:12]}
        seq, h = self.store.append("inference", res)           # log first, then publish
        self.c.publish(f"{BASE}/inference", json.dumps(dict(res, seq=seq, hash=h)), qos=1)
        log(f"[{TRUCK}] {LABELS[cls]:<8} p={res['probs']} ({ms:.3f} ms)")

        prev = self.prev_class
        self.prev_class = cls
        if cls == prev or (prev is None and cls == 0) or (prev in (None, 0) and cls == 0):
            return
        event = ("cleared" if cls == 0 else "raised" if prev in (None, 0)
                 else "escalated" if cls > prev else "downgraded")
        alert = {"truck_id": TRUCK, "ts": time.time(), "window_end_ts": window_end_ts,
                 "event": event, "class": cls, "label": LABELS[cls],
                 "prev_class": prev, "prev_label": None if prev is None else LABELS[prev],
                 "confidence": res["confidence"], "model_sha256": res["model_sha256"]}
        seq, h = self.store.append("alerts", alert, ALERT_PRIORITY)
        self.c.publish(f"{BASE}/alerts", json.dumps(dict(alert, seq=seq, hash=h)), qos=1)
        log(f"[CAB ALERT] {event.upper()}: {LABELS[cls]} (was {alert['prev_label']}), "
            f"confidence {res['confidence']:.2f}")

    def run(self):
        log(f"LogiEdge inference | truck {TRUCK} | model {MODEL_PATH} sha256 {self.model.sha256[:12]} "
            f"({self.model.inp['dtype'].__name__} in) | runtime {RUNTIME} | log {LOG_DB}")
        syncer = None
        if UPLINK_HOST:
            syncer = Syncer(self.store, self.status)
            syncer.start()
            log(f"[SYNC] uplink {UPLINK_HOST}:{UPLINK_PORT} (QoS 1, persistent session)")
        else:
            log("[SYNC] no UPLINK_HOST: offline mode, entries accumulate in the local log")
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *a: stop.set())
        signal.signal(signal.SIGINT, lambda *a: stop.set())
        self.c.connect_async(MQTT_HOST, MQTT_PORT, keepalive=30)
        self.c.loop_start()
        while not stop.wait(1.0):
            pass
        log(f"stopping: {self.n_inf} inferences, {self.n_dup} duplicate temperature readings dropped, "
            f"backlog {self.store.backlog()}")
        self.c.publish(f"{BASE}/status", self.status("offline"), qos=1, retain=True).wait_for_publish(2)
        self.c.disconnect()
        self.c.loop_stop()
        if syncer:
            syncer.stop.set()
            syncer.join(3)


if __name__ == "__main__":
    InferenceService().run()
