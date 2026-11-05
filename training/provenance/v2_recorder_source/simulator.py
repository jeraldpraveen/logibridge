"""LogiEdge cold-chain sensor simulator (Task C1).

Publishes three streams to a local Mosquitto broker:

  logibridge/trucks/{truck_id}/temperature  1 Hz    N(4.0, 0.3) degC; setpoint 4 degC
  logibridge/trucks/{truck_id}/vibration    0.5 Hz  N(0.45, 0.05) g compressor RMS
  logibridge/trucks/{truck_id}/door         events  'OPEN' / 'CLOSE'

Anomaly modes (--anomaly):
  none        normal operation
  temp_drift  temperature drifts +0.08 degC per reading (linear)
  vibration   vibration steps to N(1.2, 0.15) g (bearing wear)
  combined    both anomalies at once

--anomaly-after N delays the fault by N seconds, so one run can go
clean -> faulty (used for the drift-injection demo). --anomaly-for M ends the
fault after M seconds of fault (recovery demo).
--speedup is for quick tests only; real data collection uses real time.
"""
import argparse
import json
import random
import threading
import time

import paho.mqtt.client as mqtt

SETPOINT = 4.0
TEMP_SIGMA = 0.3
DRIFT_PER_READING = 0.08
VIB_NORMAL = (0.45, 0.05)
VIB_FAULT = (1.2, 0.15)


def topic(truck_id, stream):
    return f"logibridge/trucks/{truck_id}/{stream}"


class Simulator:
    def __init__(self, args):
        self.args = args
        self.t0 = time.time()
        self.stop = threading.Event()
        self.drift_offset = 0.0
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=f"sim-{args.truck_id}-{random.randint(0, 9999)}")
        self.client.connect(args.host, args.port, keepalive=30)
        self.client.loop_start()

    def sleep(self, seconds):
        return not self.stop.wait(seconds / self.args.speedup)

    def elapsed(self):
        # simulated seconds since start (honours --speedup)
        return (time.time() - self.t0) * self.args.speedup

    def fault_active(self):
        a = self.args
        if a.anomaly == "none":
            return False
        e = self.elapsed()
        if e < a.anomaly_after:
            return False
        if a.anomaly_for and e >= a.anomaly_after + a.anomaly_for:
            return False
        return True

    def publish(self, stream, payload, qos):
        self.client.publish(topic(self.args.truck_id, stream),
                            json.dumps(payload), qos=qos)

    def temperature_loop(self):
        a = self.args
        while not self.stop.is_set():
            if self.fault_active() and a.anomaly in ("temp_drift", "combined"):
                self.drift_offset += DRIFT_PER_READING
                if a.drift_cap is not None:
                    self.drift_offset = min(self.drift_offset, a.drift_cap)
            else:
                self.drift_offset = 0.0
            value = random.gauss(SETPOINT, TEMP_SIGMA) + self.drift_offset
            now = time.time()
            # ground truth for evaluation only (noise-free simulator state); the edge
            # node never subscribes to this topic
            self.publish("truth",
                         {"ts": now, "offset": round(self.drift_offset, 4),
                          "vib_fault": bool(self.fault_active() and a.anomaly in ("vibration", "combined")),
                          "fault": bool(self.fault_active())}, qos=0)
            self.publish("temperature",
                         {"ts": now, "value": round(value, 4)}, qos=1)
            if not self.sleep(1.0):
                break

    def vibration_loop(self):
        a = self.args
        while not self.stop.is_set():
            faulty = self.fault_active() and a.anomaly in ("vibration", "combined")
            mu, sd = VIB_FAULT if faulty else VIB_NORMAL
            self.publish("vibration",
                         {"ts": time.time(), "value": round(random.gauss(mu, sd), 4)},
                         qos=0)
            if not self.sleep(2.0):  # 0.5 Hz
                break

    def door_loop(self):
        is_open = False
        while not self.stop.is_set():
            # a door stays shut ~20-60 s between events, open ~3-10 s
            if not self.sleep(random.uniform(3, 10) if is_open else random.uniform(20, 60)):
                break
            is_open = not is_open
            self.publish("door",
                         {"ts": time.time(), "event": "OPEN" if is_open else "CLOSE"},
                         qos=2)

    def run(self):
        threads = [threading.Thread(target=f, daemon=True)
                   for f in (self.temperature_loop, self.vibration_loop, self.door_loop)]
        for t in threads:
            t.start()
        print(f"[sim] truck={self.args.truck_id} anomaly={self.args.anomaly} "
              f"after={self.args.anomaly_after}s duration={self.args.duration or 'inf'}s "
              f"speedup={self.args.speedup}x", flush=True)
        try:
            while not self.stop.is_set():
                if self.args.duration and self.elapsed() >= self.args.duration:
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        self.stop.set()
        time.sleep(0.3)
        self.client.loop_stop()
        self.client.disconnect()
        print("[sim] stopped", flush=True)


def main():
    p = argparse.ArgumentParser(description="LogiEdge cold-chain sensor simulator")
    p.add_argument("--anomaly", choices=["none", "temp_drift", "vibration", "combined"],
                   default="none")
    p.add_argument("--truck-id", default="truck01")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--duration", type=float, default=0,
                   help="seconds to run (0 = until Ctrl-C)")
    p.add_argument("--anomaly-after", type=float, default=0,
                   help="seconds of clean data before the fault starts")
    p.add_argument("--anomaly-for", type=float, default=0,
                   help="seconds the fault lasts (0 = forever)")
    p.add_argument("--drift-cap", type=float, default=None,
                   help="optional: stop the +0.08 degC/reading drift at this offset (degC). "
                        "Default = uncapped (spec behaviour)")
    p.add_argument("--speedup", type=float, default=1.0,
                   help="time compression, for quick tests only")
    Simulator(p.parse_args()).run()


if __name__ == "__main__":
    main()
