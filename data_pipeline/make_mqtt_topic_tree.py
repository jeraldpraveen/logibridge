"""Draws the LogiEdge MQTT topic tree (Task C / Final Report section 2 figure).

Run:  .venv\\Scripts\\python data_pipeline\\make_mqtt_topic_tree.py
Writes data_pipeline/mqtt_topic_tree.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "mqtt_topic_tree.png"

IN_C, OUT_C, EVAL_C = ("#E3F2FD", "#1565C0"), ("#FDECEA", "#B3261E"), ("#F1F1F1", "#757575")

# (leaf, QoS, retained, rate, publisher -> subscribers, leaves the truck?, colours)
LEAVES = [
    ("temperature", 1, False, "1 Hz", "sensor/simulator -> inference service", False, IN_C),
    ("vibration", 0, False, "0.5 Hz (RMS)", "sensor/simulator -> inference service", False, IN_C),
    ("door", 2, False, "event", "sensor/simulator -> inference service (local log)", True, IN_C),
    ("truth", 0, False, "1 Hz", "simulator -> evaluation scripts ONLY", False, EVAL_C),
    ("inference", 1, False, "every 10 s", "inference service -> drift monitor, cab, sync", True, OUT_C),
    ("alerts", 1, False, "on change", "inference service -> cab display, sync", True, OUT_C),
    ("drift", 1, False, "every 60 s", "drift monitor -> sync", True, OUT_C),
    ("status", 1, True, "on change + LWT", "inference service -> ops dashboard (live)", True, OUT_C),
]

fig, ax = plt.subplots(figsize=(13, 8.4))
ax.set_xlim(0, 13); ax.set_ylim(-1.9, 9.4); ax.axis("off")


def box(x, y, w, h, text, fc, ec, size=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", family="monospace")


def line(x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color="#555", lw=1.1)


ax.text(6.5, 9.05, "LogiEdge MQTT topic tree (local Mosquitto broker on each truck, localhost:1883)",
        ha="center", fontsize=13, fontweight="bold")

box(0.2, 4.15, 1.5, 0.6, "logibridge", "#FFFFFF", "#333", bold=True)
box(2.1, 4.15, 1.3, 0.6, "trucks", "#FFFFFF", "#333", bold=True)
box(3.8, 4.15, 1.6, 0.6, "{truck_id}", "#FFFFFF", "#333", bold=True)
line(1.7, 4.45, 2.1, 4.45); line(3.4, 4.45, 3.8, 4.45)

top, step = 8.0, 0.98
ax.text(6.0, 8.55, "leaf topic", fontsize=9, fontweight="bold")
ax.text(7.75, 8.55, "QoS", fontsize=9, fontweight="bold")
ax.text(8.35, 8.55, "rate", fontsize=9, fontweight="bold")
ax.text(9.9, 8.55, "publisher -> subscribers", fontsize=9, fontweight="bold")
line(5.6, top, 5.6, top - (len(LEAVES) - 1) * step)
line(5.4, 4.45, 5.6, 4.45)
for i, (leaf, q, ret, rate, flow, uplink, (fc, ec)) in enumerate(LEAVES):
    y = top - i * step
    line(5.6, y, 5.9, y)
    box(5.9, y - 0.27, 1.65, 0.54, leaf, fc, ec, size=10, bold=True)
    ax.text(7.9, y, f"{q}{' R' if ret else ''}", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(8.35, y, rate, va="center", fontsize=8.5)
    ax.text(9.9, y, flow + ("   [uplink]" if uplink else ""), va="center", fontsize=8.5,
            color=ec if uplink else "#222")
    if i == 3:
        ax.plot([5.75, 12.9], [y - 0.49, y - 0.49], ls="--", color="#999", lw=0.8)

legend = [("Sensor inputs: raw temperature/vibration never leave the truck; door events are logged for custody", IN_C),
          ("Simulator ground truth: evaluation only, never read by the edge node", EVAL_C),
          ("Edge outputs: reach the operations centre when coverage exists (all but status replayed from the local log)", OUT_C)]
for i, (t, (fc, ec)) in enumerate(legend):
    ax.add_patch(FancyBboxPatch((0.25, -0.35 - i * 0.42), 0.35, 0.3, boxstyle="round,pad=0.02", fc=fc, ec=ec))
    ax.text(0.75, -0.2 - i * 0.42, t, va="center", fontsize=9)
ax.text(0.25, -1.65, "R = retained.  inference, alerts, drift and door events are written to the local log first; the sync "
        "process replays them to the cloud broker\n(QoS 1, persistent session, TLS, alerts first).  status is not logged: "
        "retained + Last Will ('offline'), republished on reconnect.  Example: logibridge/trucks/truck01/inference",
        fontsize=8.5, va="center", color="#333")

fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
