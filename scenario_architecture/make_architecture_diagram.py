"""Draws system_architecture.png (Task A2)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(15, 8.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis("off")


def box(x, y, w, h, text, fc="#E8F1F8", ec="#1F5F8B", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2, label="", style="-|>", color="#333", ls="-", dy=0.12):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.4, ls=ls))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center",
                fontsize=7.5, color=color)


# truck boundary
ax.add_patch(FancyBboxPatch((0.2, 0.4), 9.6, 7.6, boxstyle="round,pad=0.05",
                            fc="#FAFAFA", ec="#555", lw=1.8, ls="--"))
ax.text(0.45, 7.7, "REFRIGERATED TRUCK  -  Edge node (Raspberry Pi 5, 12 V via DC-DC, <10 W)",
        fontsize=10, fontweight="bold", color="#333")

# sensors
box(0.5, 5.8, 2.1, 0.9, "Temperature probe\n1 Hz", fc="#FFF4E0", ec="#C77700")
box(0.5, 4.5, 2.1, 0.9, "Compressor vibration\n(accelerometer, RMS 0.5 Hz)", fc="#FFF4E0", ec="#C77700")
box(0.5, 3.2, 2.1, 0.9, "Door reed switch\nOPEN / CLOSE events", fc="#FFF4E0", ec="#C77700")
ax.text(1.55, 6.85, "SENSORS", ha="center", fontsize=8, color="#C77700", fontweight="bold")

# broker
box(3.4, 3.9, 1.9, 2.0, "Local MQTT\nbroker\n(Mosquitto\nlocalhost:1883)", fc="#E9F7EC", ec="#2E7D32", bold=True)
for y in (6.25, 4.95, 3.65):
    arrow(2.6, y, 3.4, 4.9 + (y - 4.95) * 0.35, "")
ax.text(4.35, 6.1, "temp QoS1 | vib QoS0 | door QoS2", fontsize=7, ha="center")

# pipeline
box(6.0, 5.3, 3.5, 1.4, "Preprocessing\n5-sample moving average\n30 s window, 10 s step\n6 features -> z-score (training_stats.npy)")
box(6.0, 3.5, 3.5, 1.2, "Inference (TFLite INT8 MLP 6-32-16-3)\nclass: Normal / Warning / Critical", bold=True, fc="#DCEBFA")
arrow(5.3, 5.3, 6.0, 6.0, "sensor msgs")
arrow(7.75, 5.3, 7.75, 4.7, "feature vector")
arrow(5.3, 4.5, 6.0, 4.1, "", color="#2E7D32")  # result path back
ax.text(5.45, 4.0, "", fontsize=7)

# outputs
box(6.0, 1.9, 1.7, 1.0, "Local alert\nlog (append-only,\nstore-and-forward)", fc="#FDE9E7", ec="#B3261E")
box(7.9, 1.9, 1.7, 1.0, "Cab alert\n(buzzer / display)\nworks offline", fc="#FDE9E7", ec="#B3261E")
box(0.5, 1.9, 2.3, 1.0, "PSI drift monitor\n(rolling 100, every 60 s)", fc="#F3E8FA", ec="#6A1B9A")
arrow(7.2, 3.5, 6.85, 2.9, "every result")
arrow(8.3, 3.5, 8.75, 2.9, "Warning / Critical")
arrow(6.0, 4.0, 2.8, 2.6, "confidence", color="#6A1B9A")
box(3.4, 0.7, 2.2, 0.8, "OTA: Docker image\n+ model.tflite (Ansible)", fc="#EEE", ec="#666", fs=8)

# uplink
box(10.4, 3.4, 1.6, 1.4, "Cellular uplink\n4G M2M SIM\nIntermittent\n(35-90 min gaps)", fc="#FFFBE0", ec="#B58900", bold=True)
ax.plot([6.85, 6.85, 11.2], [1.9, 1.15, 1.15], ls="--", color="#333", lw=1.4)
arrow(11.2, 1.15, 11.2, 3.4, "", ls="--")
ax.text(9.0, 1.25, "sync unsent log entries when coverage returns (MQTT QoS1, TLS)", ha="center", fontsize=7.5)

# cloud / ops centre
ax.add_patch(FancyBboxPatch((12.5, 0.4), 2.3, 7.6, boxstyle="round,pad=0.05",
                            fc="#F4F8FB", ec="#1F5F8B", lw=1.8))
ax.text(13.65, 7.7, "OPERATIONS CENTRE", ha="center", fontsize=10, fontweight="bold", color="#1F5F8B")
box(12.7, 5.8, 1.9, 1.2, "Cloud MQTT\nbroker (TLS,\nper-truck creds)")
box(12.7, 4.1, 1.9, 1.3, "Alert + chain-of-\ncustody store\n(ordered replay)")
box(12.7, 2.4, 1.9, 1.3, "Dashboard /\nescalation to\noperator")
box(12.7, 0.8, 1.9, 1.1, "Model registry /\nOTA server", fc="#EEE", ec="#666")
arrow(12.0, 4.1, 12.7, 6.4, "")
arrow(13.65, 5.8, 13.65, 5.4, "")
arrow(13.65, 4.1, 13.65, 3.7, "")
arrow(12.7, 1.4, 12.1, 3.4, "", ls="--", color="#666")
ax.text(12.25, 2.2, "OTA", fontsize=7.5, color="#666", ha="center")

ax.text(7.5, 0.1, "Solid = on-truck, always works offline.  Dashed = needs cellular coverage (buffered otherwise).",
        ha="center", fontsize=8.5, style="italic")
plt.tight_layout()
plt.savefig("system_architecture.png", dpi=170)
print("saved")
