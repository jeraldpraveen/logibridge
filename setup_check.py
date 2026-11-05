"""Run once on a new machine after cloning:  python setup_check.py

1. Restores training/raw/ (git-ignored) from the tracked copies, verifying every file
   against its committed SHA-256, so evaluation scripts can find the recordings.
2. Verifies the committed models, statistics and dataset hashes.
3. Checks the Python packages, Mosquitto, Docker and Ansible-in-WSL.
Read-only apart from creating training/raw/.
"""
import hashlib
import importlib
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "training" / "raw"
H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ok = True


def say(flag, msg):
    global ok
    ok &= flag
    print(f"  [{'OK' if flag else 'FAIL'}] {msg}")


print("1. Restore training/raw/ from tracked copies")
RAW.mkdir(parents=True, exist_ok=True)
prov = json.load(open(ROOT / "training" / "dataset_provenance.json"))["recordings_sha256"]
for name, want in prov.items():
    src = ROOT / "training" / "provenance" / "v2_recordings" / name
    if H(src) != want:
        say(False, f"{name}: tracked copy hash mismatch")
        continue
    if not (RAW / name).exists():
        shutil.copy2(src, RAW / name)
    say(H(RAW / name) == want, f"{name}")
shutil.copy2(ROOT / "training" / "provenance" / "v2_recording_provenance.json", RAW)
for line in (ROOT / "training" / "models" / "diag_frozen" / "SHA256SUMS").read_text().splitlines():
    want, name = line.split(" *")
    if name.startswith("dataset/class"):
        src = ROOT / "training" / "models" / "diag_frozen" / name
        if not (RAW / Path(name).name).exists():
            shutil.copy2(src, RAW / Path(name).name)
        say(H(RAW / Path(name).name) == want, f"v1 {Path(name).name}")

print("2. Committed artefacts")
expect = {"training/models/m1_fp32.tflite": "986ebc19", "training/models/m2_ptq_int8.tflite": "c331664a",
          "training/models/m3_pruned_ptq_int8.tflite": "f220ec06",
          "data_pipeline/training_stats.npy": "20f01b2d", "training/dataset.npz": "d2ec8995"}
for f, h in expect.items():
    say(H(ROOT / f).startswith(h), f"{f} sha256 {h}...")

print("3. Python packages (requirements.txt)")
for line in (ROOT / "requirements.txt").read_text().splitlines():
    if "==" not in line or line.startswith("#"):
        continue
    pkg, ver = line.split("==")
    try:
        from importlib.metadata import version
        got = version(pkg)
        say(got == ver, f"{pkg} {got}" + ("" if got == ver else f" (expected {ver})"))
    except Exception:
        say(False, f"{pkg} not installed")

print("4. Services")
s = socket.socket(); s.settimeout(1)
say(s.connect_ex(("127.0.0.1", 1883)) == 0, "Mosquitto listening on localhost:1883"); s.close()
for name, cmd in (("Docker", ["docker", "--version"]),
                  ("Ansible in WSL", ["wsl", "-e", "ansible", "--version"])):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        say(out.returncode == 0, f"{name}: {(out.stdout or out.stderr).strip().splitlines()[0][:60]}")
    except Exception as e:
        say(False, f"{name}: {type(e).__name__}")

print("\nALL OK" if ok else "\nSome checks failed - see above")
sys.exit(0 if ok else 1)
