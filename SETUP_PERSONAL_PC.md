# Moving LogiEdge to a new PC

Everything needed is in the GitHub repository. The local `.venv/` and `training/raw/` are not
committed; `setup_check.py` rebuilds `training/raw/` from tracked copies and verifies every hash.

## 1. Install (Windows, admin rights)

| Tool | Command / source | Notes |
|---|---|---|
| Git | `winget install Git.Git` | |
| Python **3.11** | `winget install Python.Python.3.11` | Same minor version as the models were built with |
| Mosquitto | `winget install EclipseFoundation.Mosquitto` | Installs as a Windows service |
| Docker Desktop | `winget install Docker.DockerDesktop` | Needs virtualisation enabled (Task Manager → Performance → CPU) and WSL 2. Reboot when asked |
| WSL Ubuntu | `wsl --install -d Ubuntu-24.04` | Then, inside Ubuntu: `sudo apt update && sudo apt install -y ansible` |

After Docker Desktop starts: **Settings → Resources → WSL integration → enable Ubuntu-24.04**, so
`docker` works inside WSL for Ansible.

## 2. Let containers reach Mosquitto

By default Mosquitto 2.x listens only on localhost, which a container cannot reach. Edit
`C:\Program Files\mosquitto\mosquitto.conf` as admin and add at the end:

```
listener 1883
allow_anonymous true
```

Then restart it from an admin prompt: `net stop mosquitto && net start mosquitto`.
(Anonymous access is acceptable for this local demo only; a truck deployment uses TLS and
per-truck credentials, as described in A1.)

## 3. Clone, install packages, verify

```
git clone https://github.com/jeraldpraveen/logibridge.git
cd logibridge
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python setup_check.py
```

`setup_check.py` should end with **ALL OK**. It confirms the model hashes (`986ebc19…`,
`c331664a…`, `f220ec06…`), the statistics (`20f01b2d…`) and the dataset (`d2ec8995…`), so the new
PC evaluates exactly the same files.

## 4. Git identity on the new PC

```
git config user.name  "Your Name"
git config user.email "your-email"
```

## Notes

- Do **not** retrain on the new PC. The committed models are the ones every result refers to.
- Benchmark latencies will differ on different hardware. If the benchmark is re-run there, the
  report must state which machine produced which numbers.
