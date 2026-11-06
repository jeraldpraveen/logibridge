# Task E2 — Ansible deployment playbook (and E3 OTA strategy)

| File | Purpose |
|---|---|
| `logibridge_deploy.yml` | Edge-node deployment: **exactly the 7 tasks** of the brief |
| `inventory.ini` | Inventory: the edge node is this machine (`localhost`, local connection) |
| `results/ansible_run1.txt`, `ansible_run2.txt` | Idempotency evidence: run 1 on a fresh node, run 2 immediately after |
| `results/ansible_run3_update_m3.txt`, `ansible_run4_rollback_m2.txt` | Extra: a real model update and its rollback through the same playbook |
| `ota_strategy.md` | Task E3: OTA strategy comparison |

All runs were on the **personal PC** (`../MACHINES.md`): WSL Ubuntu 24.04, ansible-core 2.16.3,
`community.docker` 3.7.0, Docker SDK for Python 5.0.3, Docker Desktop 29.9.2.

## Run it

In WSL Ubuntu, from the repository root (`/mnt/j/ASSIGNMENT/MLONEDGE/logibridge`):

```
sudo ansible-playbook -i deployment/inventory.ini deployment/logibridge_deploy.yml
```

`sudo` is needed for `/opt/logibridge`; the saved runs used `wsl -u root`, which is equivalent.
The image comes from the local registry `localhost:5000` (`registry:2`, bound to `127.0.0.1`
only), pushed by `inference/ota_layer_demo.py`.

## The 7 tasks

| # | Task (brief) | Module | Idempotent because |
|---|---|---|---|
| 1 | Create `/opt/logibridge` if not present | `file`, `state: directory` | changes only if the directory is missing |
| 2 | Copy new model file to `/opt/logibridge/model.tflite` | `copy` | compares checksums; copies only a different file |
| 3 | Copy updated `reference_dist.json` | `copy` | same |
| 4 | Stop the running inference container if running | `shell` + `when` | runs **only if task 2 or 3 changed a file**, because the container reads both from a bind mount and must restart to load them. It checks that a container is running before stopping it: a fresh truck has none, and `docker_container state=stopped` would otherwise try to create one. It reports `changed` only when it stopped something |
| 5 | Pull updated container image from local Docker registry | `docker_image`, `source: pull`, `force_source: true` | always asks the registry, reports `changed` only if the image differs |
| 6 | Start inference container with environment variables | `docker_container`, `state: started` | compares image, environment and mounts with the running container; starts or recreates only on a difference |
| 7 | Wait 15 seconds and verify the container is running | `shell`: `sleep 15 && docker inspect` | read-only (`changed_when: false`); **fails** unless the container is running with 0 restarts, which catches a crash loop |

Environment set by task 6: `MODEL_PATH=/opt/logibridge/model.tflite` (the variant is chosen
without rebuilding the image), `TRUCK_ID`, `MQTT_HOST`, `MQTT_PORT`, `UPLINK_HOST`, `LOG_DB`.
Mounts: `/opt/logibridge` (read-only) and the `logibridge-data` volume for the durable log.
Restart policy: `unless-stopped`.

## Results

| Run | Starting state | Recap | Task outcomes |
|---|---|---|---|
| 1 | Fresh node: no `/opt/logibridge`, no container | `ok=7 changed=4 failed=0` | 1, 2, 3, 6 changed; 4 ran (nothing was running); 5, 7 ok |
| **2** | Immediately after run 1, nothing changed | **`ok=6 changed=0 skipped=1 failed=0`** | 4 skipped (no file changed); every other task ok |
| 3 (extra) | `-e model_src=…/m3_pruned_ptq_int8.tflite` | `ok=7 changed=3` | 2 changed → **4 stopped the container** → 6 restarted it; container log shows model `f220ec06…` (M3) |
| 4 (extra) | Back to the committed M2 | `ok=7 changed=3` | same path; container log shows model `c331664a…` (M2) |

After the runs the node runs `localhost:5000/logibridge-inference:m2` with `/opt/logibridge/model.tflite`
= M2 (`c331664a…`), connected to the local broker. Runs 3 and 4 show that the conditional stop
in task 4, which is what makes run 2 report `changed=0`, still restarts the container when a
model really changes.

**A problem found on the way:** the first version of task 4 used
`docker_container state=stopped`. On the fresh node it failed with *"Cannot create container
when image is not specified"*, because that module creates a container if none exists. Task 4
now checks that a container is running before stopping it, which is also what the brief's
wording ("if running") asks for.
