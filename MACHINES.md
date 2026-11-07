# Machines used

Timing and energy results depend on hardware. Every timing figure in the reports names the
machine below that produced it.

| | Office PC | Personal PC |
|---|---|---|
| Used for | Tasks A1–F3: dataset, training, conversion, benchmark (`optimisation/results/`) | Task D2 onward: Docker, drift monitoring, Ansible, demo, reports |
| Model / CPU | Dell Latitude 3420, Intel Core i7-1165G7 (11th Gen, 4 cores / 8 threads, 2.8 GHz) | Samsung 750XHD, Intel Core Ultra 5 225U (12 cores / 14 threads) |
| RAM | 32 GB | 15.5 GB |
| OS | Windows 11 Enterprise 23H2 (build 22631) | Windows 11 Home 10.0.26200; WSL 2 Ubuntu 24.04.5 LTS |
| Python / TensorFlow | 3.11.4 / 2.21.0 | 3.11.9 / 2.21.0 (same pinned `requirements.txt`) |
| Services | Mosquitto | Mosquitto (listener 1883, anonymous — local demo only), Docker 29.9.2, Ansible core 2.16.3 in WSL |
| TDP estimate used for energy | 28 W (the i7-1165G7's configurable TDP is 12–28 W; 28 W is its upper value) | - |

