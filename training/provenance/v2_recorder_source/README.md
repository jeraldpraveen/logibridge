# Source snapshot: v2 dataset recorder

Exact source of the code that the v2 recording process (`generate_dataset.py`, record step)
loaded when it was launched on 2026-11-05. The recorder process kept these definitions in
memory for its whole run, even though `generate_dataset.py` and `labeling.py` were edited on
disk afterwards.

| File | SHA-256 | How it was preserved |
|---|---|---|
| `generate_dataset.py` | `9466b83d454909a0…` | Reconstructed by reversing the two later edits (docstring disclosure, build provenance block); verified byte-for-byte against the hash captured while the recorder was running |
| `labeling.py` | `0ff0e739dd807658…` | Reconstructed by reversing the later symmetric-deviation edit; verified byte-for-byte |
| `simulator.py` | `7c5ea695853fd7b6…` | Copied; unchanged on disk since launch |
| `preprocessing.py` | `9af39fde1da6bbc7…` | Copied; unchanged on disk since launch |

Full hashes are in `../v2_recording_provenance.json`. Check with `sha256sum *.py`.

**What differs from the committed versions, and why it does not affect the recordings:**
- `labeling.py` here uses the signed (upward-only) rule. The record step does not label
  anything: it only stores features, window end times and the simulator truth stream.
  Labels are assigned later by `--build`, which uses the current symmetric rule. All
  recorded offsets are >= 0, so both rules give identical labels on this data.
- `generate_dataset.py` here has no `dataset_provenance.json` block in `build()`. The record
  step never calls `build()`.

**Limitation:** the frozen v1 transition recording
(`optimisation/results/transitions_diag_frozen_records.json`) was made before recorder code
hashing was added, so it carries model and stats hashes but no `recording_code_sha256`. Its
later re-analysis carries `analysis_code_sha256`. The recording-code version for that file is
therefore not established.
