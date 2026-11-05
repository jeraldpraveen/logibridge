"""Task D2 OTA layer-cache demonstration.

1. Build the image with M2 (inference/model.tflite) -> logibridge-inference:m2
2. Replace ONLY inference/model.tflite with M3, rebuild    -> logibridge-inference:m3
   and show that every step is CACHED except `COPY model.tflite .`
3. Restore M2 (inference/model.tflite always ends as the committed M2 file)
4. Push both tags to the local registry (localhost:5000; the same registry Ansible pulls from)
   and read the manifests: the compressed layer sizes are what a truck actually downloads.
5. Bandwidth for 85 trucks: full image vs model-layer-only update, at Rs 0.10/MB.

Run from the repo root:  .venv\\Scripts\\python inference\\ota_layer_demo.py
Writes inference/results/ota_layer_cache.json and ota_build_m3.txt (raw BuildKit log).
"""
import hashlib
import json
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INF, OUT = ROOT / "inference", ROOT / "inference" / "results"
M2 = ROOT / "training" / "models" / "m2_ptq_int8.tflite"
M3 = ROOT / "training" / "models" / "m3_pruned_ptq_int8.tflite"
REG, REPO = "localhost:5000", "logibridge-inference"
TRUCKS, RS_PER_MB, CYCLES_PER_YEAR = 85, 0.10, 52 / 6
MB = 1_000_000
H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(*cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if p.returncode:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout + p.stderr


def build(tag):
    log = run("docker", "build", "--progress=plain", "--build-context", "pipeline=data_pipeline",
              "-t", f"{REPO}:{tag}", "inference")
    steps, cached = {}, set()
    for line in log.splitlines():
        m = re.match(r"#(\d+) \[(?:stage-0 )?(\d+)/(\d+)\] (.+)", line)
        if m:
            steps[m.group(1)] = f"[{m.group(2)}/{m.group(3)}] {m.group(4)}"
        m = re.match(r"#(\d+) CACHED", line)
        if m:
            cached.add(m.group(1))
    # FROM is only resolved (base image pulled once, never rebuilt): BuildKit prints no CACHED line for it
    table = [(desc, "BASE" if "] FROM " in desc else "CACHED" if sid in cached else "REBUILT") for sid, desc in
             sorted(steps.items(), key=lambda kv: int(kv[1].split("/")[0][1:]))]
    return table, log


def manifest(tag):
    acc = ", ".join(["application/vnd.oci.image.index.v1+json", "application/vnd.oci.image.manifest.v1+json",
                     "application/vnd.docker.distribution.manifest.list.v2+json",
                     "application/vnd.docker.distribution.manifest.v2+json"])
    get = lambda ref: json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://{REG}/v2/{REPO}/manifests/{ref}", headers={"Accept": acc})))
    m = get(tag)
    if "manifests" in m:                                   # index: pick the linux/amd64 image
        m = get(next(x["digest"] for x in m["manifests"]
                     if x.get("platform", {}).get("architecture") == "amd64"))
    return {"config": m["config"], "layers": m["layers"]}


def main():
    assert H(INF / "model.tflite") == H(M2), "inference/model.tflite must be the committed M2 before the demo"
    OUT.mkdir(parents=True, exist_ok=True)
    # forget only the cached `COPY model.tflite` results from earlier runs, so the demo shows the
    # rebuild a genuinely new model causes; the pip/code layers stay cached
    # start from the fleet's real state: M2 deployed, M3 never built. Remove earlier :m3 tags so their
    # layer is no longer referenced, then drop unreferenced `COPY model.tflite` cache records
    for ref in (f"{REPO}:m3", f"{REG}/{REPO}:m3"):
        subprocess.run(["docker", "rmi", ref], capture_output=True)
    run("docker", "buildx", "prune", "-f", "--all", "--filter", "description~=model.tflite")
    print("== build 1: model.tflite = M2")
    t2, _ = build("m2")
    for d, s in t2:
        print(f"  {s:<8} {d}")
    try:
        shutil.copyfile(M3, INF / "model.tflite")
        print("\n== build 2: ONLY model.tflite changed (M3)")
        t3, raw = build("m3")
        for d, s in t3:
            print(f"  {s:<8} {d}")
        (OUT / "ota_build_m3.txt").write_text(raw, encoding="utf-8")
    finally:
        shutil.copyfile(M2, INF / "model.tflite")           # always restore the committed model
    assert H(INF / "model.tflite") == H(M2)

    if "registry" not in run("docker", "ps", "-a", "--format", "{{.Names}}").split():
        run("docker", "run", "-d", "-p", "127.0.0.1:5000:5000", "--restart", "unless-stopped", "--name", "registry", "registry:2")
    else:
        run("docker", "start", "registry")
    pushes = {}
    for tag in ("m2", "m3"):
        run("docker", "tag", f"{REPO}:{tag}", f"{REG}/{REPO}:{tag}")
        pushes[tag] = run("docker", "push", f"{REG}/{REPO}:{tag}")
    print("\n== push of m3 after m2 (what the registry already had):")
    for line in pushes["m3"].splitlines():
        if re.search(r"already exists|Pushed|Layer|digest", line, re.I):
            print("  " + line.strip())

    a, b = manifest("m2"), manifest("m3")
    have = {l["digest"] for l in a["layers"]}
    new = [l for l in b["layers"] if l["digest"] not in have]
    full = sum(l["size"] for l in b["layers"]) + b["config"]["size"]
    delta = sum(l["size"] for l in new) + b["config"]["size"]
    rebuilt = [d for d, s in t3 if s == "REBUILT"]
    res = {
        "build_m3_steps": [{"step": d, "status": s} for d, s in t3],
        "only_model_layer_rebuilt": rebuilt == [d for d in rebuilt if "model.tflite" in d] and len(rebuilt) == 1,
        "image_layers": len(b["layers"]),
        "layers_changed_m2_to_m3": len(new),
        "full_image_download_bytes_compressed": full,
        "model_update_download_bytes_compressed": delta,
        "note": "compressed sizes from the registry manifest (layers + image config); a truck that already "
                "has the m2 image pulls only the changed layer(s) and the new config",
        "fleet": {"trucks": TRUCKS, "rs_per_MB": RS_PER_MB, "MB_definition": "10^6 bytes"},
    }
    full_mb, delta_mb = full / MB, delta / MB
    res["per_update_cycle"] = {
        "full_image_MB_fleet": round(full_mb * TRUCKS, 2), "full_image_cost_rs": round(full_mb * TRUCKS * RS_PER_MB, 2),
        "layer_only_MB_fleet": round(delta_mb * TRUCKS, 4), "layer_only_cost_rs": round(delta_mb * TRUCKS * RS_PER_MB, 4),
        "saving_MB_fleet": round((full_mb - delta_mb) * TRUCKS, 2),
        "saving_rs": round((full_mb - delta_mb) * TRUCKS * RS_PER_MB, 2),
        "saving_pct": round(100 * (1 - delta / full), 3)}
    res["per_year_at_6_weekly_updates"] = {
        "cycles": round(CYCLES_PER_YEAR, 2),
        "saving_MB_fleet": round((full_mb - delta_mb) * TRUCKS * CYCLES_PER_YEAR, 1),
        "saving_rs": round((full_mb - delta_mb) * TRUCKS * RS_PER_MB * CYCLES_PER_YEAR, 2)}
    json.dump(res, open(OUT / "ota_layer_cache.json", "w"), indent=2)
    c = res["per_update_cycle"]
    print(f"\nfull image {full_mb:.2f} MB vs model update {delta / 1000:.1f} KB per truck (compressed)")
    print(f"85 trucks per update: {c['full_image_MB_fleet']} MB (Rs {c['full_image_cost_rs']}) vs "
          f"{c['layer_only_MB_fleet']} MB (Rs {c['layer_only_cost_rs']}) -> saving {c['saving_MB_fleet']} MB, "
          f"Rs {c['saving_rs']} ({c['saving_pct']} %)")
    print(f"only the model layer rebuilt: {res['only_model_layer_rebuilt']}")


if __name__ == "__main__":
    main()
