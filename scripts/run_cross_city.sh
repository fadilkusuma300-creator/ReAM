#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?prepared data root}; OUT=${2:-outputs/cross_city}
mkdir -p "$OUT"
python scripts/cross_city_manifest.py --data-root "$ROOT" --output "$OUT/manifest.json"
python - "$OUT/manifest.json" "$OUT" <<'PY'
import json, subprocess, sys
from pathlib import Path
manifest=json.load(open(sys.argv[1])); out=Path(sys.argv[2])
for city, split in manifest.items():
    fold=out/city; fold.mkdir(parents=True,exist_ok=True)
    for name in ("train","valid","test"):
        cmd=[sys.executable,"scripts/merge_jsonl.py",*split[name],"--output",str(fold/f"{name}.jsonl")]
        subprocess.check_call(cmd)
    subprocess.check_call([sys.executable,"scripts/train.py","--train",str(fold/"train.jsonl"),"--valid",str(fold/"valid.jsonl"),"--output",str(fold/"model")])
    subprocess.check_call([sys.executable,"scripts/evaluate.py","--data",str(fold/"test.jsonl"),"--checkpoint",str(fold/"model/best.pt"),"--output",str(fold/"test_metrics.json")])
PY
