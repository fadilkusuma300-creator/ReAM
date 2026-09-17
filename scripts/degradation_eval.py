#!/usr/bin/env python3
from __future__ import annotations

import argparse, random
from pathlib import Path

from ream.config import load_config
from ream.data import read_jsonl, write_jsonl
from ream.degradation import ControlledDegrader, PartitionDonorIndex

# Intensity grid per degradation kind. For text_mask, neighborhood_delete and attribute_conflict the
# value is a probability or ratio; for coordinate_shift it is a distance in metres.
LEVELS = {
    "text_mask": [0.0, 0.10, 0.20, 0.30, 0.40],
    "coordinate_shift": [0.0, 20.0, 50.0, 100.0, 200.0],
    "neighborhood_delete": [0.0, 0.20, 0.40, 0.50, 0.60],
    "attribute_conflict": [0.0, 0.10, 0.20, 0.30, 0.40],
}

p=argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--kind", choices=LEVELS, required=True); p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--seed", type=int, default=13)
a=p.parse_args(); cfg=load_config(a.config); rows=read_jsonl(a.data); out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True); donors=PartitionDonorIndex(rows); d=ControlledDegrader(cfg.degradation, donors.lookup)
for level in LEVELS[a.kind]:
    changed=[]
    for i,row in enumerate(rows):
        rng=random.Random(a.seed*1_000_003+i)
        if level == 0.0:
            x=row
        elif a.kind == "attribute_conflict":
            x=d.apply(row, a.kind, rng) if rng.random() < level else row
        else:
            x=d.apply(row, a.kind, rng, level)
        changed.append(x)
    write_jsonl(out / f"{a.kind}_{level:g}.jsonl", changed)
