#!/usr/bin/env python3
from __future__ import annotations
import argparse, random
from pathlib import Path
from ream.config import load_config
from ream.data import read_jsonl, write_jsonl
from ream.degradation import ControlledDegrader

p=argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--output", required=True); p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--seed", type=int, default=13); p.add_argument("--text-mask", type=float, default=0.30); p.add_argument("--coordinate-shift-m", type=float, default=100.0)
a=p.parse_args(); cfg=load_config(a.config); d=ControlledDegrader(cfg.degradation); rows=read_jsonl(a.data); changed=[]
for i,row in enumerate(rows):
    rng=random.Random(a.seed*1_000_003+i)
    x=d.apply(row,"text_mask",rng,a.text_mask)
    x=d.apply(x,"coordinate_shift",rng,a.coordinate_shift_m)
    changed.append(x)
write_jsonl(Path(a.output), changed)
