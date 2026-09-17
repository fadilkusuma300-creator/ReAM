#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

CITIES = ["singapore", "edinburgh", "toronto", "pittsburgh"]

p=argparse.ArgumentParser(); p.add_argument("--data-root", required=True); p.add_argument("--output", required=True); a=p.parse_args(); root=Path(a.data_root)
manifest={}
# One fold per city: the held-out city supplies the test split, the other three supply train and validation.
for held_out in CITIES:
    train=[]; valid=[]; test=[]
    for city in CITIES:
        for source in ["osm_fsq","osm_yelp"]:
            base=root/source/city
            if city == held_out:
                test.append(str(base/"test.jsonl"))
            else:
                train.append(str(base/"train.jsonl")); valid.append(str(base/"valid.jsonl"))
    manifest[held_out]={"train":train,"valid":valid,"test":test}
Path(a.output).write_text(json.dumps(manifest,indent=2),encoding="utf-8")
