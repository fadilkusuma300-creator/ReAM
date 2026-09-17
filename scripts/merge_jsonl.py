#!/usr/bin/env python3
import argparse
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("inputs", nargs="+"); p.add_argument("--output", required=True); a=p.parse_args()
out=Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w",encoding="utf-8") as w:
    for path in a.inputs:
        with Path(path).open("r",encoding="utf-8") as r:
            for line in r:
                if line.strip(): w.write(line if line.endswith("\n") else line+"\n")
