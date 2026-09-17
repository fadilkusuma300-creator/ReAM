from __future__ import annotations

import json
import re
from pathlib import Path

from .data import write_jsonl

FIELD = re.compile(r"COL\s+([^\s]+)\s+VAL\s+(.*?)(?=\s+COL\s+[^\s]+\s+VAL\s+|$)")


def parse_serialized_entity(text: str):
    raw = {k.lower(): v.strip() for k, v in FIELD.findall(text)}
    def val(*names):
        for name in names:
            x = raw.get(name.lower())
            if x is not None and x.upper() != "NULL": return x
        return None
    def number(x):
        try: return float(x) if x is not None else None
        except ValueError: return None
    return {
        "name": val("name"),
        "address": val("address"),
        "postal_code": val("postalCode", "postal_code", "zip"),
        "category": val("category", "categories"),
        "lat": number(val("latitude", "lat")),
        "lon": number(val("longitude", "lon", "lng")),
        "neighbors": [],
    }


def convert_pair_file(pair_path: str | Path, output_path: str | Path, city: str, source_pair: str):
    rows = []
    with Path(pair_path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip(): continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"Expected <entity1>\\t<entity2>\\t<label> at line {idx+1}")
            left, right, label = parts[0], parts[1], parts[-1]
            rows.append({"id": f"{city}-{source_pair}-{idx}", "city": city, "source_pair": source_pair, "label": int(label), "left": parse_serialized_entity(left), "right": parse_serialized_entity(right)})
    write_jsonl(output_path, rows)
    return len(rows)


def attach_neighbors(jsonl_path: str | Path, neighbors_json: str | Path, output_path: str | Path):
    rows = [json.loads(x) for x in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if x.strip()]
    neighbors = json.loads(Path(neighbors_json).read_text(encoding="utf-8"))
    if len(rows) != len(neighbors):
        raise ValueError("Pair and neighbor files must contain the same number of examples in the same order.")
    for row, n in zip(rows, neighbors):
        row["left"]["neighbors"] = [{"name": name, "category": "", "distance_m": dist} for name, dist in zip(n.get("neigh1", []), n.get("dist1", []))]
        row["right"]["neighbors"] = [{"name": name, "category": "", "distance_m": dist} for name, dist in zip(n.get("neigh2", []), n.get("dist2", []))]
    write_jsonl(output_path, rows)
