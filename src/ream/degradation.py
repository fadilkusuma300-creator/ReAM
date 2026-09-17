from __future__ import annotations

import copy
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Callable

from .config import DegradationConfig

TEXT_FIELDS = ("name", "address", "postal_code", "category")
HOUSE_NUMBER = re.compile(r"\d+")


def _record(example: dict[str, Any], side: str) -> dict[str, Any]:
    return example[side]


def _available_text_fields(record: dict[str, Any]) -> list[str]:
    return [f for f in TEXT_FIELDS if record.get(f) not in (None, "", "NULL")]


def _shift_coordinate(lat: float, lon: float, distance_m: float, bearing_rad: float):
    """Great-circle destination point, so the shift is a true surface distance rather than a lat/lon offset."""
    radius = 6_371_000.0
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    delta = distance_m / radius
    phi2 = math.asin(math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(bearing_rad))
    lam2 = lam1 + math.atan2(
        math.sin(bearing_rad) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    return math.degrees(phi2), math.degrees(lam2)


@dataclass
class ControlledDegrader:
    config: DegradationConfig
    donor_lookup: Callable[[dict[str, Any], str, str, random.Random], str | None] | None = None

    def _side(self, rng: random.Random) -> str:
        return "left" if rng.random() < 0.5 else "right"

    def text_mask(self, example: dict[str, Any], rng: random.Random, probability: float | None = None):
        out = copy.deepcopy(example)
        side = self._side(rng)
        rec = _record(out, side)
        fields = _available_text_fields(rec)
        if not fields:
            return out
        p = probability if probability is not None else rng.uniform(self.config.text_mask_min, self.config.text_mask_max)
        masked = [f for f in fields if rng.random() < p]
        if len(masked) == len(fields):
            # Leave one field intact; masking all of them would make the branch unavailable, not degraded.
            masked.remove(rng.choice(masked))
        for f in masked:
            rec[f] = None
        return out

    def coordinate_shift(self, example: dict[str, Any], rng: random.Random, distance_m: float | None = None):
        out = copy.deepcopy(example)
        side = self._side(rng)
        rec = _record(out, side)
        if rec.get("lat") is None or rec.get("lon") is None:
            return out
        d = distance_m if distance_m is not None else rng.uniform(self.config.coordinate_shift_min_m, self.config.coordinate_shift_max_m)
        rec["lat"], rec["lon"] = _shift_coordinate(float(rec["lat"]), float(rec["lon"]), d, rng.uniform(0, 2 * math.pi))
        return out

    def neighborhood_delete(self, example: dict[str, Any], rng: random.Random, probability: float | None = None):
        out = copy.deepcopy(example)
        side = self._side(rng)
        rec = _record(out, side)
        neighbors = list(rec.get("neighbors") or [])
        if not neighbors:
            return out
        if rec.get("spatial_neighbor_count") is None:
            # Freeze the count as it stands: the spatial branch reads this field, so deletion must
            # perturb the neighborhood branch alone.
            rec["spatial_neighbor_count"] = len(neighbors)
        p = probability if probability is not None else rng.uniform(self.config.neighborhood_delete_min, self.config.neighborhood_delete_max)
        kept = [n for n in neighbors if rng.random() >= p]
        if not kept:
            kept = [rng.choice(neighbors)]
        rec["neighbors"] = kept
        return out

    def attribute_conflict(self, example: dict[str, Any], rng: random.Random):
        """Introduces a conflicting postal code or house number on one side.

        Negatives can reuse the opposite side's value; positives need an external donor so the
        conflict does not come from their own true match.
        """
        out = copy.deepcopy(example)
        side = self._side(rng)
        other = "right" if side == "left" else "left"
        rec = _record(out, side)
        if int(out["label"]) == 0:
            if _record(out, other).get("postal_code") not in (None, "", "NULL"):
                rec["postal_code"] = _record(out, other)["postal_code"]
                return out
            other_addr = str(_record(out, other).get("address") or "")
            number = HOUSE_NUMBER.search(other_addr)
            if number:
                addr = str(rec.get("address") or "")
                current = HOUSE_NUMBER.search(addr)
                rec["address"] = HOUSE_NUMBER.sub(number.group(0), addr, count=1) if current else f"{number.group(0)} {addr}".strip()
            return out
        if self.donor_lookup is not None:
            donor = self.donor_lookup(out, side, "postal_code", rng)
            if donor:
                rec["postal_code"] = donor
                return out
            donor = self.donor_lookup(out, side, "house_number", rng)
            if donor:
                addr = str(rec.get("address") or "")
                rec["address"] = HOUSE_NUMBER.sub(donor, addr, count=1) if HOUSE_NUMBER.search(addr) else f"{donor} {addr}".strip()
        return out

    def apply(self, example: dict[str, Any], kind: str, rng: random.Random, intensity: float | None = None):
        if kind == "text_mask":
            return self.text_mask(example, rng, intensity)
        if kind == "coordinate_shift":
            return self.coordinate_shift(example, rng, intensity)
        if kind == "neighborhood_delete":
            return self.neighborhood_delete(example, rng, intensity)
        if kind == "attribute_conflict":
            return self.attribute_conflict(example, rng)
        raise ValueError(f"Unknown degradation kind: {kind}")

    def training_view(self, example: dict[str, Any], rng: random.Random):
        if rng.random() < self.config.clean_fraction:
            return copy.deepcopy(example), []
        kinds = ["text_mask", "coordinate_shift", "neighborhood_delete", "attribute_conflict"]
        if rng.random() < self.config.single_fraction_of_degraded:
            selected = [rng.choice(kinds)]
        else:
            selected = rng.sample(kinds, 2)
        out = copy.deepcopy(example)
        for kind in selected:
            out = self.apply(out, kind, rng)
        return out, selected

class PartitionDonorIndex:
    """Partition-local donor index for positive-sample attribute conflicts."""
    def __init__(self, rows: list[dict[str, Any]]):
        self.records = {"left": [], "right": []}
        for row in rows:
            city = str(row.get("city", ""))
            for side in ("left", "right"):
                rec = row[side]
                self.records[side].append((city, rec))

    @staticmethod
    def _signature(record: dict[str, Any]):
        return (
            str(record.get("name") or ""), str(record.get("address") or ""),
            str(record.get("postal_code") or ""), record.get("lat"), record.get("lon"),
        )

    def lookup(self, example: dict[str, Any], side: str, field: str, rng: random.Random):
        """Draws a donor value from the same city, excluding both sides of the current pair."""
        city = str(example.get("city", ""))
        current = self._signature(example[side])
        opposite = self._signature(example["right" if side == "left" else "left"])
        candidates = []
        for c, rec in self.records[side]:
            sig = self._signature(rec)
            if c != city or sig in (current, opposite):
                continue
            if field == "postal_code":
                value = rec.get("postal_code")
                if value not in (None, "", "NULL"):
                    candidates.append(str(value))
            else:
                match = HOUSE_NUMBER.search(str(rec.get("address") or ""))
                if match:
                    candidates.append(match.group(0))
        return rng.choice(candidates) if candidates else None
