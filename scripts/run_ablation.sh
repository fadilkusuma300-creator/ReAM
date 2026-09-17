#!/usr/bin/env bash
set -euo pipefail
TRAIN=${1:?train.jsonl}; VALID=${2:?valid.jsonl}; OUT=${3:-outputs/ablation}
python scripts/train.py --train "$TRAIN" --valid "$VALID" --output "$OUT/full"
python scripts/train.py --train "$TRAIN" --valid "$VALID" --output "$OUT/static" --fusion static
python scripts/train.py --train "$TRAIN" --valid "$VALID" --output "$OUT/attention" --fusion attention
python scripts/train.py --train "$TRAIN" --valid "$VALID" --output "$OUT/confidence" --fusion confidence
python scripts/train.py --train "$TRAIN" --valid "$VALID" --output "$OUT/no_rel_supervision" --without-reliability-supervision
python scripts/train.py --train "$TRAIN" --valid "$VALID" --output "$OUT/no_quality" --without-quality
python scripts/train.py --train "$TRAIN" --valid "$VALID" --output "$OUT/no_degradation" --without-degradation
