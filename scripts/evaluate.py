#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ream.config import load_config
from ream.data import read_jsonl, PairDataset, ReAMCollator
from ream.evaluation import save_evaluation
from ream.model import ReAM
from ream.statistics import load_statistics, build_collator_statistics
from ream.training import infer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--fusion", choices=["reliability", "attention", "confidence", "static"], default="reliability")
    ap.add_argument("--without-quality", action="store_true")
    ap.add_argument("--statistics", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_model, use_fast=True)
    stats_path = Path(args.statistics) if args.statistics else Path(args.checkpoint).with_name("training_statistics.json")
    stat_args = build_collator_statistics(load_statistics(stats_path))
    collate = ReAMCollator(tokenizer, cfg.model.max_length, cfg.model.max_neighbors, cfg.model.max_neighbor_name_chars, **stat_args)
    loader = DataLoader(PairDataset(read_jsonl(args.data)), batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, collate_fn=collate)
    model = ReAM(cfg.model, fusion=args.fusion, use_quality=not args.without_quality)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    rows = infer(model, loader, device)
    result = save_evaluation(rows, float(ckpt.get("threshold", 0.5)), args.output, cfg.metrics.ece_bins, cfg.metrics.reliability_bins)
    print(json.dumps(result["metrics"], indent=2))

if __name__ == "__main__": main()
