#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ream.config import load_config
from ream.data import read_jsonl, PairDataset, DegradedPairDataset, ReAMCollator
from ream.degradation import ControlledDegrader, PartitionDonorIndex
from ream.model import ReAM
from ream.training import seed_everything, train_model
from ream.statistics import fit_statistics, save_statistics, build_collator_statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--fusion", choices=["reliability", "attention", "confidence", "static"], default="reliability")
    ap.add_argument("--without-quality", action="store_true")
    ap.add_argument("--without-reliability-supervision", action="store_true")
    ap.add_argument("--without-degradation", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config); seed_everything(cfg.seed)
    train_rows, valid_rows = read_jsonl(args.train), read_jsonl(args.valid)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_model, use_fast=True)
    # Statistics are fitted on the training rows alone and saved next to the checkpoint, so evaluation
    # can reuse them instead of refitting on the evaluation split.
    stats = fit_statistics(train_rows, tokenizer, cfg.model.max_length)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    save_statistics(stats, Path(args.output) / "training_statistics.json")
    stat_args = build_collator_statistics(stats)
    collate = ReAMCollator(tokenizer, cfg.model.max_length, cfg.model.max_neighbors, cfg.model.max_neighbor_name_chars, **stat_args)
    if args.without_degradation:
        train_ds = PairDataset(train_rows)
    else:
        donor_index = PartitionDonorIndex(train_rows)
        train_ds = DegradedPairDataset(train_rows, ControlledDegrader(cfg.degradation, donor_index.lookup), cfg.seed)
    clean_train_ds = PairDataset(train_rows)
    valid_ds = PairDataset(valid_rows)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, collate_fn=collate)
    refresh_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, collate_fn=collate)
    model = ReAM(cfg.model, fusion=args.fusion, use_quality=not args.without_quality)
    train_model(model, train_loader, refresh_loader, valid_loader, cfg, args.output, reliability_supervision=not args.without_reliability_supervision)


if __name__ == "__main__":
    main()
