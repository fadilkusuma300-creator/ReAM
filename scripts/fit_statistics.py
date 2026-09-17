#!/usr/bin/env python3
import argparse
from transformers import AutoTokenizer
from ream.config import load_config
from ream.data import read_jsonl
from ream.statistics import fit_statistics, save_statistics
p=argparse.ArgumentParser(); p.add_argument("--train", required=True); p.add_argument("--output", required=True); p.add_argument("--config", default="configs/default.yaml"); a=p.parse_args()
cfg=load_config(a.config); tok=AutoTokenizer.from_pretrained(cfg.model.text_model,use_fast=True); stats=fit_statistics(read_jsonl(a.train),tok,cfg.model.max_length); save_statistics(stats,a.output)
