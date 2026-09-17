#!/usr/bin/env python3
import argparse
from ream.geoer import convert_pair_file, attach_neighbors

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="command", required=True)
a = sub.add_parser("pairs")
a.add_argument("--input", required=True); a.add_argument("--output", required=True); a.add_argument("--city", required=True); a.add_argument("--source-pair", required=True)
b = sub.add_parser("neighbors")
b.add_argument("--pairs", required=True); b.add_argument("--neighbors", required=True); b.add_argument("--output", required=True)
args = p.parse_args()
if args.command == "pairs":
    print(convert_pair_file(args.input, args.output, args.city, args.source_pair))
else:
    attach_neighbors(args.pairs, args.neighbors, args.output)
