# scripts/make_subsets.py
from __future__ import annotations
import argparse
from pathlib import Path
from tools.subsets import labels_from_raw, stratified_indices, save_indices_json

def main():
    ap = argparse.ArgumentParser("Create deterministic CIFAR-10 subsets")
    ap.add_argument("--data-root", default="./.data")
    ap.add_argument("--split", choices=["train","test"], default="train")
    ap.add_argument("--per-class", nargs="+", type=int, required=True, help="e.g. 100 500 1000")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out-dir", default="assets")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    labels = labels_from_raw(args.data_root, args.split)

    for pc in args.per_class:
        indices = stratified_indices(labels, per_class=pc, seed=args.seed)
        out = f"{args.out_dir}/cifar10_{args.split}_{pc}pc_seed{args.seed}.json"
        save_indices_json(indices, out, split=args.split, per_class=pc, seed=args.seed)
        print(f"saved → {out} (n={len(indices)})")

if __name__ == "__main__":
    main()