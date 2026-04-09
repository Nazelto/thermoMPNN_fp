from __future__ import annotations

import argparse
import json

from .config import load_project_config
from .inference import parse_mutation_string, predict_from_pdb, run_site_saturation_scan
from .preprocessing import curate_fireprot_csv, curate_megascale_csv
from .splits import random_protein_split, write_split_manifest
from .training import train_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FP-style ThermoMPNN reproduction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Curate raw CSVs and optionally write splits")
    prepare.add_argument("--config", default="configs/train_default.yaml")
    prepare.add_argument("--paths", default="configs/local_paths.yaml")
    prepare.add_argument("--skip-splits", action="store_true")

    train = subparsers.add_parser("train", help="Train ThermoMPNN")
    train.add_argument("--config", default="configs/train_default.yaml")
    train.add_argument("--paths", default="configs/local_paths.yaml")

    predict = subparsers.add_parser("predict", help="Predict ddG from a PDB")
    predict.add_argument("--config", default="configs/train_default.yaml")
    predict.add_argument("--paths", default="configs/local_paths.yaml")
    predict.add_argument("--pdb", required=True)
    predict.add_argument("--chain")
    predict.add_argument("--mutation", action="append", default=[])
    predict.add_argument("--positions", nargs="*", type=int, default=None)
    predict.add_argument("--scan", action="store_true")

    return parser


def _run_prepare_data(config_path: str, paths_path: str, skip_splits: bool) -> None:
    config = load_project_config(config_path, paths_path)
    curated_megascale = curate_megascale_csv(
        config.local_paths.megascale_raw_csv,
        config.local_paths.megascale_curated_csv,
    )
    curated_fireprot = curate_fireprot_csv(
        config.local_paths.fireprot_raw_csv,
        config.local_paths.fireprot_curated_csv,
        structure_root=config.local_paths.fireprot_structures_root or config.local_paths.structures_root,
    )

    if not skip_splits:
        megascale_ids = sorted(
            {
                row.get("protein_id", row.get("name", ""))
                for row in curated_megascale
                if row.get("protein_id") or row.get("name")
            }
        )
        fireprot_ids = sorted(
            {
                row.get("protein_id", row.get("PDB", ""))
                for row in curated_fireprot
                if row.get("protein_id") or row.get("PDB")
            }
        )
        megascale_split = random_protein_split(megascale_ids, seed=config.training.seed)
        fireprot_split = random_protein_split(fireprot_ids, seed=config.training.seed)
        for split_name, protein_ids in megascale_split.items():
            write_split_manifest(config.local_paths.splits_dir, "megascale", split_name, protein_ids)
        for split_name, protein_ids in fireprot_split.items():
            write_split_manifest(config.local_paths.splits_dir, "fireprot", split_name, protein_ids)


def _run_predict(config_path: str, paths_path: str, args: argparse.Namespace) -> None:
    config = load_project_config(config_path, paths_path)
    if args.scan:
        results = run_site_saturation_scan(
            config,
            pdb_path=args.pdb,
            positions=args.positions,
            chain_id=args.chain,
        )
    else:
        mutations = [parse_mutation_string(entry) for entry in args.mutation]
        results = predict_from_pdb(
            config,
            pdb_path=args.pdb,
            mutations=mutations,
            chain_id=args.chain,
        )
    print(
        json.dumps(
            [
                {
                    "protein_id": results.protein_id,
                    "mutation": prediction.mutation.label,
                    "ddg": float(prediction.ddg.detach().cpu()),
                }
                for prediction in results.predictions
            ],
            indent=2,
        )
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "prepare-data":
        _run_prepare_data(args.config, args.paths, args.skip_splits)
        return
    if args.command == "train":
        config = load_project_config(args.config, args.paths)
        metrics = train_model(config)
        print(json.dumps(metrics, indent=2))
        return
    if args.command == "predict":
        _run_predict(args.config, args.paths, args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
