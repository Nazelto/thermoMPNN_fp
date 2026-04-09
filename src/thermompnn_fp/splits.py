from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from .types import ProteinRecord


def write_fasta(proteins: Iterable[ProteinRecord], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for protein in proteins:
            if not protein.sequence:
                continue
            handle.write(f">{protein.protein_id}\n{protein.sequence}\n")
    return output_path


def _ensure_mmseqs(mmseqs_bin: str) -> str:
    resolved = shutil.which(mmseqs_bin) or mmseqs_bin
    if shutil.which(resolved) is None and not Path(resolved).exists():
        raise FileNotFoundError(
            f"MMseqs2 executable not found: {mmseqs_bin}. Install MMseqs2 and update local_paths.yaml."
        )
    return resolved


def run_mmseqs_easy_cluster(
    fasta_path: str | Path,
    output_prefix: str | Path,
    *,
    mmseqs_bin: str = "mmseqs",
    min_seq_id: float = 0.25,
) -> Path:
    mmseqs = _ensure_mmseqs(mmseqs_bin)
    output_prefix = Path(output_prefix)
    tmp_dir = output_prefix.parent / f"{output_prefix.name}_tmp"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            mmseqs,
            "easy-cluster",
            str(fasta_path),
            str(output_prefix),
            str(tmp_dir),
            "--min-seq-id",
            str(min_seq_id),
        ],
        check=True,
    )
    return output_prefix.with_name(f"{output_prefix.name}_cluster.tsv")


def run_mmseqs_easy_search(
    query_fasta: str | Path,
    target_fasta: str | Path,
    output_tsv: str | Path,
    *,
    mmseqs_bin: str = "mmseqs",
    min_seq_id: float = 0.25,
) -> Path:
    mmseqs = _ensure_mmseqs(mmseqs_bin)
    output_tsv = Path(output_tsv)
    tmp_dir = output_tsv.parent / f"{output_tsv.stem}_tmp"
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            mmseqs,
            "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(output_tsv),
            str(tmp_dir),
            "--min-seq-id",
            str(min_seq_id),
        ],
        check=True,
    )
    return output_tsv


def random_protein_split(
    protein_ids: list[str],
    *,
    seed: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[str]]:
    shuffled = list(protein_ids)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_split_manifest(
    split_dir: str | Path,
    dataset_name: str,
    split_name: str,
    protein_ids: list[str],
) -> Path:
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_dir / f"{dataset_name}_{split_name}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"dataset": dataset_name, "split": split_name, "proteins": protein_ids},
            handle,
            indent=2,
        )
    return output_path
