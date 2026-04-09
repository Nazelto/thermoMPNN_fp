from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from torch.utils.data import Dataset

from .types import DatasetItem, MutationRecord, ProteinRecord


def _load_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _get(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    raise KeyError(f"Missing required columns. Tried: {names!r}")


def _mutation_from_row(row: dict[str, str], source: str) -> MutationRecord:
    position = int(_get(row, "position", "resi", "residue_index", "mutation_position"))
    wildtype = _get(row, "wildtype", "wt", "wtAA", "aa_wt")
    mutant = _get(row, "mutant", "mt", "mutAA", "aa_mut")
    ddg_raw = row.get("ddg") or row.get("ddG") or row.get("ddG_ML") or row.get("score")
    ddg = float(ddg_raw) if ddg_raw not in (None, "", "-") else None
    if position >= 1:
        return MutationRecord.from_one_based(
            position=position,
            wildtype=wildtype,
            mutant=mutant,
            ddg=ddg,
            source=source,
            extras=row,
        )
    return MutationRecord(
        position=position,
        wildtype=wildtype,
        mutant=mutant,
        ddg=ddg,
        source=source,
        extras=row,
    )


def _rows_to_proteins(
    rows: Iterable[dict[str, str]],
    *,
    structure_root: str | Path,
    source: str,
    split_manifest: str | Path | None = None,
) -> list[ProteinRecord]:
    structure_root = Path(structure_root)
    allowed_ids: set[str] | None = None
    if split_manifest:
        with Path(split_manifest).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        allowed_ids = set(payload.get("proteins", []))

    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        protein_id = _get(row, "protein_id", "name", "uid", "pdb_id", "PDB")
        grouped_rows[protein_id].append(row)

    proteins: list[ProteinRecord] = []
    for protein_id, protein_rows in grouped_rows.items():
        if allowed_ids is not None and protein_id not in allowed_ids:
            continue
        pdb_file = protein_rows[0].get("pdb_path") or protein_rows[0].get("structure_path")
        if pdb_file:
            pdb_path = Path(pdb_file)
        else:
            pdb_path = structure_root / f"{protein_id}.pdb"
        mutations = [_mutation_from_row(row, source=source) for row in protein_rows]
        proteins.append(
            ProteinRecord(
                protein_id=protein_id,
                pdb_path=pdb_path,
                sequence=protein_rows[0].get("sequence"),
                chain_id=protein_rows[0].get("chain_id"),
                mutations=mutations,
                metadata={"source": source},
            )
        )
    return proteins


class ProteinMutationDataset(Dataset[DatasetItem]):
    def __init__(self, proteins: list[ProteinRecord]):
        self.proteins = proteins

    def __len__(self) -> int:
        return len(self.proteins)

    def __getitem__(self, index: int) -> DatasetItem:
        protein = self.proteins[index]
        return DatasetItem(protein=protein, mutations=protein.mutations)


class MegaScaleDataset(ProteinMutationDataset):
    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        *,
        structure_root: str | Path,
        split_manifest: str | Path | None = None,
    ) -> "MegaScaleDataset":
        rows = _load_rows(csv_path)
        proteins = _rows_to_proteins(
            rows,
            structure_root=structure_root,
            source="megascale",
            split_manifest=split_manifest,
        )
        return cls(proteins)


class FireProtDataset(ProteinMutationDataset):
    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        *,
        structure_root: str | Path,
        split_manifest: str | Path | None = None,
    ) -> "FireProtDataset":
        rows = _load_rows(csv_path)
        proteins = _rows_to_proteins(
            rows,
            structure_root=structure_root,
            source="fireprot",
            split_manifest=split_manifest,
        )
        return cls(proteins)


def collate_protein_batches(batch: list[DatasetItem]) -> list[DatasetItem]:
    return batch
