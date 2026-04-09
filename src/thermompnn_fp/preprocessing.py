from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from .featurize import parse_pdb_backbone


def _read_rows(csv_path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        if reader.fieldnames is None:
            raise ValueError(f"No columns found in {csv_path}")
        return list(reader.fieldnames), rows


def _write_rows(csv_path: str | Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _drop_rows(rows: list[dict[str, str]], predicate: Callable[[dict[str, str]], bool]) -> list[dict[str, str]]:
    return [row for row in rows if predicate(row)]


def curate_megascale_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    unreliable_column: str = "ddG_ML",
    mutation_type_column: str = "mut_type",
    modified_wt_column: str = "is_perturbed_wt",
) -> list[dict[str, str]]:
    fieldnames, rows = _read_rows(input_csv)

    def is_valid(row: dict[str, str]) -> bool:
        ddg_value = row.get(unreliable_column, "")
        if ddg_value in ("", "-", "nan", "NaN", "None"):
            return False
        mutation_type = row.get(mutation_type_column, "").lower()
        if any(token in mutation_type for token in ("del", "ins", "double", "multi")):
            return False
        modified_wt = row.get(modified_wt_column, "").lower()
        if modified_wt in {"1", "true", "yes"}:
            return False
        return True

    curated = _drop_rows(rows, is_valid)
    _write_rows(output_csv, fieldnames, curated)
    return curated


def _closest_to_ph_74(rows: list[dict[str, str]]) -> dict[str, str]:
    def score(row: dict[str, str]) -> float:
        ph = row.get("pH") or row.get("ph") or "7.4"
        try:
            return abs(float(ph) - 7.4)
        except ValueError:
            return 999.0

    return min(rows, key=score)


def curate_fireprot_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    structure_root: str | Path | None = None,
) -> list[dict[str, str]]:
    fieldnames, rows = _read_rows(input_csv)
    required_fields = {"ddG", "PDB", "position", "wildtype", "mutant"}
    curated = [
        row
        for row in rows
        if required_fields.issubset({key for key, value in row.items() if value not in ("", None)})
    ]

    deduped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in curated:
        key = (
            row.get("UniProt_ID", ""),
            row.get("PDB", ""),
            row.get("position", ""),
            row.get("mutation", row.get("mutant", "")),
        )
        deduped.setdefault(key, []).append(row)

    selected_rows = [_closest_to_ph_74(group_rows) for group_rows in deduped.values()]

    if structure_root:
        structure_root = Path(structure_root)
        for row in selected_rows:
            pdb_id = row.get("PDB", "")
            row["pdb_path"] = str(structure_root / f"{pdb_id}.pdb")
            if Path(row["pdb_path"]).exists():
                try:
                    sequence, _, chain_id = parse_pdb_backbone(row["pdb_path"])
                except Exception:
                    continue
                row["pdb_sequence"] = sequence
                row["chain_id"] = chain_id

    output_fields = list(dict.fromkeys(fieldnames + ["pdb_path", "pdb_sequence", "chain_id"]))
    _write_rows(output_csv, output_fields, selected_rows)
    return selected_rows
