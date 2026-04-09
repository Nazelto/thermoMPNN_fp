from __future__ import annotations

import csv
from pathlib import Path

from .featurize import featurize_protein
from .pipeline import load_model, predict_mutations
from .types import ALPHABET, BatchPrediction, MutationRecord, ProteinRecord, ProjectConfig


def parse_mutation_string(mutation: str) -> MutationRecord:
    mutation = mutation.strip().upper()
    if len(mutation) < 3:
        raise ValueError(f"Invalid mutation string: {mutation!r}")
    wildtype = mutation[0]
    mutant = mutation[-1]
    position = int(mutation[1:-1])
    return MutationRecord.from_one_based(position=position, wildtype=wildtype, mutant=mutant)


def load_configured_model(config: ProjectConfig):
    checkpoint_path = Path(config.training.checkpoint_dir) / f"{config.name}_best.pt"
    effective_checkpoint_path: str | None
    if checkpoint_path.exists():
        effective_checkpoint_path = str(checkpoint_path)
    else:
        effective_checkpoint_path = (
            config.local_paths.thermompnn_transfer_checkpoint or None
        )
    return load_model(
        config.model,
        checkpoint_path=effective_checkpoint_path,
        model_weights_path=config.local_paths.proteinmpnn_checkpoint or None,
        device=config.training.device,
    )


def predict_from_pdb(
    config: ProjectConfig,
    pdb_path: str | Path,
    mutations: list[MutationRecord],
    *,
    protein_id: str | None = None,
    chain_id: str | None = None,
) -> BatchPrediction:
    model = load_configured_model(config)
    protein = ProteinRecord(
        protein_id=protein_id or Path(pdb_path).stem,
        pdb_path=Path(pdb_path),
        chain_id=chain_id,
        mutations=mutations,
    )
    backbone_input = featurize_protein(
        protein,
        num_neighbors=config.model.num_neighbors,
        rbf_bins=config.model.rbf_bins,
        distance_min=config.model.rbf_distance_min,
        distance_max=config.model.rbf_distance_max,
        device=config.training.device,
    )
    return predict_mutations(model, backbone_input, mutations)


def run_site_saturation_scan(
    config: ProjectConfig,
    pdb_path: str | Path,
    positions: list[int] | None = None,
    *,
    protein_id: str | None = None,
    chain_id: str | None = None,
    exclude_wildtype: bool = True,
) -> BatchPrediction:
    protein = ProteinRecord(
        protein_id=protein_id or Path(pdb_path).stem,
        pdb_path=Path(pdb_path),
        chain_id=chain_id,
    )
    backbone_input = featurize_protein(
        protein,
        num_neighbors=config.model.num_neighbors,
        rbf_bins=config.model.rbf_bins,
        distance_min=config.model.rbf_distance_min,
        distance_max=config.model.rbf_distance_max,
        device=config.training.device,
    )
    if positions is None:
        positions = list(range(1, len(backbone_input.sequence) + 1))

    mutations: list[MutationRecord] = []
    for one_based_position in positions:
        wt = backbone_input.sequence[one_based_position - 1]
        for mutant in ALPHABET[:-1]:
            if exclude_wildtype and mutant == wt:
                continue
            mutations.append(
                MutationRecord.from_one_based(
                    position=one_based_position,
                    wildtype=wt,
                    mutant=mutant,
                )
            )
    model = load_configured_model(config)
    return predict_mutations(model, backbone_input, mutations)


def predict_mutations_from_csv(
    config: ProjectConfig,
    csv_path: str | Path,
    *,
    pdb_path: str | Path,
    protein_id: str | None = None,
    chain_id: str | None = None,
) -> BatchPrediction:
    mutations: list[MutationRecord] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mutations.append(
                MutationRecord.from_one_based(
                    position=int(row["position"]),
                    wildtype=row["wildtype"],
                    mutant=row["mutant"],
                    ddg=float(row["ddg"]) if row.get("ddg") else None,
                )
            )
    return predict_from_pdb(
        config,
        pdb_path=pdb_path,
        mutations=mutations,
        protein_id=protein_id,
        chain_id=chain_id,
    )
