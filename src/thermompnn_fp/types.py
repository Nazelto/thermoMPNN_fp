from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import torch

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
VOCAB_DIM = len(ALPHABET)
HIDDEN_DIM = 128
EMBED_DIM = 128
BACKBONE_ATOMS = ("N", "CA", "C", "O", "CB")
THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


def aa_index(amino_acid: str) -> int:
    if amino_acid not in ALPHABET:
        raise ValueError(f"Unsupported amino acid: {amino_acid!r}")
    return ALPHABET.index(amino_acid)


@dataclass(frozen=True)
class MutationRecord:
    position: int
    wildtype: str
    mutant: str
    ddg: float | None = None
    source: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError("Mutation positions must be stored as 0-based indices.")
        if self.wildtype not in ALPHABET:
            raise ValueError(f"Invalid wildtype residue: {self.wildtype}")
        if self.mutant not in ALPHABET:
            raise ValueError(f"Invalid mutant residue: {self.mutant}")

    @classmethod
    def from_one_based(
        cls,
        position: int,
        wildtype: str,
        mutant: str,
        ddg: float | None = None,
        source: str | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> "MutationRecord":
        return cls(
            position=position - 1,
            wildtype=wildtype,
            mutant=mutant,
            ddg=ddg,
            source=source,
            extras=extras or {},
        )

    @property
    def label(self) -> str:
        return f"{self.wildtype}{self.position + 1}{self.mutant}"


@dataclass
class ProteinRecord:
    protein_id: str
    pdb_path: Path
    sequence: str | None = None
    chain_id: str | None = None
    mutations: list[MutationRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetItem:
    protein: ProteinRecord
    mutations: list[MutationRecord]


@dataclass
class BackboneInput:
    protein_id: str
    sequence: str
    sequence_tensor: torch.Tensor
    atom_coords: torch.Tensor
    ca_coords: torch.Tensor
    neighbor_idx: torch.Tensor
    edge_features: torch.Tensor
    residue_idx: torch.Tensor
    mask: torch.Tensor
    chain_id: str | None = None


@dataclass
class BackboneOutput:
    decoder_hidden_states: list[torch.Tensor]
    sequence_embedding: torch.Tensor
    node_embedding: torch.Tensor


@dataclass
class PredictionResult:
    mutation: MutationRecord
    aa_scores: torch.Tensor
    ddg: torch.Tensor


@dataclass
class BatchPrediction:
    protein_id: str
    predictions: list[PredictionResult]


@dataclass
class ModelConfig:
    hidden_dims: tuple[int, ...] = (64, 32)
    num_final_layers: int = 2
    use_light_attention: bool = True
    freeze_backbone: bool = True
    load_pretrained: bool = True
    subtract_mutation: bool = True
    hidden_dim: int = HIDDEN_DIM
    embedding_dim: int = EMBED_DIM
    num_neighbors: int = 48
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    kernel_size: int = 9
    attention_dropout: float = 0.25
    rbf_bins: int = 16
    rbf_distance_min: float = 0.0
    rbf_distance_max: float = 20.0
    strict_backbone_weights: bool = False

    @property
    def input_dim(self) -> int:
        return self.embedding_dim + self.hidden_dim * self.num_final_layers


@dataclass
class TrainConfig:
    dataset_name: Literal["megascale", "fireprot", "combo"] = "megascale"
    batch_size: int = 1
    num_workers: int = 0
    learning_rate: float = 1e-3
    mpnn_learning_rate: float | None = None
    weight_decay: float = 1e-2
    epochs: int = 100
    device: str = "cpu"
    validation_metric: Literal["pcc", "scc", "rmse"] = "pcc"
    checkpoint_dir: str = "checkpoints"
    seed: int = 0
    max_proteins: int | None = None
    max_mutations_per_protein: int | None = None
    clip_grad_norm: float | None = None


@dataclass
class LocalPaths:
    proteinmpnn_checkpoint: str = ""
    thermompnn_transfer_checkpoint: str = ""
    megascale_raw_csv: str = ""
    fireprot_raw_csv: str = ""
    megascale_curated_csv: str = "data/megascale_curated.csv"
    fireprot_curated_csv: str = "data/fireprot_curated.csv"
    structures_root: str = ""
    megascale_structures_root: str = ""
    fireprot_structures_root: str = ""
    splits_dir: str = "data/splits"
    mmseqs_bin: str = "mmseqs"
    output_dir: str = "output"


@dataclass
class ProjectConfig:
    name: str = "thermompnn-fp"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    local_paths: LocalPaths = field(default_factory=LocalPaths)


def ddg_from_scores(aa_scores: torch.Tensor, mutation: MutationRecord) -> torch.Tensor:
    return aa_scores[aa_index(mutation.mutant)] - aa_scores[aa_index(mutation.wildtype)]
