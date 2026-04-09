from __future__ import annotations

from pathlib import Path

import torch

from .protein_mpnn_utils import knn_indices, radial_basis_encode, sequence_to_tensor
from .types import ALPHABET, BACKBONE_ATOMS, BackboneInput, ProteinRecord, THREE_TO_ONE


def _parse_pdb_atom_line(
    line: str,
) -> tuple[str, str, str, int, tuple[float, float, float]] | None:
    if not line.startswith(("ATOM", "HETATM")):
        return None
    atom_name = line[12:16].strip()
    residue_name = line[17:20].strip()
    chain_id = line[21].strip() or "A"
    residue_seq = int(line[22:26].strip())
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    return atom_name, residue_name, chain_id, residue_seq, (x, y, z)


def _synthesize_cb(n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    b = ca - n
    c_vec = c - ca
    a = torch.cross(b, c_vec, dim=0)
    return ca + 0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec


def parse_pdb_backbone(
    pdb_path: str | Path,
    chain_id: str | None = None,
) -> tuple[str, torch.Tensor, str]:
    residues: list[tuple[int, str, dict[str, tuple[float, float, float]]]] = []
    current_key: tuple[str, int] | None = None
    current_atoms: dict[str, tuple[float, float, float]] = {}
    current_residue_name = ""
    selected_chain = chain_id

    with Path(pdb_path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parsed = _parse_pdb_atom_line(raw_line)
            if parsed is None:
                continue
            atom_name, residue_name, line_chain_id, residue_seq, coords = parsed
            if atom_name not in BACKBONE_ATOMS[:-1]:
                continue
            if selected_chain is None:
                selected_chain = line_chain_id
            if line_chain_id != selected_chain:
                continue

            residue_key = (line_chain_id, residue_seq)
            if current_key is None:
                current_key = residue_key
                current_residue_name = residue_name
            if residue_key != current_key:
                residues.append((current_key[1], current_residue_name, current_atoms))
                current_key = residue_key
                current_residue_name = residue_name
                current_atoms = {}
            current_atoms[atom_name] = coords

    if current_key is not None:
        residues.append((current_key[1], current_residue_name, current_atoms))

    sequence_chars: list[str] = []
    coords_list: list[torch.Tensor] = []
    for _, residue_name, atoms in residues:
        if residue_name not in THREE_TO_ONE:
            continue
        required = {"N", "CA", "C", "O"}
        if not required.issubset(atoms):
            continue
        n = torch.tensor(atoms["N"], dtype=torch.float32)
        ca = torch.tensor(atoms["CA"], dtype=torch.float32)
        c = torch.tensor(atoms["C"], dtype=torch.float32)
        o = torch.tensor(atoms["O"], dtype=torch.float32)
        cb = (
            torch.tensor(atoms["CB"], dtype=torch.float32)
            if "CB" in atoms
            else _synthesize_cb(n, ca, c)
        )
        coords_list.append(torch.stack([n, ca, c, o, cb], dim=0))
        sequence_chars.append(THREE_TO_ONE[residue_name])

    if not coords_list:
        raise ValueError(f"No usable residues found in {pdb_path}.")

    return (
        "".join(sequence_chars),
        torch.stack(coords_list, dim=0),
        selected_chain or "A",
    )


def build_edge_features(
    atom_coords: torch.Tensor,
    neighbor_idx: torch.Tensor,  # [L,K]
    rbf_bins: int,
    distance_min: float,
    distance_max: float,
) -> torch.Tensor:
    source_atoms = atom_coords.unsqueeze(1).expand(-1, neighbor_idx.shape[1], -1, -1)
    neighbor_atoms = atom_coords.index_select(0, neighbor_idx.reshape(-1)).reshape(
        atom_coords.shape[0], neighbor_idx.shape[1], atom_coords.shape[1], 3
    )
    atom_pair_distances = torch.cdist(
        source_atoms.reshape(-1, 5, 3),
        neighbor_atoms.reshape(-1, 5, 3),
    )  # [L*K,5,5] 计算source atom 和 neighbor_atom 的两两欧氏距离
    rbf = radial_basis_encode(
        atom_pair_distances.reshape(atom_coords.shape[0], neighbor_idx.shape[1], -1),
        num_bins=rbf_bins,
        distance_min=distance_min,
        distance_max=distance_max,
    )
    return rbf.reshape(atom_coords.shape[0], neighbor_idx.shape[1], -1)


def featurize_protein(
    protein: ProteinRecord,
    *,
    num_neighbors: int,
    rbf_bins: int,
    distance_min: float,
    distance_max: float,
    device: str | torch.device | None = None,
) -> BackboneInput:
    sequence, atom_coords, parsed_chain_id = parse_pdb_backbone(
        protein.pdb_path, chain_id=protein.chain_id
    )
    sequence_tensor = sequence_to_tensor(sequence, ALPHABET)
    ca_coords = atom_coords[:, 1, :]
    neighbor_idx = knn_indices(ca_coords, num_neighbors=num_neighbors)
    edge_features = build_edge_features(
        atom_coords,
        neighbor_idx,
        rbf_bins=rbf_bins,
        distance_min=distance_min,
        distance_max=distance_max,
    )
    residue_idx = torch.arange(len(sequence), dtype=torch.long)
    mask = torch.ones(len(sequence), dtype=torch.bool)

    if device is not None:
        atom_coords = atom_coords.to(device)
        ca_coords = ca_coords.to(device)
        sequence_tensor = sequence_tensor.to(device)
        neighbor_idx = neighbor_idx.to(device)
        edge_features = edge_features.to(device)
        residue_idx = residue_idx.to(device)
        mask = mask.to(device)

    return BackboneInput(
        protein_id=protein.protein_id,
        sequence=sequence,
        sequence_tensor=sequence_tensor,
        atom_coords=atom_coords,
        ca_coords=ca_coords,
        neighbor_idx=neighbor_idx,
        edge_features=edge_features,
        residue_idx=residue_idx,
        mask=mask,
        chain_id=protein.chain_id or parsed_chain_id,
    )
