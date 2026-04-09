from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def ensure_tensor(
    value: torch.Tensor | Iterable[float],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    return torch.tensor(list(value), dtype=dtype, device=device)


def radial_basis_encode(
    distances: torch.Tensor,
    *,
    num_bins: int,
    distance_min: float,
    distance_max: float,
) -> torch.Tensor:
    centers = torch.linspace(
        distance_min,
        distance_max,
        num_bins,
        device=distances.device,
        dtype=distances.dtype,
    )
    widths = (distance_max - distance_min) / max(num_bins - 1, 1)
    widths = max(widths, 1e-6)
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / (2 * widths**2))


def pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    diffs = coords.unsqueeze(1) - coords.unsqueeze(0)
    return torch.linalg.norm(diffs, dim=-1)


def knn_indices(ca_coords: torch.Tensor, num_neighbors: int) -> torch.Tensor:
    distances = pairwise_distances(ca_coords)
    distances.fill_diagonal_(math.inf)
    effective_k = min(num_neighbors, max(ca_coords.shape[0] - 1, 1))
    return torch.topk(distances, k=effective_k, largest=False).indices


def gather_neighbors(values: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    flat_idx = neighbor_idx.reshape(-1)
    gathered = values.index_select(0, flat_idx)
    return gathered.reshape(*neighbor_idx.shape, values.shape[-1])


def sequence_to_tensor(sequence: str, alphabet: str) -> torch.Tensor:
    lookup = {aa: index for index, aa in enumerate(alphabet)}
    try:
        indices = [lookup[aa] for aa in sequence]
    except KeyError as exc:
        raise ValueError(f"Encountered unsupported residue in sequence: {exc}") from exc
    return torch.tensor(indices, dtype=torch.long)


def checkpoint_state_dict(payload: dict) -> dict:
    if "model_state_dict" in payload:
        return payload["model_state_dict"]
    if "state_dict" in payload:
        return payload["state_dict"]
    return payload


def normalize_proteinmpnn_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    replacements = (
        (".W_in.", ".w_in."),
        (".W_out.", ".w_out."),
        (".W11.", ".w11."),
        (".W12.", ".w12."),
        (".W13.", ".w13."),
        (".W1.", ".w1."),
        (".W2.", ".w2."),
        (".W3.", ".w3."),
        ("W_e.", "w_e."),
        ("W_s.", "w_s."),
        ("W_out.", "w_out."),
    )
    for key, value in state_dict.items():
        normalized_key = key
        for old, new in replacements:
            normalized_key = normalized_key.replace(old, new)
        normalized[normalized_key] = value
    return normalized


def resolve_path(path: str | Path | None) -> Path | None:
    if not path:
        return None
    return Path(path).expanduser().resolve()


def gather_edges(edges: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)


def gather_nodes(nodes: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    neighbors_flat = neighbor_idx.view(neighbor_idx.shape[0], -1)
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    return neighbor_features.view(*neighbor_idx.shape[:3], -1)


def cat_neighbors_nodes(
    h_nodes: torch.Tensor,
    h_neighbors: torch.Tensor,
    neighbor_idx: torch.Tensor,
) -> torch.Tensor:
    gathered_nodes = gather_nodes(h_nodes, neighbor_idx)
    return torch.cat([h_neighbors, gathered_nodes], dim=-1)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden: int, num_ff: int):
        super().__init__()
        self.w_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.w_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = nn.GELU()

    def forward(self, h_v: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.act(self.w_in(h_v)))


class EncLayer(nn.Module):
    def __init__(self, num_hidden: int, num_in: int, dropout: float = 0.1, scale: int = 30):
        super().__init__()
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.w1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.w2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.w3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.w11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.w12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.w13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(
        self,
        h_v: torch.Tensor,
        h_e: torch.Tensor,
        e_idx: torch.Tensor,
        mask_v: torch.Tensor | None = None,
        mask_attend: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_ev = cat_neighbors_nodes(h_v, h_e, e_idx)
        h_v_expand = h_v.unsqueeze(-2).expand(-1, -1, h_ev.size(-2), -1)
        h_ev = torch.cat([h_v_expand, h_ev], dim=-1)
        h_message = self.w3(self.act(self.w2(self.act(self.w1(h_ev)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, dim=-2) / self.scale
        h_v = self.norm1(h_v + self.dropout1(dh))

        h_v = self.norm2(h_v + self.dropout2(self.dense(h_v)))
        if mask_v is not None:
            h_v = mask_v.unsqueeze(-1) * h_v

        h_ev = cat_neighbors_nodes(h_v, h_e, e_idx)
        h_v_expand = h_v.unsqueeze(-2).expand(-1, -1, h_ev.size(-2), -1)
        h_ev = torch.cat([h_v_expand, h_ev], dim=-1)
        h_message = self.w13(self.act(self.w12(self.act(self.w11(h_ev)))))
        h_e = self.norm3(h_e + self.dropout3(h_message))
        return h_v, h_e


class DecLayer(nn.Module):
    def __init__(self, num_hidden: int, num_in: int, dropout: float = 0.1, scale: int = 30):
        super().__init__()
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.w1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.w2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.w3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(
        self,
        h_v: torch.Tensor,
        h_e: torch.Tensor,
        mask_v: torch.Tensor | None = None,
        mask_attend: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_v_expand = h_v.unsqueeze(-2).expand(-1, -1, h_e.size(-2), -1)
        h_ev = torch.cat([h_v_expand, h_e], dim=-1)

        h_message = self.w3(self.act(self.w2(self.act(self.w1(h_ev)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, dim=-2) / self.scale

        h_v = self.norm1(h_v + self.dropout1(dh))
        h_v = self.norm2(h_v + self.dropout2(self.dense(h_v)))

        if mask_v is not None:
            h_v = mask_v.unsqueeze(-1) * h_v
        return h_v


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings: int, max_relative_feature: int = 32):
        super().__init__()
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 2, num_embeddings)

    def forward(self, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = (
            torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
            * mask
            + (1 - mask) * (2 * self.max_relative_feature + 1)
        )
        one_hot = torch.nn.functional.one_hot(
            encoded, 2 * self.max_relative_feature + 2
        )
        return self.linear(one_hot.float())


class ProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features: int,
        node_features: int,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        top_k: int = 30,
        augment_eps: float = 0.0,
    ):
        super().__init__()
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.num_rbf = num_rbf
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_2d = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        d_x = torch.unsqueeze(x, 1) - torch.unsqueeze(x, 2)
        distances = mask_2d * torch.sqrt(torch.sum(d_x**2, dim=3) + eps)
        d_max, _ = torch.max(distances, dim=-1, keepdim=True)
        d_adjust = distances + (1.0 - mask_2d) * d_max
        d_neighbors, e_idx = torch.topk(
            d_adjust,
            np.minimum(self.top_k, x.shape[1]),
            dim=-1,
            largest=False,
        )
        return d_neighbors, e_idx

    def _rbf(self, distances: torch.Tensor) -> torch.Tensor:
        d_min, d_max, d_count = 2.0, 22.0, self.num_rbf
        d_mu = torch.linspace(d_min, d_max, d_count, device=distances.device)
        d_mu = d_mu.view(1, 1, 1, -1)
        d_sigma = (d_max - d_min) / d_count
        d_expand = torch.unsqueeze(distances, -1)
        return torch.exp(-((d_expand - d_mu) / d_sigma) ** 2)

    def _get_rbf(self, a: torch.Tensor, b: torch.Tensor, e_idx: torch.Tensor) -> torch.Tensor:
        distances = torch.sqrt(torch.sum((a[:, :, None, :] - b[:, None, :, :]) ** 2, dim=-1) + 1e-6)
        distances = gather_edges(distances[:, :, :, None], e_idx)[:, :, :, 0]
        return self._rbf(distances)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.augment_eps > 0:
            x = x + self.augment_eps * torch.randn_like(x)

        b = x[:, :, 1, :] - x[:, :, 0, :]
        c = x[:, :, 2, :] - x[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        c_beta = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + x[:, :, 1, :]
        ca = x[:, :, 1, :]
        n = x[:, :, 0, :]
        c_atom = x[:, :, 2, :]
        o = x[:, :, 3, :]

        d_neighbors, e_idx = self._dist(ca, mask)

        rbf_all = [
            self._rbf(d_neighbors),
            self._get_rbf(n, n, e_idx),
            self._get_rbf(c_atom, c_atom, e_idx),
            self._get_rbf(o, o, e_idx),
            self._get_rbf(c_beta, c_beta, e_idx),
            self._get_rbf(ca, n, e_idx),
            self._get_rbf(ca, c_atom, e_idx),
            self._get_rbf(ca, o, e_idx),
            self._get_rbf(ca, c_beta, e_idx),
            self._get_rbf(n, c_atom, e_idx),
            self._get_rbf(n, o, e_idx),
            self._get_rbf(n, c_beta, e_idx),
            self._get_rbf(c_beta, c_atom, e_idx),
            self._get_rbf(c_beta, o, e_idx),
            self._get_rbf(o, c_atom, e_idx),
            self._get_rbf(n, ca, e_idx),
            self._get_rbf(c_atom, ca, e_idx),
            self._get_rbf(o, ca, e_idx),
            self._get_rbf(c_beta, ca, e_idx),
            self._get_rbf(c_atom, n, e_idx),
            self._get_rbf(o, n, e_idx),
            self._get_rbf(c_beta, n, e_idx),
            self._get_rbf(c_atom, c_beta, e_idx),
            self._get_rbf(o, c_beta, e_idx),
            self._get_rbf(c_atom, o, e_idx),
        ]
        rbf_all = torch.cat(rbf_all, dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], e_idx)[:, :, :, 0]

        same_chain = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        e_chains = gather_edges(same_chain[:, :, :, None], e_idx)[:, :, :, 0]
        e_positional = self.embeddings(offset.long(), e_chains)
        e = torch.cat((e_positional, rbf_all), dim=-1)
        e = self.edge_embedding(e)
        e = self.norm_edges(e)
        return e, e_idx


class ProteinMPNN(nn.Module):
    def __init__(
        self,
        num_letters: int = 21,
        node_features: int = 128,
        edge_features: int = 128,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab: int = 21,
        k_neighbors: int = 32,
        augment_eps: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.features = ProteinFeatures(
            node_features,
            edge_features,
            top_k=k_neighbors,
            augment_eps=augment_eps,
        )
        self.w_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.w_s = nn.Embedding(vocab, hidden_dim)
        self.encoder_layers = nn.ModuleList(
            [EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(num_encoder_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout) for _ in range(num_decoder_layers)]
        )
        self.w_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        mask: torch.Tensor,
        chain_m: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        randn: torch.Tensor | None = None,
        use_input_decoding_order: bool = False,
        decoding_order: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        device = x.device
        e, e_idx = self.features(x, mask, residue_idx, chain_encoding_all)
        h_v = torch.zeros((e.shape[0], e.shape[1], e.shape[-1]), device=device)
        h_e = self.w_e(e)

        mask_attend = gather_nodes(mask.unsqueeze(-1), e_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_v, h_e = layer(h_v, h_e, e_idx, mask, mask_attend)

        h_s = self.w_s(s)
        h_es = cat_neighbors_nodes(h_s, h_e, e_idx)
        h_ex_encoder = cat_neighbors_nodes(torch.zeros_like(h_s), h_e, e_idx)
        h_exv_encoder = cat_neighbors_nodes(h_v, h_ex_encoder, e_idx)

        chain_m = chain_m * mask
        if not use_input_decoding_order:
            decoding_order = torch.arange(x.size(1), device=device).unsqueeze(0)
        elif decoding_order is None:
            if randn is None:
                randn = torch.randn_like(mask)
            decoding_order = torch.argsort((chain_m + 0.0001) * torch.abs(randn))

        mask_size = e_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(
            decoding_order, num_classes=mask_size
        ).float()
        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            1 - torch.triu(torch.ones(mask_size, mask_size, device=device)),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        # ThermoMPNN-style feature extraction uses fully visible decoding.
        order_mask_backward = torch.ones_like(order_mask_backward)

        mask_attend = torch.gather(order_mask_backward, 2, e_idx).unsqueeze(-1)
        mask_1d = mask.view(mask.size(0), mask.size(1), 1, 1)
        mask_bw = mask_1d * mask_attend
        mask_fw = mask_1d * (1.0 - mask_attend)

        all_hidden: list[torch.Tensor] = []
        h_exv_encoder_fw = mask_fw * h_exv_encoder
        for layer in self.decoder_layers:
            h_esv = cat_neighbors_nodes(h_v, h_es, e_idx)
            h_esv = mask_bw * h_esv + h_exv_encoder_fw
            h_v = layer(h_v, h_esv, mask)
            all_hidden.append(h_v)

        logits = self.w_out(h_v)
        log_probs = F.log_softmax(logits, dim=-1)
        return list(reversed(all_hidden)), h_s, log_probs
