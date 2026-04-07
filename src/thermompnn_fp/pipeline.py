import torch

from thermompnn_fp.head import ThermoMPNNHead

from .types import (
    ALPHABET,
    EMBED_DIM,
    HIDDEN_DIM,
    HeadConfig,
    HeadContext,
    MutationSpec,
    PredictionResult,
    SiteFeatures,
    merge_site_features,
    num_final_layers,
)


def build_mock_proteinmpnn_input(
    sequence_length: int, config: HeadConfig, seed: int = 0
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Create mock ProteinMPNN outputs for a minimal runnable reproduction."""
    torch.manual_seed(seed)
    all_hidden = [
        torch.randn(1, sequence_length, HIDDEN_DIM)
        for _ in range(max(num_final_layers(config), 1))
    ]

    seq_embedding = torch.randn(1, sequence_length, EMBED_DIM)
    return all_hidden, seq_embedding


def extract_site_features(
    all_hidden: list[torch.Tensor],
    seq_embedding: torch.Tensor,
    mutation: MutationSpec,
    config: HeadConfig,
) -> SiteFeatures:
    """Extract per-site features from ProteinMPNN outputs."""
    decoder_hidden = torch.cat(all_hidden[: num_final_layers(config)], dim=-1)[0][
        mutation.mutation_position
    ]
    sequence_embedding = seq_embedding[0][mutation.mutation_position]
    return SiteFeatures(decoder_hidden=decoder_hidden, seq_embedding=sequence_embedding)


def score_amino_acids(head: ThermoMPNNHead, context: HeadContext) -> torch.Tensor:
    """Score amino acids at the mutation site using the head."""

    return head(context.feature_tensor)


def compute_ddg(aa_scores: torch.Tensor, mutation: MutationSpec) -> torch.Tensor:
    """Compute ΔΔG from amino acid scores."""
    wt_idx = ALPHABET.index(mutation.wildtype)
    mut_idx = ALPHABET.index(mutation.mutant)
    return aa_scores[mut_idx] - aa_scores[wt_idx]


def predict_single_mutation(
    head: ThermoMPNNHead,
    all_hidden: list[torch.Tensor],
    seq_embedding: torch.Tensor,
    mutation: MutationSpec,
    config: HeadConfig,
) -> PredictionResult:
    site_features = extract_site_features(all_hidden, seq_embedding, mutation, config)
    head_context = merge_site_features(site_features)
    aa_scores = score_amino_acids(head, head_context)
    ddg = compute_ddg(aa_scores, mutation)
    return PredictionResult(mutation=mutation, aa_scores=aa_scores, ddg=ddg)
