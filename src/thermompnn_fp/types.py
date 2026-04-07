from typing import Sequence
import pydantic
import torch


ALPHABET: str = "ACDEFGHIKLMNPQRSTVWYX"
HIDDEN_DIM: int = 128
EMBED_DIM: int = 128
VOCAB_DIM: int = len(ALPHABET)


class MutationSpec(pydantic.BaseModel):
    """Specification of a single mutation for input to the model."""

    mutation_position: int = pydantic.Field(
        max_digits=100, description="Position of the mutation in the protein sequence."
    )
    wildtype: str
    mutant: str

    @pydantic.field_validator("wildtype")
    @classmethod
    def validate_wildtype(cls, value: str) -> str:
        if value not in ALPHABET:
            raise ValueError(
                f"Invalid wildtype amino acid: {value}. Must be one of {ALPHABET}."
            )
        return value

    @pydantic.field_validator("mutant")
    @classmethod
    def validate_mutant(cls, value: str) -> str:
        if value not in ALPHABET:
            raise ValueError(
                f"Invalid mutant amino acid: {value}. Must be one of {ALPHABET}."
            )
        return value


class HeadConfig(pydantic.BaseModel):
    """Configuration for a single head in the model."""

    num_final_layers: int = pydantic.Field(
        default=2, description="Number of final layers in the head."
    )
    hidden_dims: tuple[int, ...] = (64, 32)
    use_light_attention: bool = pydantic.Field(default=True)


def input_dim(config: HeadConfig) -> int:
    return HIDDEN_DIM * config.num_final_layers + EMBED_DIM


def num_final_layers(config: HeadConfig) -> int:
    return config.num_final_layers


def hidden_dims(config: HeadConfig) -> Sequence[int]:
    return config.hidden_dims


def layer(config: HeadConfig) -> list[int]:
    return [input_dim(config), *hidden_dims(config), VOCAB_DIM]


def use_light_attention(config: HeadConfig) -> bool:
    return config.use_light_attention


class HeadContext(pydantic.BaseModel):
    """Context for a single head in the model."""

    # 目前来说是 HIDDEN_DIM*2 + EMBED_DIM = 384 维
    feature_tensor: torch.Tensor

    @pydantic.field_validator("feature_tensor")
    @classmethod
    def validate_feature_tensor(cls, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[2] != input_dim(HeadConfig()):
            raise ValueError(
                f"Invalid feature tensor shape: {value.shape}. Expected (batch_size, seq_length, {input_dim(HeadConfig())})."
            )
        return value


class SiteFeatures(pydantic.BaseModel):
    """Per-site features extracted from ProteinMPNN outputs."""

    decoder_hidden: torch.Tensor
    seq_embedding: torch.Tensor
    # TODO: validate


def merge_site_features(site_features: SiteFeatures) -> HeadContext:
    """Merge per-site features into a single tensor for input to the head."""
    return HeadContext(
        feature_tensor=torch.cat(
            [site_features.decoder_hidden, site_features.seq_embedding], dim=-1
        )
    )


class PredictionResult(pydantic.BaseModel):
    """Minimal prediction payload."""

    mutation: MutationSpec
    aa_scores: torch.Tensor
    ddg: torch.Tensor
