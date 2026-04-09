from __future__ import annotations

import torch

from thermompnn_fp.attention import LightAttention
from thermompnn_fp.head import ThermoMPNNHead
from thermompnn_fp.pipeline import (
    ThermoMPNNModel,
    convert_original_thermompnn_state_dict,
    extract_site_feature_vector,
)
from thermompnn_fp.proteinmpnn_backbone import ProteinMPNNBackbone
from thermompnn_fp.types import (
    BackboneInput,
    BackboneOutput,
    ModelConfig,
    MutationRecord,
    ddg_from_scores,
)


def test_default_input_dim_is_384() -> None:
    config = ModelConfig()
    assert config.input_dim == 384


def test_light_attention_preserves_shape() -> None:
    module = LightAttention(embeddings_dim=384)
    x = torch.randn(2, 384, 3)
    out = module(x)
    assert out.shape == x.shape


def test_head_outputs_vocab_scores() -> None:
    config = ModelConfig()
    head = ThermoMPNNHead(config)
    x = torch.randn(5, config.input_dim)
    out = head(x)
    assert out.shape == (5, 21)
    assert isinstance(head.mlp[0], torch.nn.ReLU)
    assert isinstance(head.mlp[1], torch.nn.Linear)


def test_ddg_is_mut_minus_wt() -> None:
    mutation = MutationRecord.from_one_based(position=1, wildtype="A", mutant="V")
    scores = torch.zeros(21)
    scores[0] = 1.0
    scores[17] = 3.5
    assert torch.isclose(ddg_from_scores(scores, mutation), torch.tensor(2.5))


def test_extract_site_feature_vector_uses_requested_layers() -> None:
    config = ModelConfig(num_final_layers=2)
    output = BackboneOutput(
        decoder_hidden_states=[torch.ones(4, 128), torch.full((4, 128), 2.0)],
        sequence_embedding=torch.full((4, 128), 3.0),
        node_embedding=torch.zeros(4, 128),
    )
    mutation = MutationRecord.from_one_based(position=2, wildtype="A", mutant="V")
    feature_vector = extract_site_feature_vector(output, mutation, config)
    assert feature_vector.shape == (384,)


def test_original_backbone_wrapper_preserves_current_output_contract() -> None:
    config = ModelConfig()
    backbone = ProteinMPNNBackbone(config)
    length = 5
    backbone_input = BackboneInput(
        protein_id="toy",
        sequence="ACDEF",
        sequence_tensor=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        atom_coords=torch.randn(length, 5, 3),
        ca_coords=torch.randn(length, 3),
        neighbor_idx=torch.zeros(length, 1, dtype=torch.long),
        edge_features=torch.zeros(length, 1, 25 * config.rbf_bins),
        residue_idx=torch.arange(length, dtype=torch.long),
        mask=torch.ones(length, dtype=torch.bool),
        chain_id="A",
    )
    output = backbone(backbone_input)
    assert len(output.decoder_hidden_states) == config.num_decoder_layers
    assert output.decoder_hidden_states[0].shape == (length, config.hidden_dim)
    assert output.sequence_embedding.shape == (length, config.embedding_dim)
    assert output.node_embedding.shape == (length, config.hidden_dim)


def test_convert_original_thermompnn_state_dict_maps_backbone_and_head() -> None:
    model = ThermoMPNNModel(ModelConfig())
    original_state_dict = {
        "model.prot_mpnn.W_e.weight": torch.full_like(model.backbone.mpnn.w_e.weight, 2.0),
        "model.prot_mpnn.W_e.bias": torch.full_like(model.backbone.mpnn.w_e.bias, 3.0),
        "model.light_attention.feature_convolution.weight": torch.full_like(
            model.head.light_attention.feature_convolution.weight, 4.0
        ),
        "model.light_attention.feature_convolution.bias": torch.full_like(
            model.head.light_attention.feature_convolution.bias, 5.0
        ),
        "model.both_out.1.weight": torch.full_like(model.head.mlp[1].weight, 6.0),
        "model.both_out.1.bias": torch.full_like(model.head.mlp[1].bias, 7.0),
        "model.both_out.3.weight": torch.full_like(model.head.mlp[3].weight, 8.0),
        "model.both_out.3.bias": torch.full_like(model.head.mlp[3].bias, 9.0),
        "model.both_out.5.weight": torch.full_like(model.head.mlp[5].weight, 10.0),
        "model.both_out.5.bias": torch.full_like(model.head.mlp[5].bias, 11.0),
        "model.ddg_out.weight": torch.tensor([[0.5]]),
        "model.ddg_out.bias": torch.tensor([1.25]),
    }

    converted = convert_original_thermompnn_state_dict(original_state_dict, model)

    assert torch.allclose(converted["backbone.mpnn.w_e.weight"], torch.full_like(model.backbone.mpnn.w_e.weight, 2.0))
    assert torch.allclose(converted["backbone.mpnn.w_e.bias"], torch.full_like(model.backbone.mpnn.w_e.bias, 3.0))
    assert torch.allclose(
        converted["head.light_attention.feature_convolution.weight"],
        torch.full_like(model.head.light_attention.feature_convolution.weight, 4.0),
    )
    assert torch.allclose(converted["head.mlp.1.weight"], torch.full_like(model.head.mlp[1].weight, 6.0))
    assert torch.allclose(converted["head.mlp.3.bias"], torch.full_like(model.head.mlp[3].bias, 9.0))
    assert torch.allclose(converted["head.mlp.5.weight"], torch.full_like(model.head.mlp[5].weight, 5.0))
    assert torch.allclose(converted["head.mlp.5.bias"], torch.full_like(model.head.mlp[5].bias, 6.75))
