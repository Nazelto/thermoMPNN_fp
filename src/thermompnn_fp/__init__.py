from thermompnn_fp.head import ThermoMPNNHead
from .pipeline import build_mock_proteinmpnn_input, predict_single_mutation
from .types import HeadConfig, MutationSpec, input_dim


def main() -> None:
    config = HeadConfig()
    sequence_length = 50
    all_hidden, seq_embedding = build_mock_proteinmpnn_input(sequence_length, config)

    mutation = MutationSpec(mutation_position=10, wildtype="A", mutant="V")

    head = ThermoMPNNHead(config)

    result = predict_single_mutation(head, all_hidden, seq_embedding, mutation, config)

    print("feature input dim:", input_dim(config))
    print("aa_scores shape:", tuple(result.aa_scores.shape))
    print(
        "mutation:",
        f"{result.mutation.wildtype}{result.mutation.mutation_position}{result.mutation.mutant}",
    )
    print("predicted ddG:", float(result.ddg))
    print("Hello from thermompnn-fp!")
