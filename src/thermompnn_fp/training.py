from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import (
    FireProtDataset,
    MegaScaleDataset,
    ProteinMutationDataset,
    collate_protein_batches,
)
from .featurize import featurize_protein
from .metrics import pearson_correlation, rmse, spearman_correlation
from .pipeline import ThermoMPNNModel, load_model
from .types import DatasetItem, ProjectConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dataset_from_config(
    config: ProjectConfig, split_name: str
) -> ProteinMutationDataset:
    split_manifest = (
        Path(config.local_paths.splits_dir)
        / f"{config.training.dataset_name}_{split_name}.json"
    )
    if config.training.dataset_name == "megascale":
        return MegaScaleDataset.from_csv(
            config.local_paths.megascale_curated_csv,
            structure_root=config.local_paths.megascale_structures_root
            or config.local_paths.structures_root,
            split_manifest=split_manifest if split_manifest.exists() else None,
        )
    if config.training.dataset_name == "fireprot":
        return FireProtDataset.from_csv(
            config.local_paths.fireprot_curated_csv,
            structure_root=config.local_paths.fireprot_structures_root
            or config.local_paths.structures_root,
            split_manifest=split_manifest if split_manifest.exists() else None,
        )
    raise ValueError(f"Unsupported dataset: {config.training.dataset_name}")


def _make_loader(
    dataset: ProteinMutationDataset, batch_size: int, num_workers: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_protein_batches,
    )


def _batch_loss(
    model: ThermoMPNNModel,
    batch: list[DatasetItem],
    config: ProjectConfig,
    device: torch.device,
) -> torch.Tensor:
    protein_losses: list[torch.Tensor] = []
    for item in batch:
        valid_mutations = [
            mutation for mutation in item.mutations if mutation.ddg is not None
        ]
        if not valid_mutations:
            continue
        backbone_input = featurize_protein(
            item.protein,
            num_neighbors=config.model.num_neighbors,
            rbf_bins=config.model.rbf_bins,
            distance_min=config.model.rbf_distance_min,
            distance_max=config.model.rbf_distance_max,
            device=device,
        )
        prediction_batch = model(backbone_input, valid_mutations)
        predicted = torch.stack([result.ddg for result in prediction_batch.predictions])
        target = torch.tensor(
            [float(mutation.ddg) for mutation in valid_mutations],
            dtype=predicted.dtype,
            device=device,
        )
        protein_losses.append(mse_loss(predicted, target))
    if not protein_losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(protein_losses).mean()


def evaluate_model(
    model: ThermoMPNNModel,
    dataset: ProteinMutationDataset,
    config: ProjectConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    truth: list[float] = []
    preds: list[float] = []
    with torch.no_grad():
        for item in dataset:
            valid_mutations = [
                mutation for mutation in item.mutations if mutation.ddg is not None
            ]
            if not valid_mutations:
                continue
            backbone_input = featurize_protein(
                item.protein,
                num_neighbors=config.model.num_neighbors,
                rbf_bins=config.model.rbf_bins,
                distance_min=config.model.rbf_distance_min,
                distance_max=config.model.rbf_distance_max,
                device=device,
            )
            prediction_batch = model(backbone_input, valid_mutations)
            preds.extend(
                float(result.ddg.detach().cpu())
                for result in prediction_batch.predictions
            )
            truth.extend(
                float(mutation.ddg)
                for mutation in valid_mutations
                if mutation.ddg is not None
            )
    return {
        "pcc": pearson_correlation(truth, preds),
        "scc": spearman_correlation(truth, preds),
        "rmse": rmse(truth, preds) if truth else float("nan"),
    }


def train_model(config: ProjectConfig) -> dict[str, float]:
    set_seed(config.training.seed)
    device = torch.device(config.training.device)
    train_dataset = _dataset_from_config(config, "train")
    val_dataset = _dataset_from_config(config, "val")
    train_loader = _make_loader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    model = ThermoMPNNModel(
        config.model,
        checkpoint_path=config.local_paths.proteinmpnn_checkpoint or None,
    ).to(device)

    head_parameters = list(model.head.parameters())
    parameter_groups = [
        {"params": head_parameters, "lr": config.training.learning_rate}
    ]
    if not config.model.freeze_backbone:
        parameter_groups.append(
            {
                "params": model.backbone.parameters(),
                "lr": config.training.mpnn_learning_rate
                or config.training.learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=config.training.weight_decay
    )

    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_metric = float("-inf")
    best_metrics: dict[str, float] = {}
    best_path = checkpoint_dir / f"{config.name}_best.pt"

    for epoch in range(config.training.epochs):
        model.train()
        epoch_losses: list[float] = []
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{config.training.epochs}",
            leave=False,
        )
        for batch in progress:
            optimizer.zero_grad()
            loss = _batch_loss(model, batch, config, device)
            loss.backward()
            if config.training.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.clip_grad_norm
                )
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            progress.set_postfix(loss=float(loss.detach().cpu()))

        metrics = evaluate_model(model, val_dataset, config, device)
        monitored = metrics[config.training.validation_metric]
        if config.training.validation_metric == "rmse":
            improved = best_metric == float("-inf") or monitored < best_metric
            if improved:
                best_metric = monitored
        else:
            improved = monitored > best_metric
            if improved:
                best_metric = monitored

        if improved:
            best_metrics = metrics | {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(epoch_losses or [0.0])),
            }
            torch.save(
                {
                    "config": config,
                    "model_state_dict": model.state_dict(),
                    "metrics": best_metrics,
                },
                best_path,
            )

    metrics_path = checkpoint_dir / f"{config.name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(best_metrics, handle, indent=2)
    return best_metrics


def load_trained_model(
    config: ProjectConfig, checkpoint_path: str | Path
) -> ThermoMPNNModel:
    return load_model(
        config.model,
        checkpoint_path=str(checkpoint_path),
        model_weights_path=config.local_paths.proteinmpnn_checkpoint or None,
        device=config.training.device,
    )
