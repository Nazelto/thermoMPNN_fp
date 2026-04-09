from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _to_numpy(values: torch.Tensor | list[float]) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(float)
    return np.asarray(values, dtype=float)


def rmse(y_true: torch.Tensor | list[float], y_pred: torch.Tensor | list[float]) -> float:
    truth = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def pearson_correlation(
    y_true: torch.Tensor | list[float],
    y_pred: torch.Tensor | list[float],
) -> float:
    truth = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    if truth.size < 2:
        return 0.0
    if np.std(truth) == 0 or np.std(pred) == 0:
        return 0.0
    return float(np.corrcoef(truth, pred)[0, 1])


def spearman_correlation(
    y_true: torch.Tensor | list[float],
    y_pred: torch.Tensor | list[float],
) -> float:
    truth = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    if truth.size < 2:
        return 0.0
    truth_ranks = np.argsort(np.argsort(truth))
    pred_ranks = np.argsort(np.argsort(pred))
    return pearson_correlation(truth_ranks.tolist(), pred_ranks.tolist())


@dataclass
class LinearCalibration:
    slope: float
    intercept: float

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.slope + self.intercept


def fit_linear_calibration(
    raw_scores: torch.Tensor | list[float],
    target_ddg: torch.Tensor | list[float],
) -> LinearCalibration:
    raw = _to_numpy(raw_scores)
    target = _to_numpy(target_ddg)
    if raw.size == 0:
        return LinearCalibration(slope=1.0, intercept=0.0)
    coeffs = np.polyfit(raw, target, deg=1)
    return LinearCalibration(slope=float(coeffs[0]), intercept=float(coeffs[1]))
