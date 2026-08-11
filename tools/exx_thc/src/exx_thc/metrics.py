"""Full Coulomb-metric whitening and active/raw map diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class WhitenedTensor:
    tensor: np.ndarray
    transform: np.ndarray
    eigenvalues: np.ndarray
    removed_zero_modes: int
    hermiticity_rel: float


def _complex_finite_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.iscomplexobj(array):
        raise ValueError("{} must use a complex dtype".format(name))
    if not np.isfinite(array).all():
        raise ValueError("{} must contain only finite values".format(name))
    return np.asarray(array, dtype=np.complex128)


def _square_metric(value: np.ndarray, name: str) -> np.ndarray:
    metric = _complex_finite_array(value, name)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("{} must be square".format(name))
    if metric.shape[0] == 0:
        raise ValueError("{} must be nonempty".format(name))
    return metric


def _relative_hermiticity(metric: np.ndarray) -> float:
    scale = float(np.max(np.abs(metric)))
    if scale == 0.0:
        return 0.0
    scaled = metric / scale
    denominator = float(scipy.linalg.norm(scaled))
    numerator = float(scipy.linalg.norm(scaled - scaled.conj().T))
    result = numerator / denominator
    if not math.isfinite(result):
        raise ValueError("Coulomb metric Hermiticity residual must be finite")
    return result


def whiten(v: np.ndarray, cbar: np.ndarray) -> WhitenedTensor:
    """Whiten all auxiliary rows using one full Hermitian Coulomb matrix."""

    metric = _square_metric(v, "Coulomb metric")
    coefficients = _complex_finite_array(cbar, "Cbar")
    if coefficients.ndim != 3:
        raise ValueError("Cbar must be a rank-3 tensor")
    if coefficients.shape[0] != metric.shape[0]:
        raise ValueError("Cbar auxiliary dimension must match the Coulomb metric")

    hermiticity_rel = _relative_hermiticity(metric)
    with np.errstate(over="ignore", invalid="ignore"):
        hermitian_metric = 0.5 * metric + 0.5 * metric.conj().T
    if not np.isfinite(hermitian_metric).all():
        raise ValueError("Hermitized Coulomb metric must be finite")
    eigenvalues, eigenvectors = scipy.linalg.eigh(hermitian_metric, driver="evr")
    if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
        raise ValueError("Coulomb metric eigendecomposition must be finite")

    lambda_max = float(eigenvalues[-1])
    if lambda_max > 0.0 and float(eigenvalues[0]) < -1.0e-10 * lambda_max:
        raise ValueError("Coulomb metric is not positive semidefinite")
    if lambda_max <= 0.0 and np.any(eigenvalues < 0.0):
        raise ValueError("Coulomb metric is not positive semidefinite")

    clipped = np.clip(eigenvalues, 0.0, None)
    cutoff = 1.0e-12 * max(lambda_max, 0.0)
    keep = clipped > cutoff
    factor = eigenvectors[:, keep] * np.sqrt(clipped[keep])
    with np.errstate(over="ignore", invalid="ignore"):
        whitened = np.einsum("pm,miv->piv", factor.conj().T, coefficients)
    if not np.isfinite(whitened).all():
        raise ValueError("whitened tensor must contain only finite values")
    return WhitenedTensor(
        whitened,
        factor.conj().T,
        clipped,
        int(clipped.size - np.count_nonzero(keep)),
        hermiticity_rel,
    )


def metric_relative_frobenius(v_active: np.ndarray, v_raw: np.ndarray) -> float:
    """Return the full-matrix Frobenius difference relative to ``V.raw``."""

    active = _square_metric(v_active, "V.active")
    raw = _square_metric(v_raw, "V.raw")
    if active.shape != raw.shape:
        raise ValueError("V.active and V.raw must have the same shape")

    scale = max(float(np.max(np.abs(active))), float(np.max(np.abs(raw))))
    if scale == 0.0:
        return 0.0
    scaled_active = active / scale
    scaled_raw = raw / scale
    denominator = float(scipy.linalg.norm(scaled_raw))
    numerator = float(scipy.linalg.norm(scaled_active - scaled_raw))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    result = numerator / denominator
    if not math.isfinite(result):
        raise ValueError("relative Coulomb metric difference must be finite")
    return result
