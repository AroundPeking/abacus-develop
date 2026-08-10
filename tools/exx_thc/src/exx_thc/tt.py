"""Tensor-train diagnostics for three-dimensional complex tensors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class TT3:
    g1: np.ndarray
    g2: np.ndarray
    g3: np.ndarray
    spectra: Tuple[np.ndarray, np.ndarray]
    discarded_weights: Tuple[float, float]
    error_bound: float
    ranks: Tuple[int, int]

    def reconstruct(self) -> np.ndarray:
        """Reconstruct the dense rank-three tensor represented by the cores."""

        return np.einsum("ia,ajb,bk->ijk", self.g1, self.g2, self.g3)


def _validated_tolerance(value: float) -> float:
    try:
        tolerance = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("relative tolerance must be finite and nonnegative") from error
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("relative tolerance must be finite and nonnegative")
    return tolerance


def _choose_rank(singular_values: np.ndarray, tail_tolerance: float) -> int:
    for rank in range(1, singular_values.size + 1):
        tail_norm = float(scipy.linalg.norm(singular_values[rank:]))
        if tail_norm <= tail_tolerance:
            return rank
    return int(singular_values.size)


def tt_svd_3(a: np.ndarray, relative_tol: float) -> TT3:
    """Compute a two-cut TT-SVD with an equal error budget at each cut."""

    input_tensor = np.asarray(a)
    if input_tensor.ndim != 3:
        raise ValueError("tt_svd_3 expects a rank-3 tensor")
    if any(extent == 0 for extent in input_tensor.shape):
        raise ValueError("tt_svd_3 expects a nonempty tensor")
    try:
        finite = bool(np.isfinite(input_tensor).all())
    except TypeError as error:
        raise ValueError("tt_svd_3 expects numeric finite values") from error
    if not finite:
        raise ValueError("tt_svd_3 expects finite values")
    working_dtype = np.complex128 if np.iscomplexobj(input_tensor) else np.float64
    tensor = np.asarray(input_tensor, dtype=working_dtype)
    if not np.isfinite(tensor).all():
        raise ValueError("promoted tensor must contain only finite values")
    tolerance = _validated_tolerance(relative_tol)
    n1, n2, n3 = tensor.shape
    with np.errstate(over="ignore", invalid="ignore"):
        max_abs = float(np.max(np.abs(tensor)))
    if not np.isfinite(max_abs):
        raise ValueError("tensor Frobenius norm squared must be finite")
    if max_abs == 0.0:
        return TT3(
            np.zeros((n1, 0), dtype=tensor.dtype),
            np.zeros((0, n2, 0), dtype=tensor.dtype),
            np.zeros((0, n3), dtype=tensor.dtype),
            (np.zeros(0), np.zeros(0)),
            (0.0, 0.0),
            0.0,
            (0, 0),
        )

    tensor_scaled = tensor / max_abs
    norm_scaled = float(scipy.linalg.norm(tensor_scaled))
    if not np.isfinite(norm_scaled) or max_abs > math.sqrt(np.finfo(np.float64).max) / norm_scaled:
        raise ValueError("tensor Frobenius norm squared must be finite")
    norm_magnitude = max_abs * norm_scaled
    with np.errstate(over="ignore", invalid="ignore"):
        norm_squared = float(np.multiply(norm_magnitude, norm_magnitude))
    if not np.isfinite(norm_squared):
        raise ValueError("tensor Frobenius norm squared must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        tail_tolerance = float(np.multiply(tolerance, norm_scaled / math.sqrt(2.0)))
    u1, s1, vh1 = scipy.linalg.svd(
        tensor_scaled.reshape(n1, n2 * n3), full_matrices=False, lapack_driver="gesdd"
    )
    r1 = _choose_rank(s1, tail_tolerance)
    g1 = u1[:, :r1]
    remainder = (s1[:r1, None] * vh1[:r1]).reshape(r1 * n2, n3)
    u2, s2, vh2 = scipy.linalg.svd(remainder, full_matrices=False, lapack_driver="gesdd")
    r2 = _choose_rank(s2, tail_tolerance)
    g2 = u2[:, :r2].reshape(r1, n2, r2)
    g3 = (s2[:r2] * max_abs)[:, None] * vh2[:r2]

    tail1_norm = float(scipy.linalg.norm(s1[r1:]))
    tail2_norm = float(scipy.linalg.norm(s2[r2:]))
    def normalized_squared_weight(tail_norm: float) -> float:
        weight = (tail_norm / norm_scaled) ** 2
        if tail_norm > 0.0 and weight == 0.0:
            return float(np.nextafter(0.0, 1.0))
        return weight

    weight1 = normalized_squared_weight(tail1_norm)
    weight2 = normalized_squared_weight(tail2_norm)
    with np.errstate(over="ignore", invalid="ignore"):
        error_bound = float(np.multiply(norm_squared, weight1 + weight2))
    if not np.isfinite(error_bound):
        raise ValueError("TT error bound must be finite")
    if (tail1_norm > 0.0 or tail2_norm > 0.0) and error_bound == 0.0:
        error_bound = float(np.nextafter(0.0, 1.0))
    spectra = (s1 * max_abs, s2 * max_abs)
    for value in (g1, g2, g3, *spectra):
        if not np.isfinite(value).all():
            raise ValueError("TT cores and spectra must contain only finite values")
    return TT3(
        g1,
        g2,
        g3,
        spectra,
        (weight1, weight2),
        error_bound,
        (r1, r2),
    )
