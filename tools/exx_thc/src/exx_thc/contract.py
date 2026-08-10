"""Dense and algebraic occupied-THC exchange contractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class ExchangeDiagnostics:
    matrix: np.ndarray
    hermiticity_rel: float
    temporary_shapes: Dict[str, Tuple[int, ...]]


def _complex_array(value: np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.complex128, order="C")
    except (TypeError, ValueError) as error:
        raise ValueError("{} must be numeric".format(name)) from error
    if not np.isfinite(array).all():
        raise ValueError("{} must contain only finite values".format(name))
    return array


def _finite_result(value: np.ndarray, name: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.complex128)
    if not np.isfinite(result).all():
        raise ValueError("{} contains a non-finite value".format(name))
    return result


def exchange_hermiticity_rel(matrix: np.ndarray) -> float:
    """Return ``||K-K^dagger||_F / ||K||_F`` without modifying ``K``."""

    value = _complex_array(matrix, "exchange matrix")
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("exchange matrix must be square")
    scale = (
        max(float(np.max(np.abs(value.real))), float(np.max(np.abs(value.imag))))
        if value.size
        else 0.0
    )
    if scale == 0.0:
        return 0.0
    scaled = value / scale
    denominator = float(scipy.linalg.norm(scaled))
    numerator = float(scipy.linalg.norm(scaled - scaled.conj().T))
    if not np.isfinite(denominator) or denominator == 0.0 or not np.isfinite(numerator):
        raise ValueError("exchange Hermiticity norm must be finite and nonzero")
    result = numerator / denominator
    if not np.isfinite(result):
        raise ValueError("exchange Hermiticity residual must be finite")
    return result


def dense_exchange(
    cbar: np.ndarray, v: np.ndarray, diagnostics: bool = False
) -> Union[np.ndarray, ExchangeDiagnostics]:
    """Contract the dense whitened/occupied coefficient tensor."""

    coefficients = _complex_array(cbar, "Cbar")
    metric = _complex_array(v, "Coulomb metric")
    if coefficients.ndim != 3 or any(extent == 0 for extent in coefficients.shape):
        raise ValueError("Cbar must be a nonempty rank-3 tensor")
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("Coulomb metric must be square")
    if metric.shape[0] != coefficients.shape[0]:
        raise ValueError("Cbar and Coulomb metric auxiliary dimensions must match")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        matrix = -np.einsum(
            "miv,mn,nkv->ik", coefficients, metric, coefficients.conj(), optimize=True
        )
    matrix = _finite_result(matrix, "dense exchange")
    if diagnostics:
        return ExchangeDiagnostics(matrix, exchange_hermiticity_rel(matrix), {})
    return matrix


def thc_exchange(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    diagnostics: bool = False,
) -> Union[np.ndarray, ExchangeDiagnostics]:
    """Contract THC factors without reconstructing the dense coefficient tensor."""

    auxiliary = _complex_array(t, "T")
    orbitals = _complex_array(x, "X")
    occupied = _complex_array(y, "Y")
    metric = _complex_array(v, "Coulomb metric")
    if any(factor.ndim != 2 for factor in (auxiliary, orbitals, occupied)):
        raise ValueError("T, X, and Y must be matrices")
    if any(extent == 0 for factor in (auxiliary, orbitals, occupied) for extent in factor.shape):
        raise ValueError("T, X, and Y must be nonempty")
    rank = auxiliary.shape[1]
    if orbitals.shape[1] != rank or occupied.shape[1] != rank:
        raise ValueError("T, X, and Y must have the same column rank")
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise ValueError("Coulomb metric must be square")
    if metric.shape[0] != auxiliary.shape[0]:
        raise ValueError("T and Coulomb metric auxiliary dimensions must match")

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        vt = metric @ auxiliary
    vt = _finite_result(vt, "V@T")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        z = auxiliary.conj().T @ vt
        s = occupied.T @ occupied.conj()
    z = _finite_result(z, "T^dagger@V@T")
    s = _finite_result(s, "occupied overlap")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        core = z * s
        xcore = orbitals @ core
        matrix = -(xcore @ orbitals.conj().T)
    core = _finite_result(core, "THC core")
    xcore = _finite_result(xcore, "X@core")
    matrix = _finite_result(matrix, "THC exchange")

    if diagnostics:
        shapes = {
            "vt": vt.shape,
            "z": z.shape,
            "s": s.shape,
            "core": core.shape,
            "xcore": xcore.shape,
            "K": matrix.shape,
        }
        return ExchangeDiagnostics(matrix, exchange_hermiticity_rel(matrix), shapes)
    return matrix
