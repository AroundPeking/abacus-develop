"""Stable occupied-space factors from Hermitian density matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class OccupiedFactor:
    O: np.ndarray
    O_pinv: np.ndarray
    eigenvalues: np.ndarray
    discarded_trace: float


def occupied_factor(d: np.ndarray, eigenvalue_tol: float) -> OccupiedFactor:
    """Factor a positive-semidefinite density and discard relative small modes."""

    density = np.asarray(d, dtype=np.complex128)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("density matrix must be square")
    if density.shape[0] == 0:
        raise ValueError("density matrix must be nonempty")
    if not np.all(np.isfinite(density)):
        raise ValueError("density matrix must contain only finite values")
    try:
        tolerance = float(eigenvalue_tol)
    except (TypeError, ValueError) as error:
        raise ValueError("eigenvalue tolerance must be finite and nonnegative") from error
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("eigenvalue tolerance must be finite and nonnegative")

    dh = 0.5 * (density + density.conj().T)
    w, u = scipy.linalg.eigh(dh, driver="evr")
    scale = max(float(np.max(np.abs(w))), 1.0)
    if np.min(w) < -1e-10 * scale:
        raise ValueError("density matrix is not positive semidefinite")
    w = np.clip(w, 0.0, None)
    spectral_scale = max(float(w.max()), 1.0)
    user_cutoff = tolerance * spectral_scale
    numerical_floor = np.finfo(w.dtype).eps * max(density.shape[0], 1) * spectral_scale
    keep = w > max(user_cutoff, numerical_floor)
    o = u[:, keep] * np.sqrt(w[keep])
    opinv = (u[:, keep] / np.sqrt(w[keep])).conj().T
    return OccupiedFactor(o, opinv, w, float(w[~keep].sum()))
