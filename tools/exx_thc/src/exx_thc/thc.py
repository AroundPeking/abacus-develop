"""Deterministic complex CP-ALS prototype for occupied THC factors."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class THCFactors:
    T: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    residual: float
    iterations: int
    converged: bool


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("{} must be a positive integer".format(name))
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError("{} must be a positive integer".format(name)) from error
    if result < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return int(result)


def _nonnegative_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("{} must be finite and nonnegative".format(name)) from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("{} must be finite and nonnegative".format(name))
    return result


def _seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("seed must be a nonnegative integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError("seed must be a nonnegative integer") from error
    if result < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(result)


def cp_als(
    cbar: np.ndarray,
    rank: int,
    max_iter: int = 100,
    rel_tol: float = 1.0e-8,
    ridge: float = 1.0e-12,
    seed: int = 0,
) -> THCFactors:
    """Fit ``Cbar[m,i,v] = sum_r conj(T[m,r]) X[i,r] Y[v,r]``."""

    try:
        source = np.asarray(cbar, dtype=np.complex128, order="C")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Cbar must be convertible to a complex128 tensor") from error
    if source.ndim != 3 or any(extent == 0 for extent in source.shape):
        raise ValueError("cp_als expects a nonempty rank-3 tensor")
    if not np.isfinite(source).all():
        raise ValueError("Cbar must contain only finite values")
    fitted_rank = _positive_integer(rank, "rank")
    iteration_limit = _positive_integer(max_iter, "max_iter")
    tolerance = _nonnegative_float(rel_tol, "relative tolerance")
    regularization = _nonnegative_float(ridge, "ridge")
    random_seed = _seed(seed)

    naux, nao, nocc = source.shape
    with np.errstate(over="ignore", invalid="ignore"):
        scale = float(np.max(np.abs(source)))
    if not math.isfinite(scale):
        raise ValueError("Cbar global scale must be finite")
    if scale == 0.0:
        return THCFactors(
            np.zeros((naux, fitted_rank), dtype=np.complex128),
            np.zeros((nao, fitted_rank), dtype=np.complex128),
            np.zeros((nocc, fitted_rank), dtype=np.complex128),
            0.0,
            0,
            True,
        )

    a = source / scale
    norm = float(scipy.linalg.norm(a))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("scaled Cbar norm must be finite and positive")
    rng = np.random.default_rng(random_seed)

    def init_factor(unfold: np.ndarray, rows: int) -> np.ndarray:
        u, _, _ = scipy.linalg.svd(unfold, full_matrices=False, lapack_driver="gesdd")
        output = np.empty((rows, fitted_rank), dtype=np.complex128)
        copied = min(fitted_rank, u.shape[1])
        if copied:
            random_rotation = rng.normal(size=(copied, copied)) + 1j * rng.normal(
                size=(copied, copied)
            )
            unitary, _ = np.linalg.qr(random_rotation)
            output[:, :copied] = u[:, :copied] @ unitary
        if copied < fitted_rank:
            columns = fitted_rank - copied
            output[:, copied:] = (
                rng.normal(size=(rows, columns)) + 1j * rng.normal(size=(rows, columns))
            ) / math.sqrt(rows)
        return output

    factor_a = init_factor(a.reshape(naux, nao * nocc), naux)
    factor_x = init_factor(a.transpose(1, 0, 2).reshape(nao, naux * nocc), nao)
    factor_y = init_factor(a.transpose(2, 0, 1).reshape(nocc, naux * nao), nocc)

    def khatri_rao(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.einsum("ir,jr->ijr", left, right).reshape(
            left.shape[0] * right.shape[0], fitted_rank
        )

    def solve_factor(unfold: np.ndarray, kr: np.ndarray) -> np.ndarray:
        ridge_rows = math.sqrt(regularization) * np.eye(fitted_rank, dtype=np.complex128)
        lhs = np.vstack((kr, ridge_rows))
        rhs = np.vstack(
            (
                unfold.T,
                np.zeros((fitted_rank, unfold.shape[0]), dtype=np.complex128),
            )
        )
        solved = np.linalg.lstsq(lhs, rhs, rcond=None)[0].T
        if not np.isfinite(solved).all():
            raise ValueError("CP-ALS factor solve produced a non-finite value")
        return solved

    previous = math.inf
    stable = 0
    residual = math.inf
    converged = False
    iterations = iteration_limit
    for iteration in range(1, iteration_limit + 1):
        factor_a = solve_factor(a.reshape(naux, nao * nocc), khatri_rao(factor_x, factor_y))
        factor_x = solve_factor(
            a.transpose(1, 0, 2).reshape(nao, naux * nocc),
            khatri_rao(factor_a, factor_y),
        )
        factor_y = solve_factor(
            a.transpose(2, 0, 1).reshape(nocc, naux * nao),
            khatri_rao(factor_a, factor_x),
        )
        for column in range(fitted_rank):
            for factor in (factor_a, factor_x):
                column_norm = float(scipy.linalg.norm(factor[:, column]))
                if column_norm > 0.0:
                    factor[:, column] /= column_norm
                    factor_y[:, column] *= column_norm
            pivot_index = int(np.argmax(np.abs(factor_a[:, column])))
            pivot = factor_a[pivot_index, column]
            if abs(pivot) > 0.0:
                phase = pivot / abs(pivot)
                factor_a[:, column] /= phase
                factor_y[:, column] *= phase
        rebuilt = np.einsum("mr,ir,vr->miv", factor_a, factor_x, factor_y)
        residual = float(scipy.linalg.norm(a - rebuilt) / norm)
        if not math.isfinite(residual):
            raise ValueError("CP-ALS residual must be finite")
        stable = stable + 1 if abs(previous - residual) < tolerance else 0
        if stable >= 2:
            converged = True
            iterations = iteration
            break
        previous = residual

    output_t = np.ascontiguousarray(factor_a.conj(), dtype=np.complex128)
    output_x = np.ascontiguousarray(factor_x, dtype=np.complex128)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        output_y = np.ascontiguousarray(factor_y * scale, dtype=np.complex128)
    if not all(np.isfinite(factor).all() for factor in (output_t, output_x, output_y)):
        raise ValueError("CP-ALS output factors must contain only finite values")
    return THCFactors(output_t, output_x, output_y, residual, iterations, converged)
