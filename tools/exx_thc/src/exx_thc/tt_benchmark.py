"""Accuracy and timing scans for EXX tensor-train representations."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import operator
import statistics
import time
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np

from .metrics import whiten
from .supercell import occupied_exchange
from .tt import TT3, tt_core_elements, tt_gram, tt_mode_transform, tt_svd_3


Order = Tuple[int, int, int]


@dataclass(frozen=True)
class TTBenchmarkPoint:
    route: str
    order: Order
    relative_tol: float
    canonical_shape: Tuple[int, int, int]
    ranks: Tuple[int, int]
    spectra: Tuple[np.ndarray, np.ndarray]
    discarded_weights: Tuple[float, float]
    core_shapes_elements: Tuple[int, int, int]
    dense_elements: int
    core_elements: int
    compression_ratio: float
    reconstruction_rel_fro: float
    h_rel_fro: float
    setup_seconds: float
    exx_seconds: float
    steady_seconds: float
    total_seconds: float
    samples: int
    matrix: np.ndarray

    def to_json_dict(self) -> dict:
        """Return finite JSON metadata, excluding the dense validation matrix."""

        return {
            "route": self.route,
            "order": list(self.order),
            "relative_tol": self.relative_tol,
            "canonical_shape": list(self.canonical_shape),
            "ranks": list(self.ranks),
            "spectra": [values.tolist() for values in self.spectra],
            "discarded_weights": list(self.discarded_weights),
            "core_shapes_elements": list(self.core_shapes_elements),
            "dense_elements": self.dense_elements,
            "core_elements": self.core_elements,
            "compression_ratio": self.compression_ratio,
            "reconstruction_rel_fro": self.reconstruction_rel_fro,
            "H_exact_rel_fro": self.h_rel_fro,
            "setup_seconds": self.setup_seconds,
            "exx_seconds": self.exx_seconds,
            "steady_seconds": self.steady_seconds,
            "total_seconds": self.total_seconds,
            "samples": self.samples,
        }


@dataclass(frozen=True)
class TTBenchmarkResult:
    points: Tuple[TTBenchmarkPoint, ...]
    metric_setup_seconds: float
    dense_occupied_seconds: float
    selected: Optional[TTBenchmarkPoint]

    def to_json_dict(self) -> dict:
        return {
            "metric_setup_seconds": self.metric_setup_seconds,
            "dense_occupied_seconds": self.dense_occupied_seconds,
            "points": [point.to_json_dict() for point in self.points],
            "selected": None if self.selected is None else self.selected.to_json_dict(),
        }


def _complex_finite(value: object, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.complex128, order="C")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("{} must be convertible to complex128".format(name)) from error
    if not np.isfinite(array).all():
        raise ValueError("{} must contain only finite values".format(name))
    return array


def _validated_inputs(
    coefficient: object, metric: object, occupied: object
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = _complex_finite(coefficient, "C")
    V = _complex_finite(metric, "V")
    O = _complex_finite(occupied, "O")
    if C.ndim != 3 or C.shape[0] == 0 or C.shape[1] == 0:
        raise ValueError("C must have nonempty shape (naux, nao, nao)")
    if C.shape[1] != C.shape[2]:
        raise ValueError("C must have nonempty shape (naux, nao, nao)")
    if V.ndim != 2 or V.shape != (C.shape[0], C.shape[0]):
        raise ValueError("V must have shape (naux, naux)")
    if O.ndim != 2 or O.shape[0] != C.shape[1] or O.shape[1] == 0:
        raise ValueError("O must have nonempty shape (nao, rank)")
    return C, V, O


def _validated_tolerances(values: Iterable[float]) -> Tuple[float, ...]:
    try:
        tolerances = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError("TT tolerances must be finite and nonnegative") from error
    if not tolerances or any(
        not math.isfinite(value) or value < 0.0 for value in tolerances
    ):
        raise ValueError("TT tolerances must be finite and nonnegative")
    return tolerances


def _validated_repeats(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("benchmark repeats must be a positive integer")
    try:
        repeats = operator.index(value)
    except TypeError as error:
        raise ValueError("benchmark repeats must be a positive integer") from error
    if repeats <= 0:
        raise ValueError("benchmark repeats must be a positive integer")
    return int(repeats)


def _relative_frobenius(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.shape != candidate.shape:
        raise ValueError("relative Frobenius arrays must have identical shapes")
    scale = max(float(np.max(np.abs(reference))), float(np.max(np.abs(candidate))))
    if not math.isfinite(scale):
        raise ValueError("relative Frobenius arrays must be finite")
    if scale == 0.0:
        return 0.0
    reference_scaled = reference / scale
    candidate_scaled = candidate / scale
    denominator = float(np.linalg.norm(reference_scaled))
    numerator = float(np.linalg.norm(candidate_scaled - reference_scaled))
    if denominator == 0.0:
        result = 0.0 if numerator == 0.0 else np.finfo(np.float64).max
    else:
        result = numerator / denominator
    if not math.isfinite(result):
        raise ValueError("relative Frobenius result must be finite")
    return result


def _timed(
    operation: Callable[[], object], repeats: int
) -> Tuple[float, object]:
    operation()
    samples = []
    result: object = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = operation()
        elapsed = (time.perf_counter_ns() - start) * 1.0e-9
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("benchmark timer produced an invalid duration")
        samples.append(elapsed)
    return float(statistics.median(samples)), result


def _canonical_reconstruction(tt: TT3, order: Order) -> np.ndarray:
    inverse = tuple(int(axis) for axis in np.argsort(order))
    return np.transpose(tt.reconstruct(), inverse)


def _route_matrix(
    route: str,
    tt: TT3,
    order: Order,
    occupied: np.ndarray,
    metric_transform: np.ndarray,
) -> np.ndarray:
    working = tt
    if route == "C":
        working = tt_mode_transform(working, order.index(2), occupied.T)
        working = tt_mode_transform(working, order.index(0), metric_transform)
    elif route == "B":
        working = tt_mode_transform(working, order.index(0), metric_transform)
    elif route != "X":
        raise ValueError("unknown TT benchmark route")
    return tt_gram(working, order.index(1))


def select_tt_point(
    points: Sequence[TTBenchmarkPoint], h_rel_tol: float = 1.0e-8
) -> Optional[TTBenchmarkPoint]:
    """Select by accuracy, time, 10% storage tie-break, then static C."""

    tolerance = float(h_rel_tol)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("H relative tolerance must be finite and nonnegative")
    eligible = [point for point in points if point.h_rel_fro <= tolerance]
    if not eligible:
        return None
    fastest = min(point.steady_seconds for point in eligible)
    close = [point for point in eligible if point.steady_seconds <= 1.1 * fastest]
    return min(
        close,
        key=lambda point: (
            -point.compression_ratio,
            point.route != "C",
            point.steady_seconds,
            point.route,
            point.order,
            point.relative_tol,
        ),
    )


def scan_exx_tt_routes(
    coefficient: object,
    metric: object,
    occupied: object,
    tolerances: Iterable[float],
    repeats: int = 5,
) -> TTBenchmarkResult:
    """Scan C, occupied B, and EXX-oriented metric-factor X TT routes."""

    C, V, O = _validated_inputs(coefficient, metric, occupied)
    tolerance_values = _validated_tolerances(tolerances)
    sample_count = _validated_repeats(repeats)
    dense_seconds, exact_value = _timed(
        lambda: occupied_exchange(C, V, O).matrix, sample_count
    )
    exact_h = np.asarray(exact_value)

    metric_probe = np.zeros((C.shape[0], 1, 1), dtype=np.complex128)
    metric_seconds, whitened_probe = _timed(
        lambda: whiten(V, metric_probe), sample_count
    )
    metric_transform = whitened_probe.transform.conj()
    if metric_transform.shape[0] == 0:
        raise ValueError("V must retain at least one active auxiliary mode")

    with np.errstate(over="ignore", invalid="ignore"):
        B_reference = np.einsum("mij,jv->miv", C, O, optimize=True)
        X_reference = np.einsum(
            "xm,miv->xiv", metric_transform, B_reference, optimize=True
        )
    if not np.isfinite(B_reference).all() or not np.isfinite(X_reference).all():
        raise ValueError("EXX route construction produced a non-finite tensor")

    route_tensors = {"C": C, "B": B_reference, "X": X_reference}
    points = []
    for route in ("C", "B", "X"):
        canonical = route_tensors[route]
        for order in itertools.permutations((0, 1, 2)):
            order = tuple(int(axis) for axis in order)
            for tolerance in tolerance_values:

                def setup() -> TT3:
                    if route == "C":
                        dense = C
                    else:
                        dense_b = np.einsum("mij,jv->miv", C, O, optimize=True)
                        if route == "B":
                            dense = dense_b
                        else:
                            dense = np.einsum(
                                "xm,miv->xiv",
                                metric_transform,
                                dense_b,
                                optimize=True,
                            )
                    if not np.isfinite(dense).all():
                        raise ValueError("EXX route construction produced a non-finite tensor")
                    return tt_svd_3(np.transpose(dense, order), tolerance)

                setup_seconds, tt_value = _timed(setup, sample_count)
                tt = tt_value
                exx_seconds, matrix_value = _timed(
                    lambda: _route_matrix(
                        route, tt, order, O, metric_transform
                    ),
                    sample_count,
                )
                matrix = np.asarray(matrix_value)
                reconstruction = _canonical_reconstruction(tt, order)
                core_shapes_elements = (
                    int(tt.g1.size),
                    int(tt.g2.size),
                    int(tt.g3.size),
                )
                core_elements = tt_core_elements(tt)
                dense_elements = int(canonical.size)
                compression = dense_elements / max(core_elements, 1)
                steady = exx_seconds if route == "C" else setup_seconds + exx_seconds
                point = TTBenchmarkPoint(
                    route=route,
                    order=order,
                    relative_tol=tolerance,
                    canonical_shape=tuple(int(value) for value in canonical.shape),
                    ranks=tt.ranks,
                    spectra=tt.spectra,
                    discarded_weights=tt.discarded_weights,
                    core_shapes_elements=core_shapes_elements,
                    dense_elements=dense_elements,
                    core_elements=core_elements,
                    compression_ratio=float(compression),
                    reconstruction_rel_fro=_relative_frobenius(
                        canonical, reconstruction
                    ),
                    h_rel_fro=_relative_frobenius(exact_h, matrix),
                    setup_seconds=setup_seconds,
                    exx_seconds=exx_seconds,
                    steady_seconds=steady,
                    total_seconds=setup_seconds + exx_seconds,
                    samples=sample_count,
                    matrix=matrix,
                )
                points.append(point)
    point_tuple = tuple(points)
    return TTBenchmarkResult(
        points=point_tuple,
        metric_setup_seconds=metric_seconds,
        dense_occupied_seconds=dense_seconds,
        selected=select_tt_point(point_tuple),
    )
