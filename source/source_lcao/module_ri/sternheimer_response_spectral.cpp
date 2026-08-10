#include "sternheimer_response_spectral.h"

#include "source_base/module_external/lapack_connector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace module_ri
{
namespace sternheimer_siab
{
namespace
{

using Complex = std::complex<double>;
using Matrix = std::vector<Complex>;

struct HermitianEigensystem
{
    std::vector<double> eigenvalues;
    Matrix eigenvectors;
};

struct MetricWhitener
{
    Matrix transform;
    int rank = 0;
    int dropped = 0;
    double condition = 0.0;
};

struct GeneralizedEigensystem
{
    std::vector<double> eigenvalues;
    Matrix coefficients;
};

std::size_t matrix_size(const int rows, const int columns, const char* label)
{
    if (rows <= 0 || columns <= 0
        || static_cast<std::size_t>(rows) > std::numeric_limits<std::size_t>::max()
                                                   / static_cast<std::size_t>(columns))
    {
        throw std::invalid_argument(std::string(label) + " has invalid dimensions.");
    }
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
}

void require_shape(const Matrix& matrix,
                   const int rows,
                   const int columns,
                   const char* label)
{
    if (matrix.size() != matrix_size(rows, columns, label))
    {
        throw std::invalid_argument(std::string(label) + " has inconsistent dimensions.");
    }
}

Matrix adjoint(const Matrix& matrix, const int rows, const int columns)
{
    require_shape(matrix, rows, columns, "Sternheimer response matrix");
    Matrix result(matrix_size(columns, rows, "Sternheimer response adjoint"));
    for (int row = 0; row != rows; ++row)
    {
        for (int column = 0; column != columns; ++column)
        {
            result[static_cast<std::size_t>(column * rows + row)]
                = std::conj(matrix[static_cast<std::size_t>(row * columns + column)]);
        }
    }
    return result;
}

Matrix multiply(const Matrix& left,
                const int left_rows,
                const int inner,
                const Matrix& right,
                const int right_columns)
{
    require_shape(left, left_rows, inner, "Sternheimer response left matrix");
    require_shape(right, inner, right_columns, "Sternheimer response right matrix");
    Matrix result(matrix_size(left_rows, right_columns, "Sternheimer response product"),
                  Complex(0.0, 0.0));
    for (int row = 0; row != left_rows; ++row)
    {
        for (int k = 0; k != inner; ++k)
        {
            const Complex value = left[static_cast<std::size_t>(row * inner + k)];
            for (int column = 0; column != right_columns; ++column)
            {
                result[static_cast<std::size_t>(row * right_columns + column)]
                    += value * right[static_cast<std::size_t>(k * right_columns + column)];
            }
        }
    }
    return result;
}

Matrix sandwich(const Matrix& basis,
                const int dimension,
                const int rank,
                const Matrix& matrix)
{
    const Matrix matrix_basis = multiply(matrix, dimension, dimension, basis, rank);
    return multiply(adjoint(basis, dimension, rank), rank, dimension, matrix_basis, rank);
}

Matrix hermitize(const Matrix& matrix, const int dimension)
{
    require_shape(matrix, dimension, dimension, "Sternheimer response Hermitian matrix");
    Matrix result(matrix);
    for (int row = 0; row != dimension; ++row)
    {
        result[static_cast<std::size_t>(row * dimension + row)]
            = Complex(result[static_cast<std::size_t>(row * dimension + row)].real(), 0.0);
        for (int column = row + 1; column != dimension; ++column)
        {
            const Complex value
                = 0.5 * (matrix[static_cast<std::size_t>(row * dimension + column)]
                         + std::conj(matrix[static_cast<std::size_t>(column * dimension + row)]));
            result[static_cast<std::size_t>(row * dimension + column)] = value;
            result[static_cast<std::size_t>(column * dimension + row)] = std::conj(value);
        }
    }
    return result;
}

HermitianEigensystem diagonalize_hermitian(const Matrix& input, const int dimension)
{
    const Matrix symmetric = hermitize(input, dimension);
    Matrix column_major(matrix_size(dimension, dimension, "Sternheimer response eigensolver"));
    for (int row = 0; row != dimension; ++row)
    {
        for (int column = 0; column != dimension; ++column)
        {
            column_major[static_cast<std::size_t>(column * dimension + row)]
                = symmetric[static_cast<std::size_t>(row * dimension + column)];
        }
    }

    const char jobz = 'V';
    const char uplo = 'U';
    const int lda = dimension;
    int info = 0;
    int lwork = -1;
    Complex work_query(0.0, 0.0);
    std::vector<double> eigenvalues(static_cast<std::size_t>(dimension), 0.0);
    std::vector<double> rwork(static_cast<std::size_t>(std::max(1, 3 * dimension - 2)), 0.0);
    zheev_(&jobz,
           &uplo,
           &dimension,
           column_major.data(),
           &lda,
           eigenvalues.data(),
           &work_query,
           &lwork,
           rwork.data(),
           &info);
    if (info != 0 || !std::isfinite(work_query.real()))
    {
        throw std::runtime_error("Sternheimer response eigensolver workspace query failed.");
    }
    lwork = std::max(1, static_cast<int>(std::ceil(work_query.real())));
    std::vector<Complex> work(static_cast<std::size_t>(lwork));
    zheev_(&jobz,
           &uplo,
           &dimension,
           column_major.data(),
           &lda,
           eigenvalues.data(),
           work.data(),
           &lwork,
           rwork.data(),
           &info);
    if (info != 0)
    {
        throw std::runtime_error("Sternheimer response eigensolver failed with info="
                                 + std::to_string(info) + ".");
    }

    Matrix eigenvectors(matrix_size(dimension, dimension, "Sternheimer response eigenvectors"));
    for (int row = 0; row != dimension; ++row)
    {
        for (int column = 0; column != dimension; ++column)
        {
            eigenvectors[static_cast<std::size_t>(row * dimension + column)]
                = column_major[static_cast<std::size_t>(column * dimension + row)];
        }
    }
    return HermitianEigensystem{std::move(eigenvalues), std::move(eigenvectors)};
}

MetricWhitener metric_whitener(const Matrix& metric,
                               const int dimension,
                               const double relative_tolerance,
                               const double condition_limit,
                               const bool allow_dropped)
{
    const HermitianEigensystem eigensystem = diagonalize_hermitian(metric, dimension);
    const double largest = eigensystem.eigenvalues.back();
    if (!std::isfinite(largest) || largest <= 0.0)
    {
        throw std::runtime_error("Sternheimer response metric has no positive direction.");
    }
    const double cutoff = relative_tolerance * largest;
    if (eigensystem.eigenvalues.front() < -cutoff)
    {
        throw std::runtime_error("Sternheimer response metric has a materially negative eigenvalue.");
    }

    int first = 0;
    while (first != dimension && eigensystem.eigenvalues[static_cast<std::size_t>(first)] <= cutoff)
    {
        ++first;
    }
    if (first == dimension)
    {
        throw std::runtime_error("Sternheimer response metric has no retained direction.");
    }
    if (!allow_dropped && first != 0)
    {
        throw std::runtime_error("Sternheimer fixed-AO metric is rank deficient.");
    }

    MetricWhitener result;
    result.rank = dimension - first;
    result.dropped = first;
    result.condition = largest / eigensystem.eigenvalues[static_cast<std::size_t>(first)];
    if (!std::isfinite(result.condition) || result.condition > condition_limit)
    {
        throw std::runtime_error("Sternheimer response metric condition number exceeds its limit.");
    }
    result.transform.assign(matrix_size(dimension, result.rank, "Sternheimer response whitener"),
                            Complex(0.0, 0.0));
    for (int retained = 0; retained != result.rank; ++retained)
    {
        const int eigenvector = first + retained;
        const double scale
            = 1.0 / std::sqrt(eigensystem.eigenvalues[static_cast<std::size_t>(eigenvector)]);
        for (int row = 0; row != dimension; ++row)
        {
            result.transform[static_cast<std::size_t>(row * result.rank + retained)]
                = eigensystem.eigenvectors[static_cast<std::size_t>(row * dimension + eigenvector)] * scale;
        }
    }
    return result;
}

GeneralizedEigensystem generalized_eigensystem(const Matrix& overlap,
                                               const Matrix& hamiltonian,
                                               const int dimension,
                                               const ResponseSpectralOptions& options)
{
    const MetricWhitener metric = metric_whitener(overlap,
                                                   dimension,
                                                   options.relative_rank_tolerance,
                                                   options.condition_limit,
                                                   false);
    const Matrix transformed_hamiltonian
        = sandwich(metric.transform, dimension, dimension, hamiltonian);
    const HermitianEigensystem eigensystem
        = diagonalize_hermitian(transformed_hamiltonian, dimension);
    return GeneralizedEigensystem{
        eigensystem.eigenvalues,
        multiply(metric.transform, dimension, dimension, eigensystem.eigenvectors, dimension)};
}

Matrix assemble_hermitian_blocks(const Matrix& fixed,
                                 const int fixed_dimension,
                                 const Matrix& response,
                                 const int response_dimension,
                                 const Matrix& response_fixed)
{
    require_shape(fixed, fixed_dimension, fixed_dimension, "Sternheimer fixed block");
    require_shape(response, response_dimension, response_dimension, "Sternheimer response block");
    require_shape(response_fixed,
                  response_dimension,
                  fixed_dimension,
                  "Sternheimer response-fixed block");
    const int dimension = fixed_dimension + response_dimension;
    Matrix result(matrix_size(dimension, dimension, "Sternheimer response union"), Complex(0.0, 0.0));
    for (int row = 0; row != fixed_dimension; ++row)
    {
        for (int column = 0; column != fixed_dimension; ++column)
        {
            result[static_cast<std::size_t>(row * dimension + column)]
                = fixed[static_cast<std::size_t>(row * fixed_dimension + column)];
        }
    }
    for (int row = 0; row != response_dimension; ++row)
    {
        for (int column = 0; column != response_dimension; ++column)
        {
            result[static_cast<std::size_t>((fixed_dimension + row) * dimension
                                            + fixed_dimension + column)]
                = response[static_cast<std::size_t>(row * response_dimension + column)];
        }
        for (int column = 0; column != fixed_dimension; ++column)
        {
            const Complex value
                = response_fixed[static_cast<std::size_t>(row * fixed_dimension + column)];
            result[static_cast<std::size_t>((fixed_dimension + row) * dimension + column)] = value;
            result[static_cast<std::size_t>(column * dimension + fixed_dimension + row)]
                = std::conj(value);
        }
    }
    return hermitize(result, dimension);
}

double max_abs_difference(const std::vector<double>& left, const std::vector<double>& right)
{
    if (left.size() != right.size())
    {
        throw std::invalid_argument("Sternheimer response eigenvalue dimensions differ.");
    }
    double maximum = 0.0;
    for (std::size_t index = 0; index != left.size(); ++index)
    {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

double max_abs(const Matrix& matrix)
{
    double maximum = 0.0;
    for (const Complex value: matrix)
    {
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

const FixedAOSpinData& find_fixed_spin(const FixedAOData& fixed_ao, const int spin_index)
{
    for (const FixedAOSpinData& spin: fixed_ao.spins)
    {
        if (spin.spin_index == spin_index)
        {
            return spin;
        }
    }
    throw std::invalid_argument("Sternheimer response fixed-AO spin channel is missing.");
}

void validate_options(const ResponseSpectralOptions& options)
{
    if (!std::isfinite(options.relative_rank_tolerance)
        || options.relative_rank_tolerance <= 0.0 || options.relative_rank_tolerance >= 1.0
        || !std::isfinite(options.condition_limit) || options.condition_limit < 1.0
        || !std::isfinite(options.eigenvalue_tolerance_ha)
        || options.eigenvalue_tolerance_ha <= 0.0
        || !std::isfinite(options.occupation_tolerance) || options.occupation_tolerance <= 0.0)
    {
        throw std::invalid_argument("Sternheimer response spectral options are invalid.");
    }
}

void validate_common_data(const PrimitiveGalerkinData& response, const FixedAOData& fixed_ao)
{
    if (response.n_primitive <= 0 || response.n_fixed_ao <= 0
        || fixed_ao.n_basis != response.n_fixed_ao || response.spins.empty()
        || response.auxiliary_channels.empty()
        || response.auxiliary_channels.size() != fixed_ao.auxiliary_channels.size()
        || response.frequency_ha.empty() || response.frequency_ha != fixed_ao.frequency_ha
        || response.frequency_weights_ha != fixed_ao.frequency_weights_ha)
    {
        throw std::invalid_argument("Sternheimer response and fixed-AO data are inconsistent.");
    }
    for (std::size_t auxiliary = 0; auxiliary != response.auxiliary_channels.size(); ++auxiliary)
    {
        const AuxiliaryChannelMetadata& response_channel = response.auxiliary_channels[auxiliary];
        const AuxiliaryChannelMetadata& fixed_channel = fixed_ao.auxiliary_channels[auxiliary];
        if (response_channel.channel_index != fixed_channel.channel_index
            || response_channel.atom_index != fixed_channel.atom_index
            || response_channel.angular_momentum != fixed_channel.angular_momentum
            || response_channel.radial_index != fixed_channel.radial_index
            || response_channel.magnetic_index != fixed_channel.magnetic_index
            || response_channel.label != fixed_channel.label)
        {
            throw std::invalid_argument(
                "Sternheimer response and fixed-AO auxiliary channel ordering differs.");
        }
    }
    const int nresponse = response.n_primitive;
    const int nfixed = response.n_fixed_ao;
    require_shape(response.overlap_s, nresponse, nresponse, "Sternheimer response overlap");
    require_shape(response.fixed_ao_grid_overlap,
                  nfixed,
                  nfixed,
                  "Sternheimer fixed-AO grid overlap");
    require_shape(response.primitive_ao_overlap,
                  nresponse,
                  nfixed,
                  "Sternheimer response-fixed overlap");
    require_shape(fixed_ao.overlap_s, nfixed, nfixed, "Sternheimer fixed-AO LCAO overlap");
    const std::size_t naux = response.auxiliary_channels.size();
    if (response.perturbations_ha.size() != naux
        || response.primitive_ao_perturbations_ha.size() != naux
        || fixed_ao.perturbations_ha.size() != naux)
    {
        throw std::invalid_argument("Sternheimer response perturbation counts differ.");
    }
    for (std::size_t auxiliary = 0; auxiliary != naux; ++auxiliary)
    {
        require_shape(response.perturbations_ha[auxiliary],
                      nresponse,
                      nresponse,
                      "Sternheimer response perturbation");
        require_shape(response.primitive_ao_perturbations_ha[auxiliary],
                      nresponse,
                      nfixed,
                      "Sternheimer response-fixed perturbation");
        require_shape(fixed_ao.perturbations_ha[auxiliary],
                      nfixed,
                      nfixed,
                      "Sternheimer fixed-AO perturbation");
    }
}

} // namespace

ResponseSpectralResult evaluate_response_orbital_spectral_response(
    const PrimitiveGalerkinData& response,
    const FixedAOData& fixed_ao,
    const ResponseSpectralOptions& options)
{
    validate_options(options);
    validate_common_data(response, fixed_ao);

    const int nresponse = response.n_primitive;
    const int nfixed = response.n_fixed_ao;
    const int dimension = nfixed + nresponse;
    const int naux = static_cast<int>(response.auxiliary_channels.size());
    ResponseSpectralResult result;
    result.response_m.assign(response.frequency_ha.size(),
                             Matrix(matrix_size(naux, naux, "Sternheimer response M"),
                                    Complex(0.0, 0.0)));

    for (const PrimitiveGalerkinSpinData& response_spin: response.spins)
    {
        const FixedAOSpinData& fixed_spin = find_fixed_spin(fixed_ao, response_spin.spin_index);
        if (response_spin.fixed_ao_occupations != fixed_spin.occupations)
        {
            throw std::invalid_argument("Sternheimer response fixed-AO occupations differ.");
        }
        require_shape(response_spin.hamiltonian_ha,
                      nresponse,
                      nresponse,
                      "Sternheimer response Hamiltonian");
        require_shape(response_spin.fixed_ao_grid_hamiltonian_ha,
                      nfixed,
                      nfixed,
                      "Sternheimer fixed-AO grid Hamiltonian");
        require_shape(response_spin.primitive_ao_hamiltonian_ha,
                      nresponse,
                      nfixed,
                      "Sternheimer response-fixed Hamiltonian");
        require_shape(fixed_spin.hamiltonian_ha,
                      nfixed,
                      nfixed,
                      "Sternheimer fixed-AO LCAO Hamiltonian");
        if (fixed_spin.eigenvalues_ha.size() != static_cast<std::size_t>(nfixed)
            || fixed_spin.occupations.size() != static_cast<std::size_t>(nfixed))
        {
            throw std::invalid_argument("Sternheimer fixed-AO eigensystem is incomplete.");
        }

        int occupied_index = -1;
        for (int state = 0; state != nfixed; ++state)
        {
            if (fixed_spin.occupations[static_cast<std::size_t>(state)] > options.occupation_tolerance)
            {
                if (occupied_index >= 0)
                {
                    throw std::invalid_argument(
                        "Sternheimer response spectral path currently requires one occupied state per active spin.");
                }
                occupied_index = state;
            }
        }
        if (occupied_index < 0)
        {
            continue;
        }

        const GeneralizedEigensystem fixed_eigensystem
            = generalized_eigensystem(fixed_ao.overlap_s,
                                      fixed_spin.hamiltonian_ha,
                                      nfixed,
                                      options);
        const double fixed_eigenvalue_error
            = max_abs_difference(fixed_eigensystem.eigenvalues, fixed_spin.eigenvalues_ha);
        if (fixed_eigenvalue_error > options.eigenvalue_tolerance_ha)
        {
            throw std::runtime_error("Sternheimer fixed-AO eigensystem does not reproduce its reference eigenvalues.");
        }

        Matrix occupied(static_cast<std::size_t>(dimension), Complex(0.0, 0.0));
        for (int row = 0; row != nfixed; ++row)
        {
            occupied[static_cast<std::size_t>(row)]
                = fixed_eigensystem.coefficients[static_cast<std::size_t>(row * nfixed + occupied_index)];
        }
        const Matrix fixed_grid_times_occupied
            = multiply(response.fixed_ao_grid_overlap, nfixed, nfixed,
                       Matrix(occupied.begin(), occupied.begin() + nfixed), 1);
        Complex grid_norm(0.0, 0.0);
        for (int row = 0; row != nfixed; ++row)
        {
            grid_norm += std::conj(occupied[static_cast<std::size_t>(row)])
                         * fixed_grid_times_occupied[static_cast<std::size_t>(row)];
        }
        if (!std::isfinite(grid_norm.real()) || grid_norm.real() <= 0.0
            || std::abs(grid_norm.imag()) > 1.0e-10)
        {
            throw std::runtime_error("Sternheimer fixed occupied state has non-positive grid norm.");
        }
        const double inverse_grid_norm = 1.0 / std::sqrt(grid_norm.real());
        for (Complex& value: occupied)
        {
            value *= inverse_grid_norm;
        }

        const Matrix overlap = assemble_hermitian_blocks(response.fixed_ao_grid_overlap,
                                                          nfixed,
                                                          response.overlap_s,
                                                          nresponse,
                                                          response.primitive_ao_overlap);
        const Matrix hamiltonian
            = assemble_hermitian_blocks(response_spin.fixed_ao_grid_hamiltonian_ha,
                                        nfixed,
                                        response_spin.hamiltonian_ha,
                                        nresponse,
                                        response_spin.primitive_ao_hamiltonian_ha);
        std::vector<Matrix> perturbations;
        perturbations.reserve(static_cast<std::size_t>(naux));
        for (int auxiliary = 0; auxiliary != naux; ++auxiliary)
        {
            perturbations.push_back(
                assemble_hermitian_blocks(fixed_ao.perturbations_ha[static_cast<std::size_t>(auxiliary)],
                                          nfixed,
                                          response.perturbations_ha[static_cast<std::size_t>(auxiliary)],
                                          nresponse,
                                          response.primitive_ao_perturbations_ha[static_cast<std::size_t>(auxiliary)]));
        }

        const Matrix occupied_adjoint = adjoint(occupied, dimension, 1);
        const Matrix occupied_adjoint_overlap
            = multiply(occupied_adjoint, 1, dimension, overlap, dimension);
        Matrix projector(matrix_size(dimension, dimension, "Sternheimer occupied projector"),
                         Complex(0.0, 0.0));
        for (int row = 0; row != dimension; ++row)
        {
            for (int column = 0; column != dimension; ++column)
            {
                projector[static_cast<std::size_t>(row * dimension + column)]
                    = (row == column ? Complex(1.0, 0.0) : Complex(0.0, 0.0))
                      - occupied[static_cast<std::size_t>(row)]
                            * occupied_adjoint_overlap[static_cast<std::size_t>(column)];
            }
        }
        const Matrix overlap_projector = multiply(overlap, dimension, dimension, projector, dimension);
        const Matrix projected_overlap
            = multiply(adjoint(projector, dimension, dimension),
                       dimension,
                       dimension,
                       overlap_projector,
                       dimension);
        const MetricWhitener projected_metric
            = metric_whitener(projected_overlap,
                              dimension,
                              options.relative_rank_tolerance,
                              options.condition_limit,
                              true);
        const Matrix complement_basis
            = multiply(projector,
                       dimension,
                       dimension,
                       projected_metric.transform,
                       projected_metric.rank);
        const Matrix projected_hamiltonian
            = sandwich(complement_basis, dimension, projected_metric.rank, hamiltonian);
        const HermitianEigensystem virtual_eigensystem
            = diagonalize_hermitian(projected_hamiltonian, projected_metric.rank);
        const Matrix virtual_coefficients
            = multiply(complement_basis,
                       dimension,
                       projected_metric.rank,
                       virtual_eigensystem.eigenvectors,
                       projected_metric.rank);

        const Matrix occupied_overlap_virtual
            = multiply(occupied_adjoint_overlap,
                       1,
                       dimension,
                       virtual_coefficients,
                       projected_metric.rank);
        const Matrix virtual_overlap
            = sandwich(virtual_coefficients, dimension, projected_metric.rank, overlap);
        Matrix virtual_overlap_error(virtual_overlap);
        for (int state = 0; state != projected_metric.rank; ++state)
        {
            virtual_overlap_error[static_cast<std::size_t>(state * projected_metric.rank + state)]
                -= Complex(1.0, 0.0);
        }
        const Matrix occupied_overlap
            = multiply(occupied_adjoint_overlap, 1, dimension, occupied, 1);

        Matrix couplings(matrix_size(projected_metric.rank,
                                     naux,
                                     "Sternheimer response couplings"),
                         Complex(0.0, 0.0));
        const Matrix virtual_adjoint
            = adjoint(virtual_coefficients, dimension, projected_metric.rank);
        for (int auxiliary = 0; auxiliary != naux; ++auxiliary)
        {
            const Matrix perturbation_on_occupied
                = multiply(perturbations[static_cast<std::size_t>(auxiliary)],
                           dimension,
                           dimension,
                           occupied,
                           1);
            const Matrix coupling
                = multiply(virtual_adjoint,
                           projected_metric.rank,
                           dimension,
                           perturbation_on_occupied,
                           1);
            for (int state = 0; state != projected_metric.rank; ++state)
            {
                couplings[static_cast<std::size_t>(state * naux + auxiliary)]
                    = coupling[static_cast<std::size_t>(state)];
            }
        }

        const double occupied_energy
            = fixed_spin.eigenvalues_ha[static_cast<std::size_t>(occupied_index)];
        const double occupation
            = fixed_spin.occupations[static_cast<std::size_t>(occupied_index)];
        for (std::size_t ifrequency = 0; ifrequency != response.frequency_ha.size(); ++ifrequency)
        {
            Matrix half(matrix_size(naux, naux, "Sternheimer response half"), Complex(0.0, 0.0));
            const double omega = response.frequency_ha[ifrequency];
            for (int state = 0; state != projected_metric.rank; ++state)
            {
                const Complex denominator(virtual_eigensystem.eigenvalues[static_cast<std::size_t>(state)]
                                              - occupied_energy,
                                          omega);
                for (int row = 0; row != naux; ++row)
                {
                    const Complex left
                        = std::conj(couplings[static_cast<std::size_t>(state * naux + row)]);
                    for (int column = 0; column != naux; ++column)
                    {
                        half[static_cast<std::size_t>(row * naux + column)]
                            -= occupation * left
                               * couplings[static_cast<std::size_t>(state * naux + column)]
                               / denominator;
                    }
                }
            }
            Matrix& response_m = result.response_m[ifrequency];
            for (int row = 0; row != naux; ++row)
            {
                for (int column = 0; column != naux; ++column)
                {
                    response_m[static_cast<std::size_t>(row * naux + column)]
                        += half[static_cast<std::size_t>(row * naux + column)]
                           + std::conj(half[static_cast<std::size_t>(column * naux + row)]);
                }
            }
        }

        ResponseSpectralSpinDiagnostics diagnostics;
        diagnostics.spin_index = response_spin.spin_index;
        diagnostics.retained_virtual_rank = projected_metric.rank;
        diagnostics.dropped_trial_rank = projected_metric.dropped;
        diagnostics.fixed_ao_eigenvalue_max_abs_error_ha = fixed_eigenvalue_error;
        diagnostics.occupied_grid_norm_before_normalization = grid_norm.real();
        diagnostics.projected_overlap_condition = projected_metric.condition;
        diagnostics.occupied_orthonormality_max_abs_error
            = std::abs(occupied_overlap.front() - Complex(1.0, 0.0));
        diagnostics.occupied_virtual_max_abs_overlap = max_abs(occupied_overlap_virtual);
        diagnostics.virtual_orthonormality_max_abs_error = max_abs(virtual_overlap_error);
        diagnostics.minimum_virtual_energy_ha = virtual_eigensystem.eigenvalues.front();
        diagnostics.maximum_virtual_energy_ha = virtual_eigensystem.eigenvalues.back();
        result.spin_diagnostics.push_back(diagnostics);
    }

    if (result.spin_diagnostics.empty())
    {
        throw std::invalid_argument("Sternheimer response spectral path found no active spin channel.");
    }
    return result;
}

} // namespace sternheimer_siab
} // namespace module_ri
