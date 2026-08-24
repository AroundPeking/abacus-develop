#include "source_lcao/module_ri/sternheimer_coulomb_whitening_complex.h"

#include "source_base/module_external/lapack_connector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ModuleRI
{
namespace
{

using Complex = std::complex<double>;

void validate_shape(const std::vector<Complex>& values, const int rows, const int columns, const char* label)
{
    if (rows < 0 || columns < 0 || values.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns))
    {
        throw std::invalid_argument(std::string(label) + " has inconsistent dimensions.");
    }
}

std::vector<double> diagonalize_hermitian(std::vector<Complex>& column_major, const int dimension)
{
    const char jobz = 'V';
    const char uplo = 'U';
    const int lda = dimension;
    int info = 0;
    std::vector<double> eigenvalues(static_cast<std::size_t>(dimension), 0.0);
    std::vector<double> rwork(static_cast<std::size_t>(std::max(1, 3 * dimension - 2)), 0.0);
    Complex work_query(0.0, 0.0);
    const int query = -1;

    zheev_(&jobz,
           &uplo,
           &dimension,
           column_major.data(),
           &lda,
           eigenvalues.data(),
           &work_query,
           &query,
           rwork.data(),
           &info);
    if (info != 0 || !std::isfinite(work_query.real()))
    {
        throw std::runtime_error("Complex Sternheimer Coulomb whitening eigensolver workspace query failed.");
    }
    const int lwork = std::max(1, static_cast<int>(std::ceil(work_query.real())));
    std::vector<Complex> work(static_cast<std::size_t>(lwork), Complex(0.0, 0.0));
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
        throw std::runtime_error("Complex Sternheimer Coulomb whitening eigensolver failed.");
    }
    return eigenvalues;
}

} // namespace

SternheimerComplexCoulombWhitening make_sternheimer_complex_coulomb_whitening(const std::vector<Complex>& metric,
                                                                              const int dimension,
                                                                              const double relative_threshold)
{
    if (dimension <= 0)
    {
        throw std::invalid_argument("Complex Sternheimer Coulomb whitening dimension must be positive.");
    }
    validate_shape(metric, dimension, dimension, "Complex Sternheimer Coulomb metric");
    if (!std::isfinite(relative_threshold) || relative_threshold <= 0.0 || relative_threshold >= 1.0)
    {
        throw std::invalid_argument(
            "Complex Sternheimer Coulomb whitening relative threshold must be finite and between zero and one.");
    }

    double metric_scale = 0.0;
    double max_antihermitian_error = 0.0;
    for (int row = 0; row != dimension; ++row)
    {
        for (int column = 0; column != dimension; ++column)
        {
            const Complex left = metric[static_cast<std::size_t>(row * dimension + column)];
            const Complex right = metric[static_cast<std::size_t>(column * dimension + row)];
            if (!std::isfinite(left.real()) || !std::isfinite(left.imag()) || !std::isfinite(right.real())
                || !std::isfinite(right.imag()))
            {
                throw std::invalid_argument("Complex Sternheimer Coulomb metric contains a non-finite value.");
            }
            metric_scale = std::max(metric_scale, std::max(std::abs(left), std::abs(right)));
            max_antihermitian_error = std::max(max_antihermitian_error, std::abs(left - std::conj(right)));
        }
    }
    const double hermitian_tolerance
        = std::max(1.0e-12, 4096.0 * std::numeric_limits<double>::epsilon() * static_cast<double>(dimension))
          * std::max(1.0, metric_scale);
    if (max_antihermitian_error > hermitian_tolerance)
    {
        throw std::invalid_argument("Complex Sternheimer Coulomb metric is materially non-Hermitian.");
    }

    std::vector<Complex> hermitian(metric.size(), Complex(0.0, 0.0));
    std::vector<Complex> column_major(metric.size(), Complex(0.0, 0.0));
    for (int row = 0; row != dimension; ++row)
    {
        for (int column = 0; column != dimension; ++column)
        {
            const Complex value = 0.5
                                  * (metric[static_cast<std::size_t>(row * dimension + column)]
                                     + std::conj(metric[static_cast<std::size_t>(column * dimension + row)]));
            hermitian[static_cast<std::size_t>(row * dimension + column)] = value;
            column_major[static_cast<std::size_t>(column * dimension + row)] = value;
        }
    }

    const std::vector<double> eigenvalues = diagonalize_hermitian(column_major, dimension);
    const double largest = eigenvalues.back();
    if (!std::isfinite(largest) || largest <= 0.0)
    {
        throw std::runtime_error("Complex Sternheimer Coulomb metric has no positive eigenspace.");
    }
    const double cutoff = relative_threshold * largest;
    const double roundoff_floor
        = 64.0 * std::numeric_limits<double>::epsilon() * static_cast<double>(dimension) * largest;
    const double negative_tolerance = std::max(cutoff, roundoff_floor);
    if (eigenvalues.front() < -negative_tolerance)
    {
        throw std::runtime_error("Complex Sternheimer Coulomb metric has a materially negative eigenvalue.");
    }

    int first_retained = 0;
    while (first_retained != dimension && eigenvalues[static_cast<std::size_t>(first_retained)] <= cutoff)
    {
        ++first_retained;
    }
    if (first_retained == dimension)
    {
        throw std::runtime_error("Complex Sternheimer Coulomb whitening threshold discarded every eigenvector.");
    }

    SternheimerComplexCoulombWhitening result;
    result.raw_dimension = dimension;
    result.retained_rank = dimension - first_retained;
    result.discarded_rank = first_retained;
    result.relative_threshold = relative_threshold;
    result.largest_eigenvalue = largest;
    result.smallest_retained_eigenvalue = eigenvalues[static_cast<std::size_t>(first_retained)];
    result.max_antihermitian_error = max_antihermitian_error;
    result.eigenvalues = eigenvalues;
    result.transform.assign(static_cast<std::size_t>(dimension) * static_cast<std::size_t>(result.retained_rank),
                            Complex(0.0, 0.0));

    // ZHEEV returns eigenvectors as columns in column-major storage.
    for (int retained = 0; retained != result.retained_rank; ++retained)
    {
        const int eigenvector = first_retained + retained;
        const double inverse_sqrt = 1.0 / std::sqrt(eigenvalues[static_cast<std::size_t>(eigenvector)]);
        for (int raw = 0; raw != dimension; ++raw)
        {
            result.transform[static_cast<std::size_t>(raw * result.retained_rank + retained)]
                = column_major[static_cast<std::size_t>(eigenvector * dimension + raw)] * inverse_sqrt;
        }
    }

    const std::vector<Complex> metric_times_transform
        = apply_sternheimer_complex_channel_transform(hermitian, dimension, dimension, result);
    double max_error = 0.0;
    for (int left = 0; left != result.retained_rank; ++left)
    {
        for (int right = 0; right != result.retained_rank; ++right)
        {
            Complex value(0.0, 0.0);
            for (int raw = 0; raw != dimension; ++raw)
            {
                value += std::conj(result.transform[static_cast<std::size_t>(raw * result.retained_rank + left)])
                         * metric_times_transform[static_cast<std::size_t>(raw * result.retained_rank + right)];
            }
            const Complex expected = left == right ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
            max_error = std::max(max_error, std::abs(value - expected));
        }
    }
    result.max_orthonormality_error = max_error;
    if (!std::isfinite(max_error) || max_error > 1.0e-8)
    {
        throw std::runtime_error("Complex Sternheimer Coulomb whitening failed its W^dagger V W identity check.");
    }
    return result;
}

std::vector<Complex> apply_sternheimer_complex_channel_transform(const std::vector<Complex>& rows_by_raw_channel,
                                                                 const int row_count,
                                                                 const int raw_dimension,
                                                                 const SternheimerComplexCoulombWhitening& whitening)
{
    validate_shape(rows_by_raw_channel, row_count, raw_dimension, "Complex Sternheimer raw-channel matrix");
    if (raw_dimension != whitening.raw_dimension || whitening.retained_rank <= 0)
    {
        throw std::invalid_argument(
            "Complex Sternheimer channel transform dimension does not match its whitening metric.");
    }
    validate_shape(whitening.transform,
                   whitening.raw_dimension,
                   whitening.retained_rank,
                   "Complex Sternheimer Coulomb whitening transform");

    std::vector<Complex> transformed(static_cast<std::size_t>(row_count)
                                         * static_cast<std::size_t>(whitening.retained_rank),
                                     Complex(0.0, 0.0));
#pragma omp parallel for schedule(static)
    for (int row = 0; row != row_count; ++row)
    {
        for (int raw = 0; raw != raw_dimension; ++raw)
        {
            const Complex raw_value = rows_by_raw_channel[static_cast<std::size_t>(row * raw_dimension + raw)];
            for (int retained = 0; retained != whitening.retained_rank; ++retained)
            {
                transformed[static_cast<std::size_t>(row * whitening.retained_rank + retained)]
                    += raw_value
                       * whitening.transform[static_cast<std::size_t>(raw * whitening.retained_rank + retained)];
            }
        }
    }
    return transformed;
}

} // namespace ModuleRI
