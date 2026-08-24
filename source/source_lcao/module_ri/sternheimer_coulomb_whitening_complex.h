#ifndef STERNHEIMER_COULOMB_WHITENING_COMPLEX_H
#define STERNHEIMER_COULOMB_WHITENING_COMPLEX_H

#include <complex>
#include <vector>

namespace ModuleRI
{

struct SternheimerComplexCoulombWhitening
{
    int raw_dimension = 0;
    int retained_rank = 0;
    int discarded_rank = 0;
    double relative_threshold = 0.0;
    double largest_eigenvalue = 0.0;
    double smallest_retained_eigenvalue = 0.0;
    double max_antihermitian_error = 0.0;
    double max_orthonormality_error = 0.0;
    std::vector<double> eigenvalues;
    std::vector<std::complex<double>> transform; ///< row-major raw_dimension x retained_rank
};

SternheimerComplexCoulombWhitening make_sternheimer_complex_coulomb_whitening(
    const std::vector<std::complex<double>>& metric,
    int dimension,
    double relative_threshold);

std::vector<std::complex<double>> apply_sternheimer_complex_channel_transform(
    const std::vector<std::complex<double>>& rows_by_raw_channel,
    int row_count,
    int raw_dimension,
    const SternheimerComplexCoulombWhitening& whitening);

} // namespace ModuleRI

#endif
