#ifndef STERNHEIMER_SIAB_SOURCE_H
#define STERNHEIMER_SIAB_SOURCE_H

#include <complex>
#include <vector>

namespace module_ri
{
namespace sternheimer_siab
{

using ComplexGrid = std::vector<std::complex<double>>;

ComplexGrid build_source_grid_from_rydberg_potential(
    const ComplexGrid& occupied_wavefunction,
    const std::vector<double>& perturbation_ry);

} // namespace sternheimer_siab
} // namespace module_ri

#endif
