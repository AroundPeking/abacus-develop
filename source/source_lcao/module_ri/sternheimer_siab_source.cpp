#include "sternheimer_siab_source.h"

#include <cmath>
#include <stdexcept>

namespace module_ri
{
namespace sternheimer_siab
{

ComplexGrid build_source_grid_from_rydberg_potential(
    const ComplexGrid& occupied_wavefunction,
    const std::vector<double>& perturbation_ry)
{
    if (occupied_wavefunction.empty() || perturbation_ry.empty())
    {
        throw std::invalid_argument("SIAB source inputs must be nonempty");
    }
    if (occupied_wavefunction.size() != perturbation_ry.size())
    {
        throw std::invalid_argument("SIAB source inputs must have equal sizes");
    }

    ComplexGrid source_grid(occupied_wavefunction.size());
    for (std::size_t ir = 0; ir != occupied_wavefunction.size(); ++ir)
    {
        const auto& psi = occupied_wavefunction[ir];
        const double potential_ry = perturbation_ry[ir];
        if (!std::isfinite(potential_ry)
            || !std::isfinite(psi.real())
            || !std::isfinite(psi.imag()))
        {
            throw std::invalid_argument("SIAB source inputs must be finite");
        }
        source_grid[ir] = psi * (0.5 * potential_ry);
    }
    return source_grid;
}

} // namespace sternheimer_siab
} // namespace module_ri
