#include "sternheimer_siab_fixed_ao.h"

#include "sternheimer_siab_overlap.h"

#include <stdexcept>
#include <utility>

namespace module_ri
{
namespace sternheimer_siab
{
namespace
{

constexpr double rydberg_to_hartree = 0.5;

} // namespace

FixedAOData build_fixed_ao_data(const int n_basis,
                                const std::vector<std::complex<double>>& overlap_s,
                                const std::vector<FixedAOSpinInput>& spins,
                                const std::vector<AuxiliaryChannelMetadata>& auxiliary_channels,
                                const std::vector<std::vector<std::complex<double>>>& basis_functions,
                                const std::vector<std::vector<double>>& potentials_ha,
                                const std::vector<double>& frequency_ha,
                                const std::vector<double>& frequency_weights_ha,
                                const double delta_omega)
{
    if (n_basis <= 0)
    {
        throw std::invalid_argument("Sternheimer SIAB fixed-AO assembly requires a positive basis size.");
    }
    const std::size_t dimension = static_cast<std::size_t>(n_basis);
    if (overlap_s.size() != dimension * dimension || basis_functions.size() != dimension || spins.empty())
    {
        throw std::invalid_argument("Sternheimer SIAB fixed-AO assembly dimensions are inconsistent.");
    }
    if (auxiliary_channels.size() != potentials_ha.size() || auxiliary_channels.empty()
        || frequency_ha.empty() || frequency_ha.size() != frequency_weights_ha.size())
    {
        throw std::invalid_argument("Sternheimer SIAB fixed-AO channel or frequency dimensions are inconsistent.");
    }

    FixedAOData result;
    result.n_basis = n_basis;
    result.overlap_s = overlap_s;
    result.auxiliary_channels = auxiliary_channels;
    result.frequency_ha = frequency_ha;
    result.frequency_weights_ha = frequency_weights_ha;
    result.perturbations_ha = perturbation_matrices(basis_functions, potentials_ha, delta_omega);
    result.spins.reserve(spins.size());
    for (const FixedAOSpinInput& input: spins)
    {
        if (input.eigenvalues_ry.size() != dimension || input.occupations.size() != dimension
            || input.hamiltonian_ry.size() != dimension * dimension)
        {
            throw std::invalid_argument("Sternheimer SIAB fixed-AO spin input dimensions are inconsistent.");
        }
        FixedAOSpinData output;
        output.spin_index = input.spin_index;
        output.occupations = input.occupations;
        output.eigenvalues_ha.reserve(dimension);
        for (const double value: input.eigenvalues_ry)
        {
            output.eigenvalues_ha.push_back(value * rydberg_to_hartree);
        }
        output.hamiltonian_ha.reserve(dimension * dimension);
        for (const std::complex<double>& value: input.hamiltonian_ry)
        {
            output.hamiltonian_ha.push_back(value * rydberg_to_hartree);
        }
        result.spins.push_back(std::move(output));
    }
    return result;
}

} // namespace sternheimer_siab
} // namespace module_ri
