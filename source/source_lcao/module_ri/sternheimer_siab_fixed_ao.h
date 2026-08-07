#ifndef STERNHEIMER_SIAB_FIXED_AO_H
#define STERNHEIMER_SIAB_FIXED_AO_H

#include "sternheimer_siab_data.h"

#include <complex>
#include <vector>

namespace module_ri
{
namespace sternheimer_siab
{

struct FixedAOSpinInput
{
    int spin_index;
    std::vector<double> eigenvalues_ry;
    std::vector<double> occupations;
    std::vector<std::complex<double>> hamiltonian_ry;
};

FixedAOData build_fixed_ao_data(
    int n_basis,
    const std::vector<std::complex<double>>& overlap_s,
    const std::vector<FixedAOSpinInput>& spins,
    const std::vector<AuxiliaryChannelMetadata>& auxiliary_channels,
    const std::vector<std::vector<std::complex<double>>>& basis_functions,
    const std::vector<std::vector<double>>& potentials_ha,
    const std::vector<double>& frequency_ha,
    const std::vector<double>& frequency_weights_ha,
    double delta_omega);

} // namespace sternheimer_siab
} // namespace module_ri

#endif
