#ifndef STERNHEIMER_SIAB_PRIMITIVE_GALERKIN_H
#define STERNHEIMER_SIAB_PRIMITIVE_GALERKIN_H

#include "sternheimer_siab_fixed_ao.h"
#include "sternheimer_siab_overlap.h"

#include <complex>
#include <vector>

namespace module_ri
{
namespace sternheimer_siab
{

PrimitiveGalerkinData build_primitive_galerkin_data(
    const std::vector<PrimitiveBlock>& blocks,
    const std::vector<AuxiliaryChannelMetadata>& auxiliary_channels,
    std::vector<std::vector<std::complex<double>>> primitive_basis_functions,
    std::vector<std::vector<std::complex<double>>> fixed_ao_basis_functions,
    const std::vector<FixedAOSpinInput>& fixed_ao_spins,
    const std::vector<ModuleRI::SternheimerFDHamiltonian>& hamiltonians_ry,
    const std::vector<std::vector<double>>& potentials_ha,
    const std::vector<double>& frequency_ha,
    const std::vector<double>& frequency_weights_ha,
    double delta_omega);

} // namespace sternheimer_siab
} // namespace module_ri

#endif
