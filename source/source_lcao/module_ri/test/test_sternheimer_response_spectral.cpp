#include "source_lcao/module_ri/sternheimer_response_spectral.h"

#include <complex>
#include <gtest/gtest.h>
#include <vector>

namespace
{

using Complex = std::complex<double>;
namespace siab = module_ri::sternheimer_siab;

siab::PrimitiveGalerkinData response_fixture()
{
    siab::PrimitiveGalerkinData data;
    data.n_primitive = 1;
    data.n_fixed_ao = 1;
    data.auxiliary_channels = {
        siab::AuxiliaryChannelMetadata{0, 0, 0, 0, 0, "H0_l0_n0_m0"},
        siab::AuxiliaryChannelMetadata{1, 0, 1, 0, 0, "H0_l1_n0_m0"},
    };
    data.overlap_s = {Complex(1.0, 0.0)};
    data.fixed_ao_grid_overlap = {Complex(1.0, 0.0)};
    data.primitive_ao_overlap = {Complex(0.0, 0.0)};
    data.perturbations_ha = {
        {Complex(0.0, 0.0)},
        {Complex(0.0, 0.0)},
    };
    data.primitive_ao_perturbations_ha = {
        {Complex(2.0, 0.0)},
        {Complex(0.0, 1.0)},
    };
    data.frequency_ha = {1.0};
    data.frequency_weights_ha = {0.5};

    siab::PrimitiveGalerkinSpinData spin;
    spin.spin_index = 0;
    spin.fixed_ao_occupations = {2.0};
    spin.hamiltonian_ha = {Complex(0.5, 0.0)};
    spin.fixed_ao_grid_hamiltonian_ha = {Complex(-0.5, 0.0)};
    spin.primitive_ao_hamiltonian_ha = {Complex(0.0, 0.0)};
    data.spins = {spin};
    return data;
}

siab::FixedAOData fixed_fixture()
{
    siab::FixedAOData data;
    data.n_basis = 1;
    data.overlap_s = {Complex(1.0, 0.0)};
    data.auxiliary_channels = response_fixture().auxiliary_channels;
    data.perturbations_ha = {
        {Complex(0.0, 0.0)},
        {Complex(0.0, 0.0)},
    };
    data.frequency_ha = {1.0};
    data.frequency_weights_ha = {0.5};

    siab::FixedAOSpinData spin;
    spin.spin_index = 0;
    spin.eigenvalues_ha = {-0.5};
    spin.occupations = {2.0};
    spin.hamiltonian_ha = {Complex(-0.5, 0.0)};
    data.spins = {spin};
    return data;
}

} // namespace

TEST(SternheimerResponseSpectral, ReproducesProjectedSosSignOccupationAndConjugateBranch)
{
    const siab::ResponseSpectralResult result
        = siab::evaluate_response_orbital_spectral_response(response_fixture(), fixed_fixture());

    ASSERT_EQ(result.response_m.size(), 1);
    ASSERT_EQ(result.response_m[0].size(), 4);
    EXPECT_NEAR(result.response_m[0][0].real(), -8.0, 1.0e-12);
    EXPECT_NEAR(result.response_m[0][0].imag(), 0.0, 1.0e-12);
    EXPECT_NEAR(result.response_m[0][1].real(), 0.0, 1.0e-12);
    EXPECT_NEAR(result.response_m[0][1].imag(), -4.0, 1.0e-12);
    EXPECT_NEAR(result.response_m[0][2].real(), 0.0, 1.0e-12);
    EXPECT_NEAR(result.response_m[0][2].imag(), 4.0, 1.0e-12);
    EXPECT_NEAR(result.response_m[0][3].real(), -2.0, 1.0e-12);
    EXPECT_NEAR(result.response_m[0][3].imag(), 0.0, 1.0e-12);

    ASSERT_EQ(result.spin_diagnostics.size(), 1);
    EXPECT_EQ(result.spin_diagnostics[0].retained_virtual_rank, 1);
    EXPECT_EQ(result.spin_diagnostics[0].dropped_trial_rank, 1);
    EXPECT_NEAR(result.spin_diagnostics[0].occupied_virtual_max_abs_overlap, 0.0, 1.0e-12);
}

TEST(SternheimerResponseSpectral, RejectsMismatchedAuxiliaryChannelOrdering)
{
    siab::FixedAOData fixed = fixed_fixture();
    fixed.auxiliary_channels[1].magnetic_index = 1;

    EXPECT_THROW(siab::evaluate_response_orbital_spectral_response(response_fixture(), fixed),
                 std::invalid_argument);
}
