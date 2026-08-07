#include "source_lcao/module_ri/sternheimer_siab_fixed_ao.h"

#include <complex>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace
{

namespace siab = module_ri::sternheimer_siab;
using Complex = std::complex<double>;

siab::FixedAOData build_fixture()
{
    const std::vector<Complex> overlap = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
    const std::vector<siab::FixedAOSpinInput> spins = {
        siab::FixedAOSpinInput{0,
                               {-1.0, 2.0},
                               {2.0, 0.0},
                               {{-1.0, 0.0}, {0.0, 0.2}, {0.0, -0.2}, {2.0, 0.0}}},
    };
    const std::vector<siab::AuxiliaryChannelMetadata> channels = {
        siab::AuxiliaryChannelMetadata{0, 0, 0, 0, 0, "H"},
    };
    const std::vector<std::vector<Complex>> basis = {
        {{1.0, 0.0}, {0.0, 1.0}},
        {{1.0, 1.0}, {2.0, 0.0}},
    };
    return siab::build_fixed_ao_data(
        2, overlap, spins, channels, basis, {{2.0, -1.0}}, {0.2, 0.8}, {0.3, 0.7}, 0.5);
}

} // namespace

TEST(SternheimerSIABFixedAO, ConvertsRydbergMatricesAndEigenvaluesToHartree)
{
    const siab::FixedAOData data = build_fixture();

    ASSERT_EQ(data.spins.size(), 1);
    EXPECT_DOUBLE_EQ(data.spins[0].eigenvalues_ha[0], -0.5);
    EXPECT_DOUBLE_EQ(data.spins[0].eigenvalues_ha[1], 1.0);
    EXPECT_EQ(data.spins[0].hamiltonian_ha[0], Complex(-0.5, 0.0));
    EXPECT_EQ(data.spins[0].hamiltonian_ha[1], Complex(0.0, 0.1));
    EXPECT_EQ(data.spins[0].hamiltonian_ha[2], Complex(0.0, -0.1));
    EXPECT_EQ(data.spins[0].hamiltonian_ha[3], Complex(1.0, 0.0));
    EXPECT_EQ(data.spins[0].occupations, std::vector<double>({2.0, 0.0}));
}

TEST(SternheimerSIABFixedAO, UsesHaGridPerturbationsWithoutEnergyRescaling)
{
    const siab::FixedAOData data = build_fixture();

    ASSERT_EQ(data.perturbations_ha.size(), 1);
    EXPECT_EQ(data.perturbations_ha[0][0], Complex(0.5, 0.0));
    EXPECT_EQ(data.perturbations_ha[0][1], Complex(1.0, 2.0));
    EXPECT_EQ(data.perturbations_ha[0][2], Complex(1.0, -2.0));
    EXPECT_EQ(data.perturbations_ha[0][3], Complex(0.0, 0.0));
}

TEST(SternheimerSIABFixedAO, RejectsDimensionsBeforeConstructingOutput)
{
    const std::vector<Complex> overlap(4, Complex(0.0, 0.0));
    const std::vector<siab::AuxiliaryChannelMetadata> channels = {
        siab::AuxiliaryChannelMetadata{0, 0, 0, 0, 0, "H"},
    };
    const std::vector<std::vector<Complex>> basis(2, std::vector<Complex>(2, Complex(0.0, 0.0)));
    const std::vector<siab::FixedAOSpinInput> wrong_spin = {
        siab::FixedAOSpinInput{0, {-1.0}, {2.0}, std::vector<Complex>(4, Complex(0.0, 0.0))},
    };

    EXPECT_THROW(siab::build_fixed_ao_data(
                     2, overlap, wrong_spin, channels, basis, {{1.0, 1.0}}, {0.2}, {1.0}, 0.5),
                 std::invalid_argument);
}
