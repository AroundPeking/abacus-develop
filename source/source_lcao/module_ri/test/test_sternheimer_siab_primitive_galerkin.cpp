#include "source_lcao/module_ri/sternheimer_siab_primitive_galerkin.h"

#include <complex>
#include <gtest/gtest.h>
#include <vector>

namespace
{

using Complex = std::complex<double>;
namespace siab = module_ri::sternheimer_siab;

ModuleRI::SternheimerFDHamiltonian local_hamiltonian(const std::vector<double>& potential)
{
    ModuleRI::SternheimerFDHamiltonian::Grid grid;
    grid.nx = 2;
    grid.ny = 1;
    grid.nz = 1;
    grid.hx = 1.0;
    grid.hy = 1.0;
    grid.hz = 1.0;
    return ModuleRI::SternheimerFDHamiltonian(grid, potential, 0.0);
}

} // namespace

TEST(SternheimerSIABPrimitiveGalerkin, AssemblesGridMatricesWithCorrectConjugationAndUnits)
{
    const std::vector<siab::PrimitiveBlock> blocks = {
        siab::PrimitiveBlock{"H", 0, 0, 0, 2, 0}};
    const std::vector<siab::AuxiliaryChannelMetadata> channels = {
        siab::AuxiliaryChannelMetadata{0, 0, 0, 0, 0, "H0_l0_n0_m0"}};
    const std::vector<std::vector<Complex>> primitives = {
        {{1.0, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 1.0}},
    };
    const std::vector<std::vector<Complex>> fixed_ao = {
        {{0.5, 0.0}, {0.0, 0.0}},
        {{0.0, 0.0}, {0.0, 2.0}},
    };
    const std::vector<siab::FixedAOSpinInput> spins = {
        siab::FixedAOSpinInput{
            0,
            {-1.0, 1.0},
            {1.0, 0.0},
            {{-1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}}},
    };
    const std::vector<ModuleRI::SternheimerFDHamiltonian> hamiltonians = {
        local_hamiltonian({2.0, -1.0})};

    const siab::PrimitiveGalerkinData data = siab::build_primitive_galerkin_data(
        blocks,
        channels,
        primitives,
        fixed_ao,
        spins,
        hamiltonians,
        {{3.0, 4.0}},
        {0.2},
        {0.5},
        0.5);

    EXPECT_EQ(data.n_primitive, 2);
    EXPECT_EQ(data.n_fixed_ao, 2);
    EXPECT_EQ(data.overlap_s,
              (std::vector<Complex>{{0.5, 0.0}, {0.0, 0.0},
                                    {0.0, 0.0}, {0.5, 0.0}}));
    EXPECT_EQ(data.primitive_ao_overlap,
              (std::vector<Complex>{{0.25, 0.0}, {0.0, 0.0},
                                    {0.0, 0.0}, {1.0, 0.0}}));
    EXPECT_EQ(data.fixed_ao_grid_overlap,
              (std::vector<Complex>{{0.125, 0.0}, {0.0, 0.0},
                                    {0.0, 0.0}, {2.0, 0.0}}));
    ASSERT_EQ(data.spins.size(), 1);
    EXPECT_EQ(data.spins[0].fixed_ao_occupations, (std::vector<double>{1.0, 0.0}));
    EXPECT_EQ(data.spins[0].hamiltonian_ha,
              (std::vector<Complex>{{0.5, 0.0}, {0.0, 0.0},
                                    {0.0, 0.0}, {-0.25, 0.0}}));
    EXPECT_EQ(data.spins[0].fixed_ao_grid_hamiltonian_ha,
              (std::vector<Complex>{{0.125, 0.0}, {0.0, 0.0},
                                    {0.0, 0.0}, {-1.0, 0.0}}));
    ASSERT_EQ(data.perturbations_ha.size(), 1);
    EXPECT_EQ(data.perturbations_ha[0],
              (std::vector<Complex>{{1.5, 0.0}, {0.0, 0.0},
                                    {0.0, 0.0}, {2.0, 0.0}}));
}

TEST(SternheimerSIABPrimitiveGalerkin, RejectsMismatchedSpinHamiltonians)
{
    const std::vector<siab::PrimitiveBlock> blocks = {
        siab::PrimitiveBlock{"H", 0, 0, 0, 1, 0}};
    const std::vector<siab::AuxiliaryChannelMetadata> channels = {
        siab::AuxiliaryChannelMetadata{0, 0, 0, 0, 0, "H0_l0_n0_m0"}};
    const std::vector<std::vector<Complex>> basis = {{{1.0, 0.0}, {0.0, 0.0}}};
    const std::vector<siab::FixedAOSpinInput> spins = {
        siab::FixedAOSpinInput{0, {-1.0}, {1.0}, {{-1.0, 0.0}}}};

    EXPECT_THROW(
        siab::build_primitive_galerkin_data(
            blocks, channels, basis, basis, spins, {}, {{1.0, 1.0}}, {0.2}, {0.5}, 0.5),
        std::invalid_argument);
}
