#include "source_lcao/module_ri/sternheimer_siab_overlap.h"
#include "source_lcao/module_ri/sternheimer_siab_source.h"

#include <complex>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

using Complex = std::complex<double>;
namespace siab = module_ri::sternheimer_siab;

const siab::ComplexGrid occupied_wavefunction = {
    {1.0, 2.0},
    {-0.5, 1.0},
};

const std::vector<double> perturbation_ry = {4.0, -2.0};

} // namespace

TEST(SternheimerSIABSource, BuildsSourceGridFromRydbergPotential)
{
    const auto source
        = siab::build_source_grid_from_rydberg_potential(occupied_wavefunction, perturbation_ry);

    ASSERT_EQ(source.size(), 2);
    EXPECT_EQ(source[0], Complex(2.0, 4.0));
    EXPECT_EQ(source[1], Complex(0.5, -1.0));
}

TEST(SternheimerSIABSource, MatchesExistingSourcePrimitiveOverlapConvention)
{
    const auto source
        = siab::build_source_grid_from_rydberg_potential(occupied_wavefunction, perturbation_ry);
    const std::vector<siab::ComplexGrid> primitives = {
        {{1.0, -1.0}, {2.0, 0.5}},
    };
    constexpr double delta_omega = 0.25;

    const auto q = siab::overlap_q(source, primitives, delta_omega);
    const Complex expected
        = delta_omega
          * (std::conj(source[0]) * primitives[0][0]
             + std::conj(source[1]) * primitives[0][1]);

    ASSERT_EQ(q.size(), 1);
    EXPECT_NEAR(q[0].real(), expected.real(), 1.0e-14);
    EXPECT_NEAR(q[0].imag(), expected.imag(), 1.0e-14);
}

TEST(SternheimerSIABSource, HartreeValuesPassedAsRydbergGiveHalfTheIntendedSource)
{
    const auto intended
        = siab::build_source_grid_from_rydberg_potential(occupied_wavefunction, {4.0, -2.0});
    const auto hartree_values_passed_as_rydberg
        = siab::build_source_grid_from_rydberg_potential(occupied_wavefunction, {2.0, -1.0});

    ASSERT_EQ(hartree_values_passed_as_rydberg.size(), intended.size());
    for (std::size_t ir = 0; ir != intended.size(); ++ir)
    {
        EXPECT_EQ(hartree_values_passed_as_rydberg[ir], 0.5 * intended[ir]);
    }
}

TEST(SternheimerSIABSource, RejectsEmptyInput)
{
    EXPECT_THROW(siab::build_source_grid_from_rydberg_potential({}, {}), std::invalid_argument);
}

TEST(SternheimerSIABSource, RejectsSizeMismatch)
{
    EXPECT_THROW(
        siab::build_source_grid_from_rydberg_potential({{1.0, 0.0}}, {1.0, 2.0}),
        std::invalid_argument);
    EXPECT_THROW(
        siab::build_source_grid_from_rydberg_potential({{1.0, 0.0}, {2.0, 0.0}}, {1.0}),
        std::invalid_argument);
}

TEST(SternheimerSIABSource, RejectsNonfiniteRealPotential)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();

    EXPECT_THROW(
        siab::build_source_grid_from_rydberg_potential({{1.0, 0.0}}, {nan}),
        std::invalid_argument);
    EXPECT_THROW(
        siab::build_source_grid_from_rydberg_potential({{1.0, 0.0}}, {infinity}),
        std::invalid_argument);
    EXPECT_THROW(
        siab::build_source_grid_from_rydberg_potential({{1.0, 0.0}}, {-infinity}),
        std::invalid_argument);
}

TEST(SternheimerSIABSource, RejectsNonfiniteWavefunctionComponents)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();

    EXPECT_THROW(
        siab::build_source_grid_from_rydberg_potential({{nan, 0.0}}, {1.0}),
        std::invalid_argument);
    EXPECT_THROW(
        siab::build_source_grid_from_rydberg_potential({{infinity, 0.0}}, {1.0}),
        std::invalid_argument);
    EXPECT_THROW(
        siab::build_source_grid_from_rydberg_potential({{-infinity, 0.0}}, {1.0}),
        std::invalid_argument);
    EXPECT_THROW(
        siab::build_source_grid_from_rydberg_potential({{0.0, nan}}, {1.0}),
        std::invalid_argument);
    EXPECT_THROW(
        siab::build_source_grid_from_rydberg_potential({{0.0, infinity}}, {1.0}),
        std::invalid_argument);
    EXPECT_THROW(
        siab::build_source_grid_from_rydberg_potential({{0.0, -infinity}}, {1.0}),
        std::invalid_argument);
}
