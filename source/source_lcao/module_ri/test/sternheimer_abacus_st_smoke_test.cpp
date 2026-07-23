#include "source_lcao/module_ri/sternheimer_abacus_st_smoke.h"

#include <array>
#include <complex>
#include <gtest/gtest.h>

TEST(SternheimerABACUSSTSmoke, FormatsLinearResponseReport)
{
    ModuleRI::SternheimerABACUSSTSmokeResult result;
    result.grid_data.grid.nx = 2;
    result.grid_data.grid.ny = 3;
    result.grid_data.grid.nz = 4;
    result.grid_data.volume_element = 0.125;
    result.omega = 0.5;
    result.pca_threshold = 1.0e-4;
    result.ccp_rmesh_times = 10.0;
    result.perturbation_source = "abfs_ccp";
    result.num_available_channels = 5;

    ModuleRI::SternheimerABACUSSTChannelResult channel;
    channel.band_index = 0;
    channel.channel_index = 1;
    channel.atom_index = 0;
    channel.angular_momentum = 1;
    channel.radial_index = 2;
    channel.magnetic_index = 1;
    channel.fd_eigenvalue = -1.25;
    channel.occupation = 2.0;
    channel.rhs_norm = 0.4;
    channel.projected_rhs_norm = 0.3;
    channel.solver_converged = true;
    channel.solver_iterations = 7;
    channel.solver_relative_residual = 1.0e-9;
    channel.equation_residual_norm = 2.0e-9;
    channel.polarizability = std::complex<double>(-0.12, 0.03);
    result.channels.push_back(channel);

    const std::string report = ModuleRI::format_sternheimer_abacus_st_report(result);

    EXPECT_NE(report.find("# ABACUS Sternheimer FD linear-response smoke test"), std::string::npos);
    EXPECT_NE(report.find("grid 2 3 4 size 24 dV 0.125"), std::string::npos);
    EXPECT_NE(report.find("omega_Ry 0.5"), std::string::npos);
    EXPECT_NE(report.find("pca_threshold 0.0001"), std::string::npos);
    EXPECT_NE(report.find("perturbation_source abfs_ccp"), std::string::npos);
    EXPECT_NE(report.find("available_channels 5"), std::string::npos);
    EXPECT_NE(report.find("0 1 0 1 2 1 -1.25 2 0.4 0.3 yes 7 1e-09 2e-09 -0.12 0.03"),
              std::string::npos);
}

TEST(SternheimerABACUSSTSmoke, ValidatesSpinResolvedLCAOOccupiedChannels)
{
    ModuleRI::SternheimerLCAOOccupiedChannel spin_up;
    spin_up.spin_index = 0;
    spin_up.coefficients = {{std::complex<double>(1.0, 0.0),
                             std::complex<double>(0.0, 0.0),
                             std::complex<double>(0.0, 0.0)}};

    const std::vector<ModuleRI::SternheimerLCAOOccupiedChannel> channels = {spin_up};
    EXPECT_NO_THROW(ModuleRI::validate_sternheimer_lcao_occupied_channels(channels, 2, 3));
    EXPECT_EQ(ModuleRI::sternheimer_lcao_total_occupied_bands(channels), 1);

    spin_up.coefficients.resize(4, spin_up.coefficients.front());
    ModuleRI::SternheimerLCAOOccupiedChannel spin_down;
    spin_down.spin_index = 1;
    spin_down.coefficients = {{std::complex<double>(0.0, 0.0),
                               std::complex<double>(1.0, 0.0),
                               std::complex<double>(0.0, 0.0)}};
    const std::vector<ModuleRI::SternheimerLCAOOccupiedChannel> quartet = {spin_up, spin_down};

    EXPECT_NO_THROW(ModuleRI::validate_sternheimer_lcao_occupied_channels(quartet, 2, 3));
    EXPECT_EQ(ModuleRI::sternheimer_lcao_spin_indices(quartet), (std::vector<int>{0, 1}));
    EXPECT_EQ(ModuleRI::sternheimer_lcao_occupied_bands_per_spin(quartet), (std::vector<int>{4, 1}));
    EXPECT_EQ(ModuleRI::sternheimer_lcao_total_occupied_bands(quartet), 5);

    auto duplicate_channels = channels;
    duplicate_channels.push_back(spin_up);
    EXPECT_THROW(ModuleRI::validate_sternheimer_lcao_occupied_channels(duplicate_channels, 2, 3),
                 std::invalid_argument);

    auto out_of_range_channels = channels;
    out_of_range_channels.front().spin_index = 2;
    EXPECT_THROW(ModuleRI::validate_sternheimer_lcao_occupied_channels(out_of_range_channels, 2, 3),
                 std::invalid_argument);
}

TEST(SternheimerABACUSSTSmoke, SelectsExplicitKSVirtualBandsFromOccupations)
{
    ModuleRI::SternheimerLCAOOccupiedChannel spin_up;
    spin_up.spin_index = 0;
    spin_up.coefficients = {{std::complex<double>(1.0, 0.0),
                             std::complex<double>(0.0, 0.0),
                             std::complex<double>(0.0, 0.0)}};
    spin_up.unoccupied_coefficients = {
        {std::complex<double>(0.0, 0.0), std::complex<double>(1.0, 0.0), std::complex<double>(0.0, 0.0)},
        {std::complex<double>(0.0, 0.0), std::complex<double>(0.0, 0.0), std::complex<double>(1.0, 0.0)}};

    const std::vector<ModuleRI::SternheimerLCAOOccupiedChannel> channels = {spin_up};
    EXPECT_NO_THROW(ModuleRI::validate_sternheimer_lcao_occupied_channels(channels, 1, 3));
    EXPECT_EQ(ModuleRI::sternheimer_lcao_unoccupied_bands_per_spin(channels), (std::vector<int>{2}));
    EXPECT_EQ(ModuleRI::sternheimer_lcao_total_unoccupied_bands(channels), 2);

    EXPECT_EQ(ModuleRI::parse_sternheimer_lcao_virtual_source("projected_ao"),
              ModuleRI::SternheimerLCAOVirtualSource::ProjectedAO);
    EXPECT_EQ(ModuleRI::parse_sternheimer_lcao_virtual_source("ks_bands"),
              ModuleRI::SternheimerLCAOVirtualSource::KSBands);
    EXPECT_EQ(ModuleRI::sternheimer_lcao_virtual_source_name(ModuleRI::SternheimerLCAOVirtualSource::KSBands),
              "ks_bands");
    EXPECT_THROW(ModuleRI::parse_sternheimer_lcao_virtual_source("svd_guess"), std::invalid_argument);

    auto invalid = channels;
    invalid.front().unoccupied_coefficients.front().pop_back();
    EXPECT_THROW(ModuleRI::validate_sternheimer_lcao_occupied_channels(invalid, 1, 3), std::invalid_argument);
}

TEST(SternheimerABACUSSTSmoke, AcceptsOnlyPhysicalGammaSpinRows)
{
    const std::vector<std::array<double, 3>> one_gamma = {{{0.0, 0.0, 0.0}}};
    const std::vector<std::array<double, 3>> two_gamma = {{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};

    EXPECT_NO_THROW(ModuleRI::validate_sternheimer_lcao_gamma_layout(1, 1, 1, one_gamma));
    EXPECT_NO_THROW(ModuleRI::validate_sternheimer_lcao_gamma_layout(2, 2, 2, two_gamma));

    EXPECT_THROW(ModuleRI::validate_sternheimer_lcao_gamma_layout(1, 2, 2, two_gamma), std::invalid_argument);

    const std::vector<std::array<double, 3>> non_gamma = {{{0.25, 0.0, 0.0}}};
    EXPECT_THROW(ModuleRI::validate_sternheimer_lcao_gamma_layout(1, 1, 1, non_gamma), std::invalid_argument);
}

TEST(SternheimerABACUSSTSmoke, SelectsZeroOrderSourceByResponseMode)
{
    EXPECT_FALSE(ModuleRI::sternheimer_uses_lcao_zero_order(false));
    EXPECT_TRUE(ModuleRI::sternheimer_uses_lcao_zero_order(true));
}

TEST(SternheimerABACUSSTSmoke, SelectsExplicitABFSForPerturbationChannels)
{
    EXPECT_EQ(ModuleRI::sternheimer_abfs_perturbation_source({}), "product_pca");
    EXPECT_EQ(ModuleRI::sternheimer_abfs_perturbation_source({"H-fixed.abfs"}), "explicit_abfs");
    EXPECT_TRUE(ModuleRI::sternheimer_builds_product_pca_auxiliary_basis({}));
    EXPECT_FALSE(ModuleRI::sternheimer_builds_product_pca_auxiliary_basis({"H-fixed.abfs"}));
}

TEST(SternheimerABACUSSTSmoke, EstimatesSIABDenseMemoryWithoutRawGridChannels)
{
    const auto estimate = ModuleRI::estimate_sternheimer_siab_dense_memory(100, 4, 3, 5, 7, 2, 6);
    EXPECT_EQ(estimate.coulomb_metric_bytes, 4U * 4U * sizeof(double));
    EXPECT_EQ(estimate.transformed_potential_bytes, 100U * 3U * sizeof(double));
    EXPECT_EQ(estimate.channel_transform_workspace_bytes,
              1024U * (4U + 3U) * sizeof(double) + 1024U * sizeof(std::size_t));
    EXPECT_EQ(estimate.reciprocal_primitive_bytes, 5U * 7U * sizeof(std::complex<double>));
    EXPECT_EQ(estimate.primitive_overlap_bytes, 5U * 5U * sizeof(std::complex<double>));
    const std::uint64_t row_bytes = 7U * sizeof(double) + 5U * sizeof(std::complex<double>);
    EXPECT_EQ(estimate.gathered_reference_row_bytes, 3U * 2U * 3U * 6U * row_bytes);
    EXPECT_EQ(estimate.total_bytes,
              estimate.coulomb_metric_bytes + estimate.transformed_potential_bytes
                  + estimate.channel_transform_workspace_bytes
                  + estimate.reciprocal_primitive_bytes + estimate.primitive_overlap_bytes
                  + estimate.gathered_reference_row_bytes);
}
