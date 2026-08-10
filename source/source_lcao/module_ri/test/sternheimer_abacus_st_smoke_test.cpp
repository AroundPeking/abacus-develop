#include "source_lcao/module_ri/sternheimer_abacus_st_smoke.h"

#include "source_lcao/module_ri/sternheimer_abfs_perturbation.h"
#include "source_lcao/module_ri/sternheimer_channel_parallel.h"
#include "source_lcao/module_ri/sternheimer_delta.h"

#include <array>
#include <atomic>
#include <chrono>
#include <complex>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

TEST(SternheimerABACUSSTSmoke, UsesProductionDefaults)
{
    EXPECT_DOUBLE_EQ(ModuleRI::default_sternheimer_solver_tolerance(), 1.0e-6);
    EXPECT_EQ(ModuleRI::parse_sternheimer_lcao_virtual_source(""),
              ModuleRI::SternheimerLCAOVirtualSource::KSBands);
}

TEST(SternheimerABACUSSTSmoke, MovesSampledPotentialsOutOfChannelStorage)
{
    std::vector<ModuleRI::SternheimerABFGridChannel> channels(2);
    channels[0].potential_r = {1.0, 2.0};
    channels[1].potential_r = {3.0};

    const auto potentials = ModuleRI::take_sternheimer_channel_potentials(channels);

    EXPECT_EQ(potentials, (std::vector<std::vector<double>>{{1.0, 2.0}, {3.0}}));
    EXPECT_TRUE(channels[0].potential_r.empty());
    EXPECT_TRUE(channels[1].potential_r.empty());
}

TEST(SternheimerABACUSSTSmoke, AllocatesOnlyValuesForGalerkinSidecars)
{
    ModuleRI::SternheimerDeltaGridFunction function;

    ModuleRI::allocate_sternheimer_grid_function_storage(function, 7, false);

    EXPECT_EQ(function.values.size(), 7U);
    for (const auto& gradient: function.gradients)
    {
        EXPECT_TRUE(gradient.empty());
    }
}

TEST(SternheimerABACUSSTSmoke, MovesGridValuesAndReleasesGradients)
{
    std::vector<ModuleRI::SternheimerDeltaGridFunction> functions(2);
    for (auto& function: functions)
    {
        ModuleRI::allocate_sternheimer_grid_function_storage(function, 3, true);
    }
    functions[0].values = {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}};
    functions[1].values = {{4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0}};
    const auto* first_values = functions[0].values.data();
    const auto* second_values = functions[1].values.data();

    const auto values = ModuleRI::take_sternheimer_grid_values(functions);

    ASSERT_EQ(values.size(), 2U);
    EXPECT_EQ(values[0].data(), first_values);
    EXPECT_EQ(values[1].data(), second_values);
    for (const auto& function: functions)
    {
        EXPECT_TRUE(function.values.empty());
        for (const auto& gradient: function.gradients)
        {
            EXPECT_TRUE(gradient.empty());
            EXPECT_EQ(gradient.capacity(), 0U);
        }
    }
}

TEST(SternheimerChannelParallel, ExecutesConcurrentlyAndReturnsChannelOrder)
{
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    const int previous_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    omp_set_num_threads(4);
#endif

    std::atomic<int> active_workers{0};
    std::atomic<int> peak_workers{0};
    const std::vector<int> results
        = ModuleRI::run_sternheimer_channel_tasks<int>(32, [&active_workers, &peak_workers](const int channel_index) {
              const int active = active_workers.fetch_add(1) + 1;
              int observed_peak = peak_workers.load();
              while (active > observed_peak && !peak_workers.compare_exchange_weak(observed_peak, active))
              {
              }
              std::this_thread::sleep_for(std::chrono::milliseconds(2));
              active_workers.fetch_sub(1);
              return channel_index * channel_index;
          });

#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
#endif
    ASSERT_EQ(results.size(), 32U);
    for (int channel_index = 0; channel_index != 32; ++channel_index)
    {
        EXPECT_EQ(results[static_cast<std::size_t>(channel_index)], channel_index * channel_index);
    }
#ifdef _OPENMP
    EXPECT_GT(peak_workers.load(), 1);
#else
    EXPECT_EQ(peak_workers.load(), 1);
#endif
}

TEST(SternheimerChannelParallel, HonorsExplicitMaximumWorkerCount)
{
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    const int previous_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    omp_set_num_threads(4);
#endif

    std::atomic<int> active_workers{0};
    std::atomic<int> peak_workers{0};
    const std::vector<int> results
        = ModuleRI::run_sternheimer_channel_tasks<int>(
            32,
            [&active_workers, &peak_workers](const int channel_index) {
                const int active = active_workers.fetch_add(1) + 1;
                int observed_peak = peak_workers.load();
                while (active > observed_peak && !peak_workers.compare_exchange_weak(observed_peak, active))
                {
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                active_workers.fetch_sub(1);
                return channel_index;
            },
            2);

#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
#endif
    ASSERT_EQ(results.size(), 32U);
    for (int channel_index = 0; channel_index != 32; ++channel_index)
    {
        EXPECT_EQ(results[static_cast<std::size_t>(channel_index)], channel_index);
    }
#ifdef _OPENMP
    EXPECT_EQ(peak_workers.load(), 2);
#else
    EXPECT_EQ(peak_workers.load(), 1);
#endif
}

TEST(SternheimerChannelParallel, SingleWorkerPreservesNestedGridTeam)
{
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    const int previous_dynamic = omp_get_dynamic();
    const int previous_active_levels = omp_get_max_active_levels();
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    omp_set_max_active_levels(1);
#endif

    const std::vector<int> nested_team_sizes = ModuleRI::run_sternheimer_channel_tasks<int>(
        4,
        [](const int) {
#ifdef _OPENMP
            int nested_team_size = 0;
#pragma omp parallel
            {
#pragma omp single
                nested_team_size = omp_get_num_threads();
            }
            return nested_team_size;
#else
            return 1;
#endif
        },
        1);

#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
    omp_set_max_active_levels(previous_active_levels);
#endif
    ASSERT_EQ(nested_team_sizes.size(), 4U);
    for (const int nested_team_size: nested_team_sizes)
    {
#ifdef _OPENMP
        EXPECT_EQ(nested_team_size, 4);
#else
        EXPECT_EQ(nested_team_size, 1);
#endif
    }
}

TEST(SternheimerChannelParallel, RethrowsFirstIndexedExceptionAfterAllTasksFinish)
{
    std::atomic<int> completed_tasks{0};
    try
    {
        static_cast<void>(ModuleRI::run_sternheimer_channel_tasks<int>(16, [&completed_tasks](const int channel_index) {
            completed_tasks.fetch_add(1);
            if (channel_index == 3 || channel_index == 9)
            {
                throw std::runtime_error("channel " + std::to_string(channel_index));
            }
            return channel_index;
        }));
        FAIL() << "Expected a channel task exception.";
    }
    catch (const std::runtime_error& error)
    {
        EXPECT_STREQ(error.what(), "channel 3");
    }
    EXPECT_EQ(completed_tasks.load(), 16);

    EXPECT_THROW(ModuleRI::run_sternheimer_channel_tasks<int>(-1, [](const int) { return 0; }), std::invalid_argument);
    EXPECT_THROW(ModuleRI::run_sternheimer_channel_tasks<int>(1, [](const int) { return 0; }, -1),
                 std::invalid_argument);
}

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

TEST(SternheimerABACUSSTSmoke, RequiresCompleteKSVirtualSubspace)
{
    EXPECT_EQ(ModuleRI::expected_sternheimer_ks_virtual_states(21, 0), 21);
    EXPECT_EQ(ModuleRI::expected_sternheimer_ks_virtual_states(21, 16), 16);
    EXPECT_NO_THROW(ModuleRI::validate_sternheimer_ks_virtual_subspace(1, 21, 21, 21));
    EXPECT_THROW(ModuleRI::validate_sternheimer_ks_virtual_subspace(2, 21, 20, 20),
                 std::runtime_error);
    EXPECT_THROW(ModuleRI::validate_sternheimer_ks_virtual_subspace(2, 21, 21, 20),
                 std::runtime_error);
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

TEST(SternheimerABACUSSTSmoke, RequiresCompleteFixedAOMatricesForEverySpin)
{
    ModuleRI::SternheimerLCAOFixedAOMatrices matrices;
    matrices.n_basis = 2;
    matrices.overlap_s = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
    matrices.spins = {
        module_ri::sternheimer_siab::FixedAOSpinInput{
            0, {-1.0, 1.0}, {1.0, 0.0}, {{-1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}}},
        module_ri::sternheimer_siab::FixedAOSpinInput{
            1, {-0.9, 1.1}, {1.0, 0.0}, {{-0.9, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.1, 0.0}}},
    };

    EXPECT_NO_THROW(ModuleRI::validate_sternheimer_lcao_fixed_ao_matrices(matrices, 2, 2));

    matrices.spins.pop_back();
    EXPECT_THROW(ModuleRI::validate_sternheimer_lcao_fixed_ao_matrices(matrices, 2, 2), std::invalid_argument);
    matrices.spins.push_back(matrices.spins.front());
    EXPECT_THROW(ModuleRI::validate_sternheimer_lcao_fixed_ao_matrices(matrices, 2, 2), std::invalid_argument);
}

TEST(SternheimerABACUSSTSmoke, SelectsZeroOrderSourceByResponseMode)
{
    EXPECT_FALSE(ModuleRI::sternheimer_uses_lcao_zero_order(false));
    EXPECT_TRUE(ModuleRI::sternheimer_uses_lcao_zero_order(true));
}

TEST(SternheimerABACUSSTSmoke, SelectsIndependentFixedAOGalerkinOutputMode)
{
    const auto disabled = ModuleRI::select_sternheimer_output_mode(false, false, false, false, false);
    EXPECT_FALSE(disabled.run);
    EXPECT_FALSE(disabled.write_fixed_ao);
    EXPECT_FALSE(disabled.fixed_ao_only);

    const auto fixed_only = ModuleRI::select_sternheimer_output_mode(false, false, true, false, false);
    EXPECT_TRUE(fixed_only.run);
    EXPECT_FALSE(fixed_only.write_librpa);
    EXPECT_FALSE(fixed_only.write_siab_targets);
    EXPECT_TRUE(fixed_only.write_fixed_ao);
    EXPECT_TRUE(fixed_only.fixed_ao_only);

    const auto librpa_only = ModuleRI::select_sternheimer_output_mode(true, false, false, false, false);
    EXPECT_TRUE(librpa_only.run);
    EXPECT_TRUE(librpa_only.write_librpa);
    EXPECT_FALSE(librpa_only.write_fixed_ao);
    EXPECT_FALSE(librpa_only.fixed_ao_only);

    const auto librpa_with_fixed = ModuleRI::select_sternheimer_output_mode(true, false, true, false, false);
    EXPECT_TRUE(librpa_with_fixed.run);
    EXPECT_TRUE(librpa_with_fixed.write_librpa);
    EXPECT_TRUE(librpa_with_fixed.write_fixed_ao);
    EXPECT_FALSE(librpa_with_fixed.fixed_ao_only);

    const auto siab_only = ModuleRI::select_sternheimer_output_mode(false, true, false, false, false);
    EXPECT_TRUE(siab_only.run);
    EXPECT_FALSE(siab_only.write_librpa);
    EXPECT_TRUE(siab_only.write_siab_targets);
    EXPECT_FALSE(siab_only.write_fixed_ao);
    EXPECT_FALSE(siab_only.fixed_ao_only);

    const auto primitive_only
        = ModuleRI::select_sternheimer_output_mode(false, false, false, true, false);
    EXPECT_TRUE(primitive_only.run);
    EXPECT_FALSE(primitive_only.write_librpa);
    EXPECT_FALSE(primitive_only.write_siab_targets);
    EXPECT_TRUE(primitive_only.write_fixed_ao);
    EXPECT_TRUE(primitive_only.write_primitive);
    EXPECT_TRUE(primitive_only.fixed_ao_only);

    const auto response_only
        = ModuleRI::select_sternheimer_output_mode(false, false, false, false, true);
    EXPECT_TRUE(response_only.run);
    EXPECT_FALSE(response_only.write_librpa);
    EXPECT_FALSE(response_only.write_siab_targets);
    EXPECT_TRUE(response_only.write_fixed_ao);
    EXPECT_FALSE(response_only.write_primitive);
    EXPECT_TRUE(response_only.write_response_orbitals);
    EXPECT_TRUE(response_only.fixed_ao_only);
}

TEST(SternheimerABACUSSTSmoke, ReordersResponseOrbitalsFromLNMToLMNBlocks)
{
    const std::vector<ModuleRI::SternheimerResponseOrbitalAtomSpec> atoms{
        {"H", 0, {2, 2}},
        {"H", 1, {2, 2}},
    };
    const ModuleRI::SternheimerResponseOrbitalLayout layout
        = ModuleRI::build_sternheimer_response_orbital_layout(atoms);

    EXPECT_EQ(layout.sampled_indices,
              (std::vector<std::size_t>{0, 1, 2, 5, 3, 6, 4, 7,
                                        8, 9, 10, 13, 11, 14, 12, 15}));
    ASSERT_EQ(layout.blocks.size(), 8);
    EXPECT_EQ(layout.blocks[0].atom_index, 0);
    EXPECT_EQ(layout.blocks[0].l, 0);
    EXPECT_EQ(layout.blocks[0].m, 0);
    EXPECT_EQ(layout.blocks[0].n_primitive, 2);
    EXPECT_EQ(layout.blocks[0].offset, 0);
    EXPECT_EQ(layout.blocks[1].m, -1);
    EXPECT_EQ(layout.blocks[1].offset, 2);
    EXPECT_EQ(layout.blocks[3].m, 1);
    EXPECT_EQ(layout.blocks[3].offset, 6);
    EXPECT_EQ(layout.blocks[4].atom_index, 1);
    EXPECT_EQ(layout.blocks[4].offset, 8);
    EXPECT_EQ(layout.blocks[7].m, 1);
    EXPECT_EQ(layout.blocks[7].offset, 14);
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
