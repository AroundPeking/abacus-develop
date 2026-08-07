#include "source_lcao/module_ri/sternheimer_abfs_perturbation.h"
#include "source_lcao/module_ri/sternheimer_abacus_st_smoke.h"

#include <gtest/gtest.h>

#include <cstdlib>

TEST(SternheimerABFSPerturbation, SamplesSChannelOnFDGrid)
{
    ModuleRI::SternheimerRadialPerturbation radial;
    radial.angular_momentum = 0;
    radial.radial_index = 0;
    radial.label = "toy_s";
    radial.radial_grid = {0.0, 1.0};
    radial.radial_values = {2.0, 4.0};

    ModuleRI::SternheimerFDHamiltonian::Grid grid;
    grid.nx = 2;
    grid.ny = 1;
    grid.nz = 1;
    grid.hx = 1.0;
    grid.hy = 1.0;
    grid.hz = 1.0;
    grid.periodic = false;

    const std::vector<std::vector<ModuleRI::SternheimerRadialPerturbation>> radials_by_type = {{radial}};
    const std::vector<int> atom_types = {0};
    const std::vector<ModuleBase::Vector3<double>> atom_positions = {ModuleBase::Vector3<double>(0.0, 0.0, 0.0)};

    const std::vector<ModuleRI::SternheimerABFGridChannel> channels
        = ModuleRI::sample_sternheimer_abf_grid_channels(radials_by_type, atom_types, atom_positions, grid, 1);

    ASSERT_EQ(channels.size(), 1);
    EXPECT_EQ(channels[0].atom_index, 0);
    EXPECT_EQ(channels[0].atom_local_index, 0);
    EXPECT_EQ(channels[0].angular_momentum, 0);
    EXPECT_EQ(channels[0].magnetic_index, 0);
    EXPECT_EQ(channels[0].label, "toy_s");
    ASSERT_EQ(channels[0].potential_r.size(), 2);

    constexpr double y00 = 0.28209479177387814347;
    EXPECT_NEAR(channels[0].potential_r[0], 2.0 * y00, 1.0e-14);
    EXPECT_NEAR(channels[0].potential_r[1], 4.0 * y00, 1.0e-14);
    EXPECT_NEAR(channels[0].max_abs, 4.0 * y00, 1.0e-14);
}

TEST(SternheimerABFSPerturbation, MapsRealHarmonicIndexToPhysicalMagneticQuantumNumber)
{
    EXPECT_EQ(ModuleRI::sternheimer_physical_magnetic_index(0, 0), 0);
    EXPECT_EQ(ModuleRI::sternheimer_physical_magnetic_index(1, 0), -1);
    EXPECT_EQ(ModuleRI::sternheimer_physical_magnetic_index(1, 1), 0);
    EXPECT_EQ(ModuleRI::sternheimer_physical_magnetic_index(1, 2), 1);
    EXPECT_EQ(ModuleRI::sternheimer_physical_magnetic_index(2, 0), -2);
    EXPECT_EQ(ModuleRI::sternheimer_physical_magnetic_index(2, 4), 2);
    EXPECT_THROW(ModuleRI::sternheimer_physical_magnetic_index(-1, 0), std::invalid_argument);
    EXPECT_THROW(ModuleRI::sternheimer_physical_magnetic_index(1, -1), std::invalid_argument);
    EXPECT_THROW(ModuleRI::sternheimer_physical_magnetic_index(1, 3), std::invalid_argument);
}

TEST(SternheimerABFSPerturbation, StreamsRawChannelsIntoLinearCombinations)
{
    ModuleRI::SternheimerRadialPerturbation radial0;
    radial0.angular_momentum = 0;
    radial0.radial_index = 0;
    radial0.label = "toy_s0";
    radial0.radial_grid = {0.0, 1.0};
    radial0.radial_values = {2.0, 4.0};
    ModuleRI::SternheimerRadialPerturbation radial1 = radial0;
    radial1.radial_index = 1;
    radial1.label = "toy_s1";
    radial1.radial_values = {-1.0, 3.0};
    ModuleRI::SternheimerRadialPerturbation radial2 = radial0;
    radial2.angular_momentum = 1;
    radial2.radial_index = 0;
    radial2.label = "toy_p0";
    radial2.radial_values = {0.5, -2.0};

    ModuleRI::SternheimerFDHamiltonian::Grid grid;
    grid.nx = 1025;
    grid.ny = 1;
    grid.nz = 1;
    grid.hx = 0.001;
    grid.hy = 1.0;
    grid.hz = 1.0;
    grid.periodic = false;

    const std::vector<std::vector<ModuleRI::SternheimerRadialPerturbation>> radials_by_type
        = {{radial0, radial1, radial2}};
    const std::vector<int> atom_types = {0};
    const std::vector<ModuleBase::Vector3<double>> atom_positions
        = {ModuleBase::Vector3<double>(0.0, 0.0, 0.0)};
    const auto dense
        = ModuleRI::sample_sternheimer_abf_grid_channels(radials_by_type, atom_types, atom_positions, grid);
    auto metadata
        = ModuleRI::describe_sternheimer_abf_grid_channels(radials_by_type, atom_types, atom_positions);

    ASSERT_EQ(metadata.size(), dense.size());
    ASSERT_EQ(metadata.size(), 5);
    EXPECT_TRUE(metadata[0].potential_r.empty());
    EXPECT_TRUE(metadata[1].potential_r.empty());
    EXPECT_EQ(metadata[0].label, dense[0].label);
    EXPECT_EQ(metadata[1].channel_index, dense[1].channel_index);

    // Row-major raw-to-output transform: two s and three p channels form three outputs.
    const std::vector<double> transform = {
        1.0, 0.5, -2.0,
        3.0, -1.0, 0.25,
        0.7, -0.2, 0.4,
        -0.3, 1.2, 0.8,
        0.1, 0.6, -0.9,
    };
    const auto streamed = ModuleRI::sample_sternheimer_abf_grid_channel_transform(radials_by_type,
                                                                                   atom_types,
                                                                                   atom_positions,
                                                                                   grid,
                                                                                   metadata,
                                                                                   transform,
                                                                                   3);
    ASSERT_EQ(streamed.size(), 3);
    ASSERT_EQ(dense[0].potential_r.size(), 1025);
    for (std::size_t raw = 0; raw != dense.size(); ++raw)
    {
        EXPECT_NEAR(metadata[raw].max_abs, dense[raw].max_abs, 1.0e-14);
    }
    for (int output = 0; output != 3; ++output)
    {
        ASSERT_EQ(streamed[static_cast<std::size_t>(output)].size(), dense[0].potential_r.size());
        for (std::size_t ir = 0; ir != dense[0].potential_r.size(); ++ir)
        {
            double expected = 0.0;
            for (std::size_t raw = 0; raw != dense.size(); ++raw)
            {
                expected += dense[raw].potential_r[ir]
                            * transform[raw * 3 + static_cast<std::size_t>(output)];
            }
            EXPECT_NEAR(streamed[static_cast<std::size_t>(output)][ir], expected, 1.0e-14);
        }
    }
}

TEST(SternheimerABFSPerturbation, ReadsABFSDiagnosticOnlyEnvironmentFlag)
{
    unsetenv("ABACUS_STERNHEIMER_FD_ST_ABFS_DIAG_ONLY");
    EXPECT_FALSE(ModuleRI::sternheimer_abfs_diag_only_enabled());

    setenv("ABACUS_STERNHEIMER_FD_ST_ABFS_DIAG_ONLY", "1", 1);
    EXPECT_TRUE(ModuleRI::sternheimer_abfs_diag_only_enabled());

    setenv("ABACUS_STERNHEIMER_FD_ST_ABFS_DIAG_ONLY", "false", 1);
    EXPECT_FALSE(ModuleRI::sternheimer_abfs_diag_only_enabled());

    unsetenv("ABACUS_STERNHEIMER_FD_ST_ABFS_DIAG_ONLY");
}
