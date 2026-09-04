#include "../librpa_thermal_metadata.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
void require(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

void require_near(const double actual, const double expected, const double tolerance, const char* message)
{
    require(std::abs(actual - expected) <= tolerance, message);
}

void test_kelvin_to_rydberg_is_physical_kbt()
{
    constexpr double temperature_kelvin = 300.0;
    constexpr double expected_rydberg = 300.0 * 6.333623126e-6;
    require_near(RpaLriDetail::kelvin_to_rydberg(temperature_kelvin),
                 expected_rydberg,
                 1.0e-15,
                 "Kelvin input must convert to k_B T in Ry without a factor of two");
}

void test_fermi_dirac_aliases()
{
    require(RpaLriDetail::is_fermi_dirac_smearing("fd"), "fd alias must be accepted");
    require(RpaLriDetail::is_fermi_dirac_smearing("fermi-dirac"), "fermi-dirac alias must be accepted");
    require(!RpaLriDetail::is_fermi_dirac_smearing("gauss"), "Gaussian smearing must not be marked thermal");
    require(!RpaLriDetail::is_fermi_dirac_smearing("mp"), "Methfessel-Paxton smearing must not be marked thermal");
}

void test_metadata_serialization()
{
    RpaLriDetail::LibrpaThermalMetadata metadata;
    metadata.chemical_potential_ha = 0.125;
    metadata.kbt_ha = 0.0025;
    metadata.smearing_sigma_ry = 0.005;
    metadata.max_occupation_per_band = 2.0;
    metadata.spin_channels = 1;
    metadata.kpoints_per_spin = 8;
    metadata.bands = 12;

    const std::string text = RpaLriDetail::serialize_librpa_thermal_metadata(metadata);
    require(text.find("format thermal_occupation_v1\n") != std::string::npos, "metadata format marker is missing");
    require(text.find("occupation_model fermi_dirac\n") != std::string::npos, "occupation model is missing");
    require(text.find("chemical_potential_ha 1.25000000000000000e-01\n") != std::string::npos,
            "chemical potential must use deterministic precision");
    require(text.find("kbt_ha 2.50000000000000005e-03\n") != std::string::npos, "kBT field is missing");
    require(text.find("smearing_sigma_ry 5.00000000000000010e-03\n") != std::string::npos,
            "Ry smearing field is missing");
    require(text.find("occupation_storage band_out_times_nk\n") != std::string::npos,
            "occupation storage convention is missing");
    require(text.find("max_occupation_per_band 2.00000000000000000e+00\n") != std::string::npos,
            "spin normalization is missing");
    require(text.find("spin_channels 1\n") != std::string::npos, "spin channel count is missing");
    require(text.find("kpoints_per_spin 8\n") != std::string::npos, "k-point count is missing");
    require(text.find("bands 12\n") != std::string::npos, "band count is missing");
}

void test_invalid_metadata_is_rejected()
{
    RpaLriDetail::LibrpaThermalMetadata metadata;
    metadata.chemical_potential_ha = 0.0;
    metadata.kbt_ha = 0.01;
    metadata.smearing_sigma_ry = 0.02;
    metadata.max_occupation_per_band = 1.0;
    metadata.spin_channels = 2;
    metadata.kpoints_per_spin = 4;
    metadata.bands = 6;

    for (const double invalid_kbt:
         {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        auto invalid = metadata;
        invalid.kbt_ha = invalid_kbt;
        bool rejected = false;
        try
        {
            (void)RpaLriDetail::serialize_librpa_thermal_metadata(invalid);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        require(rejected, "nonpositive or nonfinite kBT must be rejected");
    }

    auto inconsistent = metadata;
    inconsistent.smearing_sigma_ry = 0.03;
    bool rejected = false;
    try
    {
        (void)RpaLriDetail::serialize_librpa_thermal_metadata(inconsistent);
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    require(rejected, "Ha and Ry thermal energies must be consistent");
}

void test_writer_removes_stale_non_fd_sidecar()
{
    const std::string path = "librpa_thermal_metadata_test.dat";
    RpaLriDetail::LibrpaThermalMetadata metadata;
    metadata.chemical_potential_ha = 0.125;
    metadata.kbt_ha = 0.0025;
    metadata.smearing_sigma_ry = 0.005;
    metadata.max_occupation_per_band = 2.0;
    metadata.spin_channels = 1;
    metadata.kpoints_per_spin = 8;
    metadata.bands = 12;

    require(RpaLriDetail::write_librpa_thermal_metadata(path, "fd", metadata), "FD metadata must be written");
    std::ifstream written(path.c_str());
    require(written.good(), "FD metadata file is missing");
    written.close();

    require(!RpaLriDetail::write_librpa_thermal_metadata(path, "gauss", metadata),
            "non-FD metadata must not be written");
    std::ifstream removed(path.c_str());
    require(!removed.good(), "a stale non-FD thermal sidecar must be removed");
}
} // namespace

int main()
{
    test_kelvin_to_rydberg_is_physical_kbt();
    test_fermi_dirac_aliases();
    test_metadata_serialization();
    test_invalid_metadata_is_rejected();
    test_writer_removes_stale_non_fd_sidecar();
    std::cout << "LibRPA thermal metadata tests passed\n";
    return 0;
}
