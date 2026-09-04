#include "librpa_thermal_metadata.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace RpaLriDetail
{
namespace
{
void validate_librpa_thermal_metadata(const LibrpaThermalMetadata& metadata)
{
    if (!std::isfinite(metadata.chemical_potential_ha))
    {
        throw std::invalid_argument("thermal chemical potential must be finite");
    }
    if (!std::isfinite(metadata.kbt_ha) || metadata.kbt_ha <= 0.0)
    {
        throw std::invalid_argument("thermal kBT must be positive and finite");
    }
    if (!std::isfinite(metadata.smearing_sigma_ry) || metadata.smearing_sigma_ry <= 0.0)
    {
        throw std::invalid_argument("thermal smearing sigma must be positive and finite");
    }
    const double scale = std::max({1.0, metadata.smearing_sigma_ry, 2.0 * metadata.kbt_ha});
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() * scale;
    if (std::abs(metadata.smearing_sigma_ry - 2.0 * metadata.kbt_ha) > tolerance)
    {
        throw std::invalid_argument("thermal Ha and Ry energies are inconsistent");
    }
    if (metadata.max_occupation_per_band != 1.0 && metadata.max_occupation_per_band != 2.0)
    {
        throw std::invalid_argument("maximum occupation per band must be one or two");
    }
    if (metadata.spin_channels <= 0 || metadata.kpoints_per_spin <= 0 || metadata.bands <= 0)
    {
        throw std::invalid_argument("thermal metadata dimensions must be positive");
    }
}
} // namespace

bool is_fermi_dirac_smearing(const std::string& smearing_method)
{
    return smearing_method == "fd" || smearing_method == "fermi-dirac";
}

std::string serialize_librpa_thermal_metadata(const LibrpaThermalMetadata& metadata)
{
    validate_librpa_thermal_metadata(metadata);

    std::ostringstream output;
    output << "format thermal_occupation_v1\n";
    output << "occupation_model fermi_dirac\n";
    output << std::scientific << std::setprecision(17);
    output << "chemical_potential_ha " << metadata.chemical_potential_ha << '\n';
    output << "kbt_ha " << metadata.kbt_ha << '\n';
    output << "smearing_sigma_ry " << metadata.smearing_sigma_ry << '\n';
    output << "occupation_storage band_out_times_nk\n";
    output << "max_occupation_per_band " << metadata.max_occupation_per_band << '\n';
    output << std::defaultfloat;
    output << "spin_channels " << metadata.spin_channels << '\n';
    output << "kpoints_per_spin " << metadata.kpoints_per_spin << '\n';
    output << "bands " << metadata.bands << '\n';
    return output.str();
}

bool write_librpa_thermal_metadata(const std::string& path,
                                   const std::string& smearing_method,
                                   const LibrpaThermalMetadata& metadata)
{
    if (!is_fermi_dirac_smearing(smearing_method))
    {
        errno = 0;
        if (std::remove(path.c_str()) != 0 && errno != ENOENT)
        {
            throw std::runtime_error("failed to remove stale LibRPA thermal metadata: " + path);
        }
        return false;
    }

    std::ofstream output(path.c_str(), std::ios::out | std::ios::trunc);
    if (!output.good())
    {
        throw std::runtime_error("failed to open LibRPA thermal metadata: " + path);
    }
    output << serialize_librpa_thermal_metadata(metadata);
    if (!output.good())
    {
        throw std::runtime_error("failed to write LibRPA thermal metadata: " + path);
    }
    return true;
}

} // namespace RpaLriDetail
