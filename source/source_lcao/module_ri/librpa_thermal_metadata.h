#pragma once

#include "source_base/constants.h"

#include <string>

namespace RpaLriDetail
{

const double kBoltzmannRydbergPerKelvin = 2.0 * ModuleBase::K_BOLTZMAN_AU;

inline double kelvin_to_rydberg(const double temperature_kelvin)
{
    return temperature_kelvin * kBoltzmannRydbergPerKelvin;
}

bool is_fermi_dirac_smearing(const std::string& smearing_method);

struct LibrpaThermalMetadata
{
    double chemical_potential_ha = 0.0;
    double kbt_ha = 0.0;
    double smearing_sigma_ry = 0.0;
    double max_occupation_per_band = 0.0;
    int spin_channels = 0;
    int kpoints_per_spin = 0;
    int bands = 0;
};

std::string serialize_librpa_thermal_metadata(const LibrpaThermalMetadata& metadata);

// Returns true when a thermal sidecar is written. For non-FD smearing, a
// stale sidecar at the same path is removed and false is returned.
bool write_librpa_thermal_metadata(const std::string& path,
                                   const std::string& smearing_method,
                                   const LibrpaThermalMetadata& metadata);

} // namespace RpaLriDetail
