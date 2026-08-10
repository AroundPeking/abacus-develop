#ifndef STERNHEIMER_RESPONSE_SPECTRAL_H
#define STERNHEIMER_RESPONSE_SPECTRAL_H

#include "sternheimer_siab_data.h"

#include <complex>
#include <vector>

namespace module_ri
{
namespace sternheimer_siab
{

struct ResponseSpectralOptions
{
    double relative_rank_tolerance = 1.0e-8;
    double condition_limit = 1.0e12;
    double eigenvalue_tolerance_ha = 1.0e-8;
    double occupation_tolerance = 1.0e-8;
};

struct ResponseSpectralSpinDiagnostics
{
    int spin_index = -1;
    int retained_virtual_rank = 0;
    int dropped_trial_rank = 0;
    double fixed_ao_eigenvalue_max_abs_error_ha = 0.0;
    double occupied_grid_norm_before_normalization = 0.0;
    double projected_overlap_condition = 0.0;
    double occupied_orthonormality_max_abs_error = 0.0;
    double occupied_virtual_max_abs_overlap = 0.0;
    double virtual_orthonormality_max_abs_error = 0.0;
    double minimum_virtual_energy_ha = 0.0;
    double maximum_virtual_energy_ha = 0.0;
};

struct ResponseSpectralResult
{
    std::vector<std::vector<std::complex<double>>> response_m;
    std::vector<ResponseSpectralSpinDiagnostics> spin_diagnostics;
};

ResponseSpectralResult evaluate_response_orbital_spectral_response(
    const PrimitiveGalerkinData& response,
    const FixedAOData& fixed_ao,
    const ResponseSpectralOptions& options = ResponseSpectralOptions());

} // namespace sternheimer_siab
} // namespace module_ri

#endif
