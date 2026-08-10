#ifndef STERNHEIMER_SIAB_DATA_H
#define STERNHEIMER_SIAB_DATA_H

#include <complex>
#include <string>
#include <vector>

namespace module_ri
{
namespace sternheimer_siab
{

struct PrimitiveBlock
{
    std::string element;
    int atom_index;
    int l;
    int m;
    int n_primitive;
    int offset;
};

struct ReferenceRow
{
    int occupied_state;
    int auxiliary_channel;
    double frequency_ha;
    double occupation;
    double frequency_weight;
    double norm;
    std::vector<std::complex<double>> q;
    int frequency_index;
};

struct AuxiliaryChannelMetadata
{
    int channel_index;
    int atom_index;
    int angular_momentum;
    int radial_index;
    int magnetic_index;
    std::string label;
};

struct FixedAOSpinData
{
    int spin_index;
    std::vector<double> eigenvalues_ha;
    std::vector<double> occupations;
    std::vector<std::complex<double>> hamiltonian_ha;
};

struct FixedAOData
{
    int n_basis;
    std::vector<FixedAOSpinData> spins;
    std::vector<AuxiliaryChannelMetadata> auxiliary_channels;
    std::vector<std::complex<double>> overlap_s;
    std::vector<std::vector<std::complex<double>>> perturbations_ha;
    std::vector<double> frequency_ha;
    std::vector<double> frequency_weights_ha;
};

struct PrimitiveGalerkinSpinData
{
    int spin_index;
    std::vector<double> fixed_ao_occupations;
    std::vector<std::complex<double>> hamiltonian_ha;
    std::vector<std::complex<double>> fixed_ao_grid_hamiltonian_ha;
    std::vector<std::complex<double>> primitive_ao_hamiltonian_ha;
};

struct PrimitiveGalerkinData
{
    int n_primitive;
    int n_fixed_ao;
    std::vector<PrimitiveBlock> blocks;
    std::vector<PrimitiveGalerkinSpinData> spins;
    std::vector<AuxiliaryChannelMetadata> auxiliary_channels;
    std::vector<std::complex<double>> overlap_s;
    std::vector<std::vector<std::complex<double>>> perturbations_ha;
    std::vector<std::complex<double>> primitive_ao_overlap;
    std::vector<std::complex<double>> fixed_ao_grid_overlap;
    std::vector<std::vector<std::complex<double>>> primitive_ao_perturbations_ha;
    std::vector<double> frequency_ha;
    std::vector<double> frequency_weights_ha;
};

// The eight fields below are required-core v1; Task 4 may append deterministic optional provenance fields.
struct Provenance
{
    std::string abacus_commit;
    std::string auxiliary_basis_sha256;
    std::vector<double> cell_bohr; ///< complete 3x3 row-major lattice vectors in Bohr
    double ecut_ry;
    std::string kernel;
    std::string orbital_sha256;
    std::string response_orbital_sha256;
    std::string pseudopotential_sha256;
    std::string spin_convention;

    // Task 4 production fields.  Keeping these unset preserves the exact Task 3 canonical fixture.
    std::string executable_sha256;
    double exx_pca_thr = -1.0;
    int sternheimer_nfreq = 0;
    std::vector<double> frequency_ha;
    std::vector<double> frequency_weights_ha;
    int mpi_ranks = 0;
    int omp_threads = 0;

    // Global full-Coulomb whitening fields for SIAB-only production targets.
    std::string auxiliary_whitening;
    int raw_auxiliary_dimension = 0;
    int whitened_auxiliary_rank = 0;
    int discarded_auxiliary_rank = 0;
    double coulomb_relative_threshold = -1.0;
    std::vector<double> coulomb_eigenvalues;
    double coulomb_max_orthonormality_error = -1.0;
    std::string coulomb_transform_sha256;
};

} // namespace sternheimer_siab
} // namespace module_ri

#endif
