#ifndef STERNHEIMER_BASIS_OPT_PERIODIC_H
#define STERNHEIMER_BASIS_OPT_PERIODIC_H

#include <array>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace module_ri
{
namespace sternheimer_basis_opt
{

enum class ChunkKind : std::uint32_t
{
    overlap = 1,
    source = 2,
    response = 3,
    coulomb_metric = 4,
    coulomb_whitening = 5,
    hamiltonian = 6,
    occupied_projection = 7,
};

struct PeriodicChunkHeader
{
    std::array<char, 16> magic{};
    std::uint32_t version = 1;
    ChunkKind kind = ChunkKind::overlap;
    std::int32_t iq = 0;
    std::int32_t ik = 0;
    std::int32_t ifrequency = -1;
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
};

struct PeriodicChunk
{
    PeriodicChunkHeader header;
    std::vector<std::complex<double>> values;
};

PeriodicChunkHeader make_periodic_chunk_header(ChunkKind kind,
                                               int iq,
                                               int ik,
                                               int ifrequency,
                                               std::uint64_t rows,
                                               std::uint64_t columns);

void write_periodic_chunk_atomic(const std::string& path,
                                 const PeriodicChunkHeader& header,
                                 const std::vector<std::complex<double>>& values);

PeriodicChunk read_periodic_chunk(const std::string& path);

struct ManifestEntry
{
    std::string file_path;
    std::string relative_path;
    std::string sha256;
    PeriodicChunkHeader header;
    double q_weight = 0.0;
    double k_weight = 0.0;
    double frequency = -1.0;
};

struct KPointRecord
{
    int source_ik = 0;
    int target_ik = 0;
    std::array<double, 3> source_kpoint{};
    std::array<double, 3> target_kpoint{};
    std::array<int, 3> reciprocal_shift{};
    double k_weight = 0.0;
    std::vector<double> occupations;
    std::vector<double> eigenvalues_ry;
};

struct Manifest
{
    std::string abacus_commit;
    std::string executable_sha256;
    std::string orbital_sha256;
    std::string pseudopotential_sha256;
    std::string auxiliary_basis_sha256;
    std::string primitive_blocks_sha256;
    std::string physics_hash;
    std::string kernel;
    int q_count = 0;
    int selected_iq = 0;
    std::array<double, 3> qpoint{};
    double q_weight = 0.0;
    int k_count = 0;
    int frequency_count = 0;
    int raw_auxiliary_dimension = 0;
    int whitened_auxiliary_rank = 0;
    int discarded_auxiliary_rank = 0;
    double coulomb_relative_threshold = 0.0;
    double coulomb_max_orthonormality_error = 0.0;
    std::string coulomb_transform_sha256;
    int primitive_count = 0;
    std::vector<double> frequency_ha;
    std::vector<double> frequency_weights_ha;
    std::vector<KPointRecord> kpoints;
    std::vector<ManifestEntry> entries;
};

ManifestEntry make_manifest_entry(const std::string& file_path,
                                  const std::string& relative_path,
                                  const PeriodicChunkHeader& header,
                                  double q_weight,
                                  double k_weight,
                                  double frequency);

void write_manifest_atomic(const std::string& path, const Manifest& manifest);

} // namespace sternheimer_basis_opt
} // namespace module_ri

#endif
