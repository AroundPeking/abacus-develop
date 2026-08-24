#include "source_lcao/module_ri/sternheimer_basis_opt_periodic.h"

#include "source_lcao/module_ri/sternheimer_siab_provenance.h"

#include <complex>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using module_ri::sternheimer_basis_opt::ChunkKind;
using module_ri::sternheimer_basis_opt::Manifest;
using module_ri::sternheimer_basis_opt::ManifestEntry;
using module_ri::sternheimer_basis_opt::KPointRecord;
using module_ri::sternheimer_basis_opt::PeriodicChunk;
using module_ri::sternheimer_basis_opt::PeriodicChunkHeader;

class TemporaryFiles
{
  public:
    ~TemporaryFiles()
    {
        for (const std::string& path: paths_)
        {
            std::remove(path.c_str());
            std::remove((path + ".tmp").c_str());
        }
    }

    std::string add(const std::string& path)
    {
        paths_.push_back(path);
        return path;
    }

  private:
    std::vector<std::string> paths_;
};

PeriodicChunkHeader header(const ChunkKind kind,
                           const int iq,
                           const int ik,
                           const int ifrequency,
                           const std::uint64_t rows,
                           const std::uint64_t columns)
{
    return module_ri::sternheimer_basis_opt::make_periodic_chunk_header(kind, iq, ik, ifrequency, rows, columns);
}

Manifest canonical_manifest(const std::vector<ManifestEntry>& entries)
{
    Manifest manifest;
    manifest.abacus_commit = "1111111111111111111111111111111111111111";
    manifest.executable_sha256 = "2222222222222222222222222222222222222222222222222222222222222222";
    manifest.orbital_sha256 = "3333333333333333333333333333333333333333333333333333333333333333";
    manifest.pseudopotential_sha256 = "4444444444444444444444444444444444444444444444444444444444444444";
    manifest.auxiliary_basis_sha256 = "5555555555555555555555555555555555555555555555555555555555555555";
    manifest.primitive_blocks_sha256 = "7777777777777777777777777777777777777777777777777777777777777777";
    manifest.physics_hash = "6666666666666666666666666666666666666666666666666666666666666666";
    manifest.kernel = "full_coulomb";
    manifest.q_count = 2;
    manifest.selected_iq = 1;
    manifest.qpoint = {0.0, 0.0, 0.0};
    manifest.q_weight = 0.25;
    manifest.k_count = 2;
    manifest.frequency_count = 1;
    manifest.raw_auxiliary_dimension = 3;
    manifest.whitened_auxiliary_rank = 2;
    manifest.discarded_auxiliary_rank = 1;
    manifest.coulomb_relative_threshold = 1.0e-10;
    manifest.coulomb_max_orthonormality_error = 2.0e-14;
    manifest.coulomb_transform_sha256 = "8888888888888888888888888888888888888888888888888888888888888888";
    manifest.primitive_count = 4;
    manifest.frequency_ha = {0.75};
    manifest.frequency_weights_ha = {0.125};
    manifest.kpoints = {
        KPointRecord{1, 2, {0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0, 0, 0}, 0.5, {2.0}},
        KPointRecord{2, 1, {0.5, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1, 0, 0}, 0.5, {2.0}},
    };
    manifest.entries = entries;
    return manifest;
}

bool file_exists(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    return static_cast<bool>(input);
}

std::vector<char> read_bytes(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void write_bytes(const std::string& path, const std::vector<char>& bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

} // namespace

TEST(SternheimerBasisOptPeriodic, ComplexChunkRoundTripsAndHasDeterministicHash)
{
    TemporaryFiles files;
    const std::string first = files.add("sternheimer_basis_opt_periodic_roundtrip_a.bin");
    const std::string second = files.add("sternheimer_basis_opt_periodic_roundtrip_b.bin");
    const std::vector<std::complex<double>> values = {
        {1.0, -2.0},
        {3.5, 4.25},
        {-5.0, 6.0},
        {7.0, 0.0},
    };
    const PeriodicChunkHeader expected_header = header(ChunkKind::response, 3, 5, 7, 2, 2);

    module_ri::sternheimer_basis_opt::write_periodic_chunk_atomic(first, expected_header, values);
    module_ri::sternheimer_basis_opt::write_periodic_chunk_atomic(second, expected_header, values);
    EXPECT_FALSE(file_exists(first + ".tmp"));
    EXPECT_FALSE(file_exists(second + ".tmp"));
    EXPECT_EQ(module_ri::sternheimer_siab::sha256_file(first), module_ri::sternheimer_siab::sha256_file(second));

    const PeriodicChunk actual = module_ri::sternheimer_basis_opt::read_periodic_chunk(first);
    EXPECT_EQ(actual.header.version, 1U);
    EXPECT_EQ(actual.header.kind, ChunkKind::response);
    EXPECT_EQ(actual.header.iq, 3);
    EXPECT_EQ(actual.header.ik, 5);
    EXPECT_EQ(actual.header.ifrequency, 7);
    EXPECT_EQ(actual.header.rows, 2U);
    EXPECT_EQ(actual.header.columns, 2U);
    EXPECT_EQ(actual.values, values);
}

TEST(SternheimerBasisOptPeriodic, RejectsTruncatedPayloadAndTrailingBytes)
{
    TemporaryFiles files;
    const std::string truncated = files.add("sternheimer_basis_opt_periodic_truncated.bin");
    const std::string trailing = files.add("sternheimer_basis_opt_periodic_trailing.bin");
    const std::vector<std::complex<double>> values = {{1.0, 2.0}, {3.0, 4.0}};
    const PeriodicChunkHeader expected_header = header(ChunkKind::source, 1, 2, -1, 1, 2);

    module_ri::sternheimer_basis_opt::write_periodic_chunk_atomic(truncated, expected_header, values);
    std::vector<char> truncated_bytes = read_bytes(truncated);
    ASSERT_FALSE(truncated_bytes.empty());
    truncated_bytes.pop_back();
    write_bytes(truncated, truncated_bytes);
    EXPECT_THROW(module_ri::sternheimer_basis_opt::read_periodic_chunk(truncated), std::runtime_error);

    module_ri::sternheimer_basis_opt::write_periodic_chunk_atomic(trailing, expected_header, values);
    {
        std::ofstream output(trailing, std::ios::binary | std::ios::app);
        output.put('\0');
    }
    EXPECT_THROW(module_ri::sternheimer_basis_opt::read_periodic_chunk(trailing), std::runtime_error);
}

TEST(SternheimerBasisOptPeriodic, ManifestRejectsDuplicateRecordsAndWritesDeterministically)
{
    TemporaryFiles files;
    const std::string chunk_path = files.add("sternheimer_basis_opt_periodic_manifest_chunk.bin");
    const std::string manifest_a = files.add("sternheimer_basis_opt_periodic_manifest_a.txt");
    const std::string manifest_b = files.add("sternheimer_basis_opt_periodic_manifest_b.txt");
    const PeriodicChunkHeader chunk_header = header(ChunkKind::response, 1, 2, 0, 1, 1);
    module_ri::sternheimer_basis_opt::write_periodic_chunk_atomic(chunk_path, chunk_header, {{1.0, -1.0}});

    const ManifestEntry entry = module_ri::sternheimer_basis_opt::make_manifest_entry(chunk_path,
                                                                                      "chunks/response_q1_k2_w0.bin",
                                                                                      chunk_header,
                                                                                      0.25,
                                                                                      0.5,
                                                                                      0.75);
    const Manifest manifest = canonical_manifest({entry});
    module_ri::sternheimer_basis_opt::write_manifest_atomic(manifest_a, manifest);
    module_ri::sternheimer_basis_opt::write_manifest_atomic(manifest_b, manifest);
    EXPECT_FALSE(file_exists(manifest_a + ".tmp"));
    EXPECT_EQ(module_ri::sternheimer_siab::sha256_file(manifest_a),
              module_ri::sternheimer_siab::sha256_file(manifest_b));
    const std::vector<char> manifest_bytes = read_bytes(manifest_a);
    const std::string manifest_text(manifest_bytes.begin(), manifest_bytes.end());
    EXPECT_NE(manifest_text.find("selected_iq 1\n"), std::string::npos);
    EXPECT_NE(manifest_text.find("primitive_blocks_sha256 7777777777777777"), std::string::npos);
    EXPECT_NE(manifest_text.find("qpoint 0.00000000000000000e+00"), std::string::npos);
    EXPECT_NE(manifest_text.find("frequency 0 7.50000000000000000e-01 1.25000000000000000e-01"),
              std::string::npos);
    EXPECT_NE(manifest_text.find("kpoint 1 2"), std::string::npos);

    Manifest duplicate = canonical_manifest({entry, entry});
    EXPECT_THROW(module_ri::sternheimer_basis_opt::write_manifest_atomic(manifest_a, duplicate), std::invalid_argument);

    ManifestEntry mismatched = entry;
    mismatched.sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    EXPECT_THROW(module_ri::sternheimer_basis_opt::write_manifest_atomic(manifest_a, canonical_manifest({mismatched})),
                 std::runtime_error);

    Manifest invalid_q_weight = canonical_manifest({entry});
    invalid_q_weight.q_weight = 0.0;
    EXPECT_THROW(module_ri::sternheimer_basis_opt::write_manifest_atomic(manifest_a, invalid_q_weight),
                 std::invalid_argument);

    Manifest invalid_k_layout = canonical_manifest({entry});
    invalid_k_layout.kpoints[1].source_ik = 1;
    EXPECT_THROW(module_ri::sternheimer_basis_opt::write_manifest_atomic(manifest_a, invalid_k_layout),
                 std::invalid_argument);
}

TEST(SternheimerBasisOptPeriodic, RejectsInvalidHeaderDimensionsAndIndices)
{
    TemporaryFiles files;
    const std::string path = files.add("sternheimer_basis_opt_periodic_invalid.bin");
    EXPECT_THROW(header(ChunkKind::response, 0, 1, 0, 1, 1), std::invalid_argument);
    EXPECT_THROW(header(ChunkKind::response, 1, -1, 0, 1, 1), std::invalid_argument);
    EXPECT_NO_THROW(header(ChunkKind::coulomb_metric, 1, 0, -1, 1, 1));
    EXPECT_THROW(header(ChunkKind::coulomb_metric, 1, 1, -1, 1, 1), std::invalid_argument);
    EXPECT_THROW(header(ChunkKind::response, 1, 1, -1, 1, 1), std::invalid_argument);
    EXPECT_THROW(header(ChunkKind::source, 1, 1, 0, 1, 1), std::invalid_argument);
    EXPECT_THROW(header(ChunkKind::overlap, 1, 1, -1, 0, 1), std::invalid_argument);

    const PeriodicChunkHeader valid = header(ChunkKind::overlap, 1, 1, -1, 1, 1);
    EXPECT_THROW(module_ri::sternheimer_basis_opt::write_periodic_chunk_atomic(path, valid, {}), std::invalid_argument);
}
