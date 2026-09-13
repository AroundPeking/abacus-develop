#include "source_lcao/module_ri/sternheimer_basis_opt_periodic.h"

#include "source_lcao/module_ri/sternheimer_siab_provenance.h"

#include <complex>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <iostream>
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
    manifest.kpoints[0].eigenvalues_ry = {-0.75};
    manifest.kpoints[1].eigenvalues_ry = {-0.50};
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

TEST(PeriodicFrozenAuxiliary, KeepsFrozenRankInsteadOfReselecting)
{
    TemporaryFiles files;
    const auto vpath = files.add("frozen_v.bin");
    const auto wpath = files.add("frozen_w.bin");
    const std::vector<std::complex<double>> v = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    const std::vector<std::complex<double>> w = {1., 0., 0., 1., 0., 0.};
    using namespace module_ri::sternheimer_basis_opt;
    write_periodic_chunk_atomic(vpath, header(ChunkKind::coulomb_metric, 22, 0, -1, 3, 3), v);
    write_periodic_chunk_atomic(wpath, header(ChunkKind::coulomb_whitening, 22, 0, -1, 3, 2), w);
    const auto result = read_frozen_auxiliary_transform(vpath, module_ri::sternheimer_siab::sha256_file(vpath),
        wpath, module_ri::sternheimer_siab::sha256_file(wpath), 22, 3, v);
    EXPECT_EQ(result.rank, 2);
    EXPECT_EQ(result.transform, w);
    EXPECT_DOUBLE_EQ(result.metric_relative_error, 0.);
    EXPECT_DOUBLE_EQ(result.identity_max_error, 0.);
}

TEST(PeriodicFrozenAuxiliary, AlignsRawSignGauge)
{
    TemporaryFiles files;
    const auto vpath = files.add("signed_v.bin");
    const auto wpath = files.add("signed_w.bin");
    using namespace module_ri::sternheimer_basis_opt;
    write_periodic_chunk_atomic(vpath, header(ChunkKind::coulomb_metric, 22, 0, -1, 2, 2),
                                {1., .2, .2, 1.});
    write_periodic_chunk_atomic(wpath, header(ChunkKind::coulomb_whitening, 22, 0, -1, 2, 1), {0., 1.});
    const auto vs = module_ri::sternheimer_siab::sha256_file(vpath);
    const auto ws = module_ri::sternheimer_siab::sha256_file(wpath);
    const auto result = read_frozen_auxiliary_transform(vpath, vs, wpath, ws, 22, 2, {1., -.2, -.2, 1.});
    ASSERT_EQ(result.transform.size(), 2);
    EXPECT_EQ(result.transform[1], std::complex<double>(-1., 0.));
    EXPECT_DOUBLE_EQ(result.metric_relative_error, 0.);
    EXPECT_DOUBLE_EQ(result.identity_max_error, 0.);
    EXPECT_THROW(read_frozen_auxiliary_transform(vpath, std::string(64, 'a'), wpath, ws, 22, 2,
                                               {1., -.2, -.2, 1.}), std::invalid_argument);
    EXPECT_THROW(read_frozen_auxiliary_transform(vpath, vs, wpath, ws, 23, 2,
                                               {1., -.2, -.2, 1.}), std::invalid_argument);
    EXPECT_THROW(read_frozen_auxiliary_transform(vpath, vs, wpath, ws, 22, 2,
                                               {1., -.3, -.3, 1.}), std::runtime_error);
}

TEST(PeriodicFrozenAuxiliary, RequiresCompleteOperatorsOnlyRequest)
{
    using module_ri::sternheimer_basis_opt::validate_frozen_auxiliary_request;
    const std::string sha(64, 'a');
    EXPECT_FALSE(validate_frozen_auxiliary_request(false, "", "", ""));
    EXPECT_FALSE(validate_frozen_auxiliary_request(true, "", "", ""));
    EXPECT_TRUE(validate_frozen_auxiliary_request(true, "reference", sha, sha));
    EXPECT_THROW(validate_frozen_auxiliary_request(false, "reference", sha, sha), std::invalid_argument);
    EXPECT_THROW(validate_frozen_auxiliary_request(true, "", sha, sha), std::invalid_argument);
    EXPECT_THROW(validate_frozen_auxiliary_request(true, "reference", "", sha), std::invalid_argument);
    EXPECT_THROW(validate_frozen_auxiliary_request(true, "reference", sha, ""), std::invalid_argument);
    EXPECT_THROW(validate_frozen_auxiliary_request(true, "reference", sha, std::string(64, 'z')),
                 std::invalid_argument);
}

TEST(PeriodicFrozenAuxiliary, ReferenceQ2Integration)
{
    const char* root = std::getenv("FROZEN_AUXILIARY_TEST_ROOT");
    if (root == nullptr) GTEST_SKIP() << "Optional frozen q2 fixture is not supplied.";
    using namespace module_ri::sternheimer_basis_opt;
    const std::string vpath = std::string(root) + "/coulomb_metric.bin";
    const std::string wpath = std::string(root) + "/coulomb_whitening.bin";
    const auto current = read_periodic_chunk(std::string(root) + "/current_metric.bin");
    const auto result = read_frozen_auxiliary_transform(vpath, module_ri::sternheimer_siab::sha256_file(vpath),
        wpath, module_ri::sternheimer_siab::sha256_file(wpath), 22, 320, current.values);
    EXPECT_EQ(result.rank, 307);
    EXPECT_EQ(result.transform.size(), 320U * 307U);
    EXPECT_LT(result.metric_relative_error, 1.0e-10);
    // An independent identity diagnostic, not the physical source-agreement gate.
    EXPECT_LT(result.identity_max_error, 1.0e-6);
    std::cout << "frozen_q2 rank=" << result.rank << " metric_error=" << result.metric_relative_error
              << " identity_error=" << result.identity_max_error << '\n';
}

namespace
{
Manifest operators_manifest(TemporaryFiles& files)
{
    Manifest manifest = canonical_manifest({});
    manifest.operators_only = true;
    manifest.frozen_charge_sha256 = std::string(64, 'a');
    const auto add = [&](const ChunkKind kind, const int ik, const int rows, const int columns) {
        const auto h = header(kind, 1, ik, -1, rows, columns);
        const std::string path = files.add("operators_" + std::to_string(static_cast<int>(kind))
                                           + "_" + std::to_string(ik) + ".bin");
        module_ri::sternheimer_basis_opt::write_periodic_chunk_atomic(
            path, h, std::vector<std::complex<double>>(rows * columns, {1.0, 0.0}));
        manifest.entries.push_back(module_ri::sternheimer_basis_opt::make_manifest_entry(
            path, path, h, manifest.q_weight, ik == 0 ? 1.0 : 0.5, -1.0));
    };
    add(ChunkKind::coulomb_metric, 0, 3, 3);
    add(ChunkKind::coulomb_whitening, 0, 3, 2);
    for (int ik = 1; ik <= 2; ++ik)
    {
        add(ChunkKind::overlap, ik, 4, 4);
        add(ChunkKind::hamiltonian, ik, 4, 4);
        add(ChunkKind::source, ik, 2, 4);
        add(ChunkKind::occupied_projection, ik, 1, 4);
    }
    return manifest;
}
} // namespace

TEST(SternheimerBasisOptPeriodic, OperatorsOnlyIsDistinctCompleteAndDeterministic)
{
    TemporaryFiles files;
    auto manifest = operators_manifest(files);
    const auto a = files.add("operators_manifest_a.dat");
    const auto b = files.add("operators_manifest_b.dat");
    module_ri::sternheimer_basis_opt::write_manifest_atomic(a, manifest);
    std::reverse(manifest.entries.begin(), manifest.entries.end());
    module_ri::sternheimer_basis_opt::write_manifest_atomic(b, manifest);
    EXPECT_EQ(read_bytes(a), read_bytes(b));
    const auto bytes = read_bytes(a);
    const std::string text(bytes.begin(), bytes.end());
    EXPECT_EQ(text.find("ABACUS_STERNHEIMER_BASIS_OPERATORS_MANIFEST_V1\n"), 0U);
    EXPECT_NE(text.find("response_solved no\n"), std::string::npos);
    EXPECT_NE(text.find("frozen_charge_sha256 " + std::string(64, 'a')), std::string::npos);
    EXPECT_EQ(text.find("all_converged yes"), std::string::npos);
    EXPECT_EQ(text.find("frozen_auxiliary"), std::string::npos);
    EXPECT_EQ(text.find("auxiliary_transform_origin"), std::string::npos);
}

TEST(SternheimerBasisOptPeriodic, FrozenAuxiliaryProvenanceIsExplicitAndRestricted)
{
    TemporaryFiles files;
    auto manifest = operators_manifest(files);
    const auto path = files.add("frozen_operators_manifest.dat");
    manifest.frozen_auxiliary_metric_sha256 = std::string(64, 'b');
    EXPECT_THROW(module_ri::sternheimer_basis_opt::write_manifest_atomic(path, manifest), std::invalid_argument);
    manifest.frozen_auxiliary_whitening_sha256 = std::string(64, 'c');
    module_ri::sternheimer_basis_opt::write_manifest_atomic(path, manifest);
    const auto bytes = read_bytes(path);
    const std::string text(bytes.begin(), bytes.end());
    EXPECT_NE(text.find("auxiliary_transform_origin frozen_reference\n"), std::string::npos);
    EXPECT_NE(text.find("frozen_auxiliary_whitening_sha256 " + std::string(64, 'c')), std::string::npos);
    manifest.operators_only = false;
    EXPECT_THROW(module_ri::sternheimer_basis_opt::write_manifest_atomic(path, manifest), std::invalid_argument);
}

TEST(SternheimerBasisOptPeriodic, OperatorsOnlyRejectsMissingKindsAndBadDensityProvenance)
{
    TemporaryFiles files;
    const auto complete = operators_manifest(files);
    const auto path = files.add("operators_manifest_invalid.dat");
    for (std::size_t i = 0; i < complete.entries.size(); ++i)
    {
        auto missing = complete;
        missing.entries.erase(missing.entries.begin() + i);
        EXPECT_THROW(module_ri::sternheimer_basis_opt::write_manifest_atomic(path, missing), std::invalid_argument);
    }
    auto invalid = complete;
    invalid.frozen_charge_sha256.clear();
    EXPECT_THROW(module_ri::sternheimer_basis_opt::write_manifest_atomic(path, invalid), std::invalid_argument);
    invalid = complete;
    invalid.entries.back().header.columns = 5;
    EXPECT_THROW(module_ri::sternheimer_basis_opt::write_manifest_atomic(path, invalid), std::invalid_argument);
}

TEST(SternheimerBasisOptPeriodic, OperatorsOnlyRejectsResponseChunks)
{
    TemporaryFiles files;
    auto manifest = operators_manifest(files);
    const auto path = files.add("operators_manifest_response.dat");
    for (const auto kind : {ChunkKind::response, ChunkKind::reference_response})
    {
        auto invalid = manifest;
        const int ik = kind == ChunkKind::response ? 1 : 0;
        const auto h = header(kind, 1, ik, 0, 2, 4);
        const auto chunk = files.add("operators_forbidden_" + std::to_string(ik) + ".bin");
        module_ri::sternheimer_basis_opt::write_periodic_chunk_atomic(chunk, h, std::vector<std::complex<double>>(8));
        invalid.entries.push_back(module_ri::sternheimer_basis_opt::make_manifest_entry(
            chunk, chunk, h, 0.25, ik == 0 ? 1.0 : 0.5, 0.75));
        EXPECT_THROW(module_ri::sternheimer_basis_opt::write_manifest_atomic(path, invalid), std::invalid_argument);
    }
}

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
    EXPECT_NE(manifest_text.find("auxiliary_basis_source product_pca\n"), std::string::npos);
    EXPECT_NE(manifest_text.find("primitive_blocks_sha256 7777777777777777"), std::string::npos);
    EXPECT_NE(manifest_text.find("qpoint 0.00000000000000000e+00"), std::string::npos);
    EXPECT_NE(manifest_text.find("frequency 0 7.50000000000000000e-01 1.25000000000000000e-01"),
              std::string::npos);
    EXPECT_NE(manifest_text.find("kpoint 1 2"), std::string::npos);
    EXPECT_NE(manifest_text.find("eigenvalues_ry 1 1 -7.50000000000000000e-01"),
              std::string::npos);

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
    EXPECT_NO_THROW(header(ChunkKind::hamiltonian, 1, 1, -1, 2, 2));
    EXPECT_NO_THROW(header(ChunkKind::occupied_projection, 1, 1, -1, 1, 2));
    EXPECT_NO_THROW(header(ChunkKind::reference_response, 1, 0, 0, 2, 2));
    EXPECT_THROW(header(ChunkKind::coulomb_metric, 1, 1, -1, 1, 1), std::invalid_argument);
    EXPECT_THROW(header(ChunkKind::response, 1, 1, -1, 1, 1), std::invalid_argument);
    EXPECT_THROW(header(ChunkKind::reference_response, 1, 1, 0, 1, 1), std::invalid_argument);
    EXPECT_THROW(header(ChunkKind::reference_response, 1, 0, -1, 1, 1), std::invalid_argument);
    EXPECT_THROW(header(ChunkKind::source, 1, 1, 0, 1, 1), std::invalid_argument);
    EXPECT_THROW(header(ChunkKind::overlap, 1, 1, -1, 0, 1), std::invalid_argument);

    const PeriodicChunkHeader valid = header(ChunkKind::overlap, 1, 1, -1, 1, 1);
    EXPECT_THROW(module_ri::sternheimer_basis_opt::write_periodic_chunk_atomic(path, valid, {}), std::invalid_argument);
}
