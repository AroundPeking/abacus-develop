#include "../exx_compression_io.h"

#include "../exx_compression_dump.h"

#include <array>
#include <atomic>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace
{

class TemporarySnapshot
{
  public:
    TemporarySnapshot()
    {
        static std::atomic<unsigned long> next_id{0};
        path_ = testing::TempDir() + "exx_compression_io_" + std::to_string(next_id++) + ".bin";
    }

    ~TemporarySnapshot()
    {
        std::remove(path_.c_str());
    }

    TemporarySnapshot(const TemporarySnapshot&) = delete;
    TemporarySnapshot& operator=(const TemporarySnapshot&) = delete;

    const std::string& path() const
    {
        return path_;
    }

  private:
    std::string path_;
};

class TemporaryDumpRoot
{
  public:
    TemporaryDumpRoot()
    {
        static std::atomic<unsigned long> next_id{0};
        path_ = testing::TempDir() + "exx_compression_dump_" + std::to_string(next_id++);
        std::remove(path_.c_str());
        setenv("ABACUS_EXX_COMPRESSION_DUMP", path_.c_str(), 1);
    }

    ~TemporaryDumpRoot()
    {
        DIR* directory = opendir(path_.c_str());
        if (directory != nullptr)
        {
            while (dirent* entry = readdir(directory))
            {
                const std::string name = entry->d_name;
                if (name != "." && name != "..")
                {
                    std::remove((path_ + "/" + name).c_str());
                }
            }
            closedir(directory);
        }
        rmdir(path_.c_str());
        unsetenv("ABACUS_EXX_COMPRESSION_DUMP");
    }

    TemporaryDumpRoot(const TemporaryDumpRoot&) = delete;
    TemporaryDumpRoot& operator=(const TemporaryDumpRoot&) = delete;

    const std::string& path() const
    {
        return path_;
    }

  private:
    std::string path_;
};

ExxCompressionDump::ManifestContext make_manifest_context()
{
    ExxCompressionDump::ManifestContext context;
    context.scalar_type = "real64";
    context.period = {{2, 2, 1}};
    context.lattice_vectors = {{{{3.0, 0.0, 0.0}}, {{0.0, 3.0, 0.0}}, {{0.0, 0.0, 8.0}}}};
    context.atom_positions[0] = {{0.0, 0.0, 0.0}};
    context.c_threshold = 1.0e-4;
    context.v_threshold = 2.0e-4;
    context.v_threshold_long = 3.0e-4;
    context.d_threshold = 4.0e-4;
    return context;
}

std::string read_text(const std::string& path)
{
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_required_snapshot_entries()
{
    const ExxCompressionIO::TensorMap<double> empty_map;
    for (const std::string& object: {"C.active", "V.raw", "V.active", "D.raw", "D.active", "H.lri", "H.final"})
    {
        ExxCompressionDump::write_if_enabled(object, empty_map, 0, "full", MPI_COMM_WORLD);
    }
    ExxCompressionDump::write_scalar_if_enabled("E.lri", 1.0, 0, "full", MPI_COMM_WORLD);
    ExxCompressionDump::write_scalar_if_enabled("E.final", -2.0, -1, "total", MPI_COMM_WORLD);
}

std::vector<unsigned char> read_bytes(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_bytes(const std::string& path, const std::vector<unsigned char>& bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void write_uint32_le(std::vector<unsigned char>& bytes, const std::size_t offset, const std::uint32_t value)
{
    ASSERT_LE(offset + 4, bytes.size());
    for (std::size_t byte = 0; byte != 4; ++byte)
    {
        bytes[offset + byte] = static_cast<unsigned char>((value >> (8 * byte)) & 0xffU);
    }
}

void write_uint64_le(std::vector<unsigned char>& bytes, const std::size_t offset, const std::uint64_t value)
{
    ASSERT_LE(offset + 8, bytes.size());
    for (std::size_t byte = 0; byte != 8; ++byte)
    {
        bytes[offset + byte] = static_cast<unsigned char>((value >> (8 * byte)) & 0xffU);
    }
}

template <class T>
void expect_shape(const RI::Tensor<T>& tensor, const std::vector<std::size_t>& expected)
{
    ASSERT_EQ(tensor.shape.size(), expected.size());
    for (std::size_t dim = 0; dim != expected.size(); ++dim)
    {
        EXPECT_EQ(tensor.shape[dim], expected[dim]);
    }
}

TEST(ExxCompressionIOTest, RealMapRoundTrip)
{
    ExxCompressionIO::TensorMap<double> original;
    RI::Tensor<double> tensor({2, 3});
    for (std::size_t index = 0; index != 6; ++index)
    {
        tensor.ptr()[index] = static_cast<double>(index) - 2.5;
    }
    original[7][{4, {-2, 3, 1}}] = tensor;

    TemporarySnapshot snapshot;
    ExxCompressionIO::write_map(snapshot.path(), original, 2, 5);
    const auto restored = ExxCompressionIO::read_map<double>(snapshot.path());

    ASSERT_EQ(restored.size(), std::size_t{1});
    const auto& restored_tensor = restored.at(7).at({4, {-2, 3, 1}});
    expect_shape(restored_tensor, {2, 3});
    for (std::size_t index = 0; index != 6; ++index)
    {
        EXPECT_EQ(restored_tensor.ptr()[index], tensor.ptr()[index]);
    }
}

TEST(ExxCompressionIOTest, ComplexMapRoundTrip)
{
    ExxCompressionIO::TensorMap<std::complex<double>> original;
    RI::Tensor<std::complex<double>> tensor({2, 2, 3});
    for (std::size_t index = 0; index != 12; ++index)
    {
        tensor.ptr()[index] = {static_cast<double>(index), -0.25 * static_cast<double>(index)};
    }
    original[1][{0, {2, -1, 0}}] = tensor;

    TemporarySnapshot snapshot;
    ExxCompressionIO::write_map(snapshot.path(), original, 0, 4);
    const auto restored = ExxCompressionIO::read_map<std::complex<double>>(snapshot.path());

    ASSERT_EQ(restored.size(), std::size_t{1});
    const auto& restored_tensor = restored.at(1).at({0, {2, -1, 0}});
    expect_shape(restored_tensor, {2, 2, 3});
    for (std::size_t index = 0; index != 12; ++index)
    {
        EXPECT_EQ(restored_tensor.ptr()[index], tensor.ptr()[index]);
    }
}

TEST(ExxCompressionIOTest, BadMagicThrows)
{
    TemporarySnapshot snapshot;
    ExxCompressionIO::write_map(snapshot.path(), ExxCompressionIO::TensorMap<double>{}, 0, 1);
    auto bytes = read_bytes(snapshot.path());
    ASSERT_FALSE(bytes.empty());
    bytes[0] = 'X';
    write_bytes(snapshot.path(), bytes);

    EXPECT_THROW(ExxCompressionIO::read_map<double>(snapshot.path()), std::runtime_error);
}

TEST(ExxCompressionIOTest, SchemaIsLittleEndian)
{
    ExxCompressionIO::TensorMap<double> original;
    RI::Tensor<double> tensor({1});
    tensor.ptr()[0] = 1.0;
    original[0x01020304][{0x10203040, {-2, 3, 4}}] = tensor;

    TemporarySnapshot snapshot;
    ExxCompressionIO::write_map(snapshot.path(), original, 1, 3);
    const auto bytes = read_bytes(snapshot.path());

    ASSERT_EQ(bytes.size(), std::size_t{80});
    EXPECT_EQ(std::vector<unsigned char>(bytes.begin(), bytes.begin() + 8),
              (std::vector<unsigned char>{'E', 'X', 'X', 'C', 'M', 'P', '1', 0}));
    EXPECT_EQ(std::vector<unsigned char>(bytes.begin() + 8, bytes.begin() + 12),
              (std::vector<unsigned char>{1, 0, 0, 0}));
    EXPECT_EQ(std::vector<unsigned char>(bytes.begin() + 16, bytes.begin() + 24),
              (std::vector<unsigned char>{1, 0, 0, 0, 3, 0, 0, 0}));
    EXPECT_EQ(std::vector<unsigned char>(bytes.begin() + 32, bytes.begin() + 40),
              (std::vector<unsigned char>{4, 3, 2, 1, 0x40, 0x30, 0x20, 0x10}));
    EXPECT_EQ(std::vector<unsigned char>(bytes.begin() + 72, bytes.end()),
              (std::vector<unsigned char>{0, 0, 0, 0, 0, 0, 0xf0, 0x3f}));
}

TEST(ExxCompressionIOTest, ScalarMismatchThrows)
{
    TemporarySnapshot snapshot;
    ExxCompressionIO::write_map(snapshot.path(), ExxCompressionIO::TensorMap<double>{}, 0, 1);

    EXPECT_THROW(ExxCompressionIO::read_map<std::complex<double>>(snapshot.path()), std::runtime_error);
}

TEST(ExxCompressionIOTest, InvalidRankThrows)
{
    TemporarySnapshot snapshot;
    EXPECT_THROW(ExxCompressionIO::write_map(snapshot.path(), ExxCompressionIO::TensorMap<double>{}, 1, 1),
                 std::runtime_error);
    EXPECT_THROW(ExxCompressionIO::write_map(snapshot.path(), ExxCompressionIO::TensorMap<double>{}, -1, 1),
                 std::runtime_error);
    EXPECT_THROW(ExxCompressionIO::write_map(snapshot.path(), ExxCompressionIO::TensorMap<double>{}, 0, 0),
                 std::runtime_error);
}

TEST(ExxCompressionIOTest, TruncatedDataThrows)
{
    ExxCompressionIO::TensorMap<double> original;
    original[0][{0, {0, 0, 0}}] = RI::Tensor<double>({2});
    TemporarySnapshot snapshot;
    ExxCompressionIO::write_map(snapshot.path(), original, 0, 1);
    auto bytes = read_bytes(snapshot.path());
    ASSERT_GT(bytes.size(), std::size_t{1});
    bytes.pop_back();
    write_bytes(snapshot.path(), bytes);

    EXPECT_THROW(ExxCompressionIO::read_map<double>(snapshot.path()), std::runtime_error);
}

TEST(ExxCompressionIOTest, InvalidShapeMetadataThrows)
{
    ExxCompressionIO::TensorMap<double> original;
    original[0][{0, {0, 0, 0}}] = RI::Tensor<double>({2, 3});
    TemporarySnapshot snapshot;
    ExxCompressionIO::write_map(snapshot.path(), original, 0, 1);

    auto bad_ndim = read_bytes(snapshot.path());
    write_uint32_le(bad_ndim, 52, 0);
    write_bytes(snapshot.path(), bad_ndim);
    EXPECT_THROW(ExxCompressionIO::read_map<double>(snapshot.path()), std::runtime_error);

    ExxCompressionIO::write_map(snapshot.path(), original, 0, 1);
    auto zero_shape = read_bytes(snapshot.path());
    write_uint64_le(zero_shape, 56, 0);
    write_bytes(snapshot.path(), zero_shape);
    EXPECT_THROW(ExxCompressionIO::read_map<double>(snapshot.path()), std::runtime_error);

    ExxCompressionIO::write_map(snapshot.path(), original, 0, 1);
    auto bad_value_count = read_bytes(snapshot.path());
    write_uint64_le(bad_value_count, 72, 5);
    write_bytes(snapshot.path(), bad_value_count);
    EXPECT_THROW(ExxCompressionIO::read_map<double>(snapshot.path()), std::runtime_error);
}

TEST(ExxCompressionIOTest, DuplicateRecordThrows)
{
    ExxCompressionIO::TensorMap<double> original;
    original[0][{0, {0, 0, 0}}] = RI::Tensor<double>({1});
    TemporarySnapshot snapshot;
    ExxCompressionIO::write_map(snapshot.path(), original, 0, 1);
    auto bytes = read_bytes(snapshot.path());
    const std::vector<unsigned char> record(bytes.begin() + 32, bytes.end());
    bytes.insert(bytes.end(), record.begin(), record.end());
    write_uint64_le(bytes, 24, 2);
    write_bytes(snapshot.path(), bytes);

    EXPECT_THROW(ExxCompressionIO::read_map<double>(snapshot.path()), std::runtime_error);
}

TEST(ExxCompressionIOTest, TrailingGarbageThrows)
{
    TemporarySnapshot snapshot;
    ExxCompressionIO::write_map(snapshot.path(), ExxCompressionIO::TensorMap<double>{}, 0, 1);
    auto bytes = read_bytes(snapshot.path());
    bytes.push_back(0xff);
    write_bytes(snapshot.path(), bytes);

    EXPECT_THROW(ExxCompressionIO::read_map<double>(snapshot.path()), std::runtime_error);
}

TEST(ExxCompressionIOTest, Int32ExtremaRoundTrip)
{
    ExxCompressionIO::TensorMap<double> original;
    RI::Tensor<double> tensor({1});
    tensor.ptr()[0] = 7.0;
    original[std::numeric_limits<std::int32_t>::min()]
            [{std::numeric_limits<std::int32_t>::max(),
              {std::numeric_limits<std::int32_t>::min(), 0, std::numeric_limits<std::int32_t>::max()}}]
        = tensor;

    TemporarySnapshot snapshot;
    ExxCompressionIO::write_map(snapshot.path(), original, 0, 1);
    const auto restored = ExxCompressionIO::read_map<double>(snapshot.path());

    EXPECT_EQ(restored.begin()->first, std::numeric_limits<std::int32_t>::min());
    const auto& key = restored.begin()->second.begin()->first;
    EXPECT_EQ(key.first, std::numeric_limits<std::int32_t>::max());
    EXPECT_EQ(key.second,
              (ExxCompressionIO::TC{
                  {std::numeric_limits<std::int32_t>::min(), 0, std::numeric_limits<std::int32_t>::max()}}));
}

TEST(ExxCompressionDumpTest, EnvironmentAndPaths)
{
    EXPECT_FALSE(ExxCompressionDump::enabled(nullptr));
    EXPECT_FALSE(ExxCompressionDump::enabled(""));
    EXPECT_FALSE(ExxCompressionDump::enabled("0"));
    EXPECT_TRUE(ExxCompressionDump::enabled("/tmp/snap"));
    EXPECT_EQ(ExxCompressionDump::map_path("/tmp/snap", "D", 1, "short", 3),
              "/tmp/snap/D.spin1.short.rank000003.exxcmp");
    EXPECT_EQ(ExxCompressionDump::scalar_path("/tmp/snap", "E.lri", 0, "full", 12),
              "/tmp/snap/E.lri.spin0.full.rank000012.scalar");
}

TEST(ExxCompressionDumpTest, MapPublishDoesNotOverwriteFinalFile)
{
    TemporaryDumpRoot dump_root;
    ExxCompressionDump::initialize(make_manifest_context(), MPI_COMM_WORLD);

    ExxCompressionIO::TensorMap<double> tensor_map;
    RI::Tensor<double> tensor({1});
    tensor.ptr()[0] = 2.5;
    tensor_map[0][{0, {0, 0, 0}}] = tensor;
    ExxCompressionDump::write_if_enabled("D.raw", tensor_map, 1, "short", MPI_COMM_WORLD);
    const std::string final_path = ExxCompressionDump::map_path(dump_root.path(), "D.raw", 1, "short", 0);
    EXPECT_EQ(ExxCompressionIO::read_map<double>(final_path).at(0).at({0, {0, 0, 0}}).ptr()[0], 2.5);

    EXPECT_THROW(ExxCompressionDump::write_if_enabled("D.raw", tensor_map, 1, "short", MPI_COMM_WORLD),
                 std::runtime_error);
    EXPECT_EQ(ExxCompressionIO::read_map<double>(final_path).at(0).at({0, {0, 0, 0}}).ptr()[0], 2.5);
}

TEST(ExxCompressionDumpTest, ScalarUsesRealImaginaryColumns)
{
    TemporaryDumpRoot dump_root;
    ExxCompressionDump::initialize(make_manifest_context(), MPI_COMM_WORLD);
    ExxCompressionDump::write_scalar_if_enabled("E.lri", 1.25, 0, "full", MPI_COMM_WORLD);

    std::ifstream input(ExxCompressionDump::scalar_path(dump_root.path(), "E.lri", 0, "full", 0));
    double real = 0.0;
    double imag = 1.0;
    input >> real >> imag;
    EXPECT_TRUE(input.good() || input.eof());
    EXPECT_DOUBLE_EQ(real, 1.25);
    EXPECT_DOUBLE_EQ(imag, 0.0);
}

TEST(ExxCompressionDumpTest, MalformedManifestTailIsRejected)
{
    TemporaryDumpRoot dump_root;
    ASSERT_EQ(mkdir(dump_root.path().c_str(), 0755), 0);
    {
        std::ofstream manifest(dump_root.path() + "/manifest.txt");
        manifest << "EXX_COMPRESSION_MANIFEST 1\nENTRY\nobject=D.active\n";
    }

    EXPECT_THROW(ExxCompressionDump::initialize(make_manifest_context(), MPI_COMM_WORLD), std::runtime_error);
}

TEST(ExxCompressionDumpTest, FirstElectronicSnapshotGateClosesOnlyAfterCompletion)
{
    TemporaryDumpRoot dump_root;
    ExxCompressionDump::initialize(make_manifest_context(), MPI_COMM_WORLD);

    EXPECT_TRUE(ExxCompressionDump::begin_first_electronic_snapshot(MPI_COMM_WORLD));
    EXPECT_THROW(ExxCompressionDump::mark_complete(MPI_COMM_WORLD), std::runtime_error);
    EXPECT_TRUE(ExxCompressionDump::begin_first_electronic_snapshot(MPI_COMM_WORLD));

    write_required_snapshot_entries();
    ExxCompressionDump::mark_complete(MPI_COMM_WORLD);

    EXPECT_FALSE(ExxCompressionDump::begin_first_electronic_snapshot(MPI_COMM_WORLD));
    EXPECT_NE(read_text(dump_root.path() + "/manifest.txt").find("session_state=complete\n"), std::string::npos);
    EXPECT_FALSE(std::ifstream(dump_root.path() + "/snapshot.complete").good());
}

TEST(ExxCompressionDumpTest, SameProcessSkipsRepeatedInitializationButRejectsUnrelatedRoot)
{
    TemporaryDumpRoot dump_root;
    EXPECT_TRUE(ExxCompressionDump::initialize(make_manifest_context(), MPI_COMM_WORLD));
    const std::string manifest_path = dump_root.path() + "/manifest.txt";
    const std::string original = read_text(manifest_path);

    EXPECT_FALSE(ExxCompressionDump::initialize(make_manifest_context(), MPI_COMM_WORLD));
    EXPECT_EQ(read_text(manifest_path), original);
    EXPECT_NE(original.find("session_state=incomplete\n"), std::string::npos);

    TemporaryDumpRoot unrelated_root;
    ASSERT_EQ(mkdir(unrelated_root.path().c_str(), 0755), 0);
    {
        std::ofstream manifest(unrelated_root.path() + "/manifest.txt");
        manifest << original;
    }
    EXPECT_THROW(ExxCompressionDump::initialize(make_manifest_context(), MPI_COMM_WORLD), std::runtime_error);
    EXPECT_EQ(read_text(unrelated_root.path() + "/manifest.txt"), original);
}

TEST(ExxCompressionDumpTest, ManifestEntryUpdatesRemainStructurallyComplete)
{
    TemporaryDumpRoot dump_root;
    ExxCompressionDump::initialize(make_manifest_context(), MPI_COMM_WORLD);
    const ExxCompressionIO::TensorMap<double> empty_map;
    ExxCompressionDump::write_if_enabled("C.active", empty_map, -1, "full", MPI_COMM_WORLD);

    const std::string manifest = read_text(dump_root.path() + "/manifest.txt");
    EXPECT_EQ(manifest.rfind("END_ENTRY\n"), manifest.size() - std::string("END_ENTRY\n").size());
    EXPECT_EQ(manifest.find("\nENTRY\n"), manifest.rfind("\nENTRY\n"));
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
