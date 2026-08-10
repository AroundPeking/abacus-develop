#include "../exx_compression_io.h"

#include <array>
#include <atomic>
#include <complex>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
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

} // namespace
