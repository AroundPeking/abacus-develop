#pragma once

#include <RI/global/Tensor.h>
#include <array>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ExxCompressionIO
{

using TC = std::array<int, 3>;
using TAC = std::pair<int, TC>;

template <class T>
using TensorMap = std::map<int, std::map<TAC, RI::Tensor<T>>>;

namespace detail
{

constexpr std::array<unsigned char, 8> magic{{'E', 'X', 'X', 'C', 'M', 'P', '1', 0}};
constexpr std::uint32_t version = 1;

template <class T>
struct is_supported_scalar : std::false_type
{
};

template <>
struct is_supported_scalar<double> : std::true_type
{
};

template <>
struct is_supported_scalar<std::complex<double>> : std::true_type
{
};

inline void write_bytes(std::ostream& output, const unsigned char* bytes, const std::size_t count)
{
    output.write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(count));
    if (!output)
    {
        throw std::runtime_error("failed to write EXX compression snapshot");
    }
}

inline void read_bytes(std::istream& input, unsigned char* bytes, const std::size_t count)
{
    input.read(reinterpret_cast<char*>(bytes), static_cast<std::streamsize>(count));
    if (!input)
    {
        throw std::runtime_error("truncated EXX compression snapshot");
    }
}

template <class UInt>
void write_unsigned(std::ostream& output, const UInt value)
{
    static_assert(std::is_unsigned<UInt>::value, "write_unsigned requires an unsigned integer");
    std::array<unsigned char, sizeof(UInt)> bytes{};
    for (std::size_t byte = 0; byte != bytes.size(); ++byte)
    {
        bytes[byte] = static_cast<unsigned char>((value >> (8 * byte)) & static_cast<UInt>(0xffU));
    }
    write_bytes(output, bytes.data(), bytes.size());
}

template <class UInt>
UInt read_unsigned(std::istream& input)
{
    static_assert(std::is_unsigned<UInt>::value, "read_unsigned requires an unsigned integer");
    std::array<unsigned char, sizeof(UInt)> bytes{};
    read_bytes(input, bytes.data(), bytes.size());
    UInt value = 0;
    for (std::size_t byte = 0; byte != bytes.size(); ++byte)
    {
        value |= static_cast<UInt>(bytes[byte]) << (8 * byte);
    }
    return value;
}

inline void write_int32(std::ostream& output, const std::int32_t value)
{
    write_unsigned(output, static_cast<std::uint32_t>(value));
}

inline std::int32_t read_int32(std::istream& input)
{
    const std::uint32_t value = read_unsigned<std::uint32_t>(input);
    if (value <= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
    {
        return static_cast<std::int32_t>(value);
    }
    return -1 - static_cast<std::int32_t>(std::numeric_limits<std::uint32_t>::max() - value);
}

inline void write_double(std::ostream& output, const double value)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t), "snapshot format requires binary64 double");
    static_assert(std::numeric_limits<double>::is_iec559, "snapshot format requires IEEE754 double");
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_unsigned(output, bits);
}

inline double read_double(std::istream& input)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t), "snapshot format requires binary64 double");
    static_assert(std::numeric_limits<double>::is_iec559, "snapshot format requires IEEE754 double");
    const std::uint64_t bits = read_unsigned<std::uint64_t>(input);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

template <class T>
struct scalar_io;

template <>
struct scalar_io<double>
{
    static constexpr std::uint8_t code = 1;
    static constexpr std::uint64_t bytes_per_value = 8;

    static void write(std::ostream& output, const double value)
    {
        write_double(output, value);
    }

    static double read(std::istream& input)
    {
        return read_double(input);
    }
};

template <>
struct scalar_io<std::complex<double>>
{
    static constexpr std::uint8_t code = 2;
    static constexpr std::uint64_t bytes_per_value = 16;

    static void write(std::ostream& output, const std::complex<double>& value)
    {
        write_double(output, value.real());
        write_double(output, value.imag());
    }

    static std::complex<double> read(std::istream& input)
    {
        const double real = read_double(input);
        const double imag = read_double(input);
        return {real, imag};
    }
};

inline std::int32_t checked_int32(const int value, const char* field)
{
    const std::int64_t wide_value = value;
    if (wide_value < std::numeric_limits<std::int32_t>::min() || wide_value > std::numeric_limits<std::int32_t>::max())
    {
        throw std::runtime_error(std::string(field) + " does not fit int32");
    }
    return static_cast<std::int32_t>(value);
}

inline std::uint64_t checked_shape_product(const RI::Shape_Vector& shape)
{
    if (shape.size() < 1 || shape.size() > 4)
    {
        throw std::runtime_error("tensor ndim must be in [1, 4]");
    }

    std::uint64_t product = 1;
    for (const std::size_t extent: shape)
    {
        if (extent == 0)
        {
            throw std::runtime_error("tensor shape extents must be nonzero");
        }
        if (extent > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())
            || product > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(extent))
        {
            throw std::runtime_error("tensor shape product overflows uint64");
        }
        product *= static_cast<std::uint64_t>(extent);
    }
    return product;
}

inline std::size_t checked_size_t(const std::uint64_t value, const char* field)
{
    if (value > std::numeric_limits<std::size_t>::max())
    {
        throw std::runtime_error(std::string(field) + " does not fit size_t");
    }
    return static_cast<std::size_t>(value);
}

template <class T>
RI::Tensor<T> make_tensor(const std::array<std::size_t, 4>& shape, const std::uint32_t ndim)
{
    try
    {
        switch (ndim)
        {
        case 1:
            return RI::Tensor<T>({shape[0]});
        case 2:
            return RI::Tensor<T>({shape[0], shape[1]});
        case 3:
            return RI::Tensor<T>({shape[0], shape[1], shape[2]});
        case 4:
            return RI::Tensor<T>({shape[0], shape[1], shape[2], shape[3]});
        default:
            throw std::runtime_error("tensor ndim must be in [1, 4]");
        }
    }
    catch (const std::bad_alloc&)
    {
        throw std::runtime_error("unable to allocate tensor from snapshot");
    }
}

inline std::uint64_t remaining_bytes(std::istream& input)
{
    const std::istream::pos_type current = input.tellg();
    if (current == std::istream::pos_type(-1))
    {
        throw std::runtime_error("failed to inspect EXX compression snapshot");
    }
    input.seekg(0, std::ios::end);
    const std::istream::pos_type end = input.tellg();
    if (end == std::istream::pos_type(-1) || end < current)
    {
        throw std::runtime_error("failed to inspect EXX compression snapshot");
    }
    input.seekg(current);
    if (!input)
    {
        throw std::runtime_error("failed to inspect EXX compression snapshot");
    }
    return static_cast<std::uint64_t>(end - current);
}

} // namespace detail

template <class T>
void write_map(const std::string& path, const TensorMap<T>& tensor_map, const int rank, const int nranks)
{
    static_assert(detail::is_supported_scalar<T>::value,
                  "EXX compression snapshots support only double and std::complex<double>");
    const std::int64_t wide_rank = rank;
    const std::int64_t wide_nranks = nranks;
    if (wide_nranks <= 0 || wide_rank < 0 || wide_rank >= wide_nranks
        || wide_nranks > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::runtime_error("invalid EXX compression snapshot rank metadata");
    }

    std::uint64_t records = 0;
    for (const auto& outer: tensor_map)
    {
        detail::checked_int32(outer.first, "ia1");
        if (outer.second.size() > std::numeric_limits<std::uint64_t>::max() - records)
        {
            throw std::runtime_error("EXX compression snapshot record count overflows uint64");
        }
        records += static_cast<std::uint64_t>(outer.second.size());
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        throw std::runtime_error("unable to open EXX compression snapshot for writing: " + path);
    }

    detail::write_bytes(output, detail::magic.data(), detail::magic.size());
    detail::write_unsigned(output, detail::version);
    detail::write_unsigned(output, detail::scalar_io<T>::code);
    detail::write_unsigned(output, static_cast<std::uint8_t>(0));
    detail::write_unsigned(output, static_cast<std::uint8_t>(0));
    detail::write_unsigned(output, static_cast<std::uint8_t>(0));
    detail::write_unsigned(output, static_cast<std::uint32_t>(rank));
    detail::write_unsigned(output, static_cast<std::uint32_t>(nranks));
    detail::write_unsigned(output, records);

    for (const auto& outer: tensor_map)
    {
        const std::int32_t ia1 = detail::checked_int32(outer.first, "ia1");
        for (const auto& inner: outer.second)
        {
            const std::int32_t ia2 = detail::checked_int32(inner.first.first, "ia2");
            const RI::Tensor<T>& tensor = inner.second;
            const std::uint64_t value_count = detail::checked_shape_product(tensor.shape);
            if (!tensor.data || tensor.data->size() != detail::checked_size_t(value_count, "tensor value count"))
            {
                throw std::runtime_error("tensor storage does not match its shape");
            }

            detail::write_int32(output, ia1);
            detail::write_int32(output, ia2);
            for (const int cell: inner.first.second)
            {
                detail::write_int32(output, detail::checked_int32(cell, "R"));
            }
            detail::write_unsigned(output, static_cast<std::uint32_t>(tensor.shape.size()));
            for (const std::size_t extent: tensor.shape)
            {
                detail::write_unsigned(output, static_cast<std::uint64_t>(extent));
            }
            detail::write_unsigned(output, value_count);
            for (std::uint64_t index = 0; index != value_count; ++index)
            {
                detail::scalar_io<T>::write(output, tensor.ptr()[static_cast<std::size_t>(index)]);
            }
        }
    }

    output.close();
    if (!output)
    {
        throw std::runtime_error("failed to finalize EXX compression snapshot: " + path);
    }
}

template <class T>
TensorMap<T> read_map(const std::string& path)
{
    static_assert(detail::is_supported_scalar<T>::value,
                  "EXX compression snapshots support only double and std::complex<double>");
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("unable to open EXX compression snapshot for reading: " + path);
    }

    std::array<unsigned char, 8> file_magic{};
    detail::read_bytes(input, file_magic.data(), file_magic.size());
    if (file_magic != detail::magic)
    {
        throw std::runtime_error("invalid EXX compression snapshot magic");
    }
    if (detail::read_unsigned<std::uint32_t>(input) != detail::version)
    {
        throw std::runtime_error("unsupported EXX compression snapshot version");
    }
    if (detail::read_unsigned<std::uint8_t>(input) != detail::scalar_io<T>::code)
    {
        throw std::runtime_error("EXX compression snapshot scalar type mismatch");
    }
    for (int byte = 0; byte != 3; ++byte)
    {
        if (detail::read_unsigned<std::uint8_t>(input) != 0)
        {
            throw std::runtime_error("nonzero reserved EXX compression snapshot header byte");
        }
    }
    const std::uint32_t rank = detail::read_unsigned<std::uint32_t>(input);
    const std::uint32_t nranks = detail::read_unsigned<std::uint32_t>(input);
    if (nranks == 0 || rank >= nranks)
    {
        throw std::runtime_error("invalid EXX compression snapshot rank metadata");
    }
    const std::uint64_t records = detail::read_unsigned<std::uint64_t>(input);

    TensorMap<T> tensor_map;
    for (std::uint64_t record = 0; record != records; ++record)
    {
        const int ia1 = detail::read_int32(input);
        const int ia2 = detail::read_int32(input);
        TC cell{};
        for (int& coordinate: cell)
        {
            coordinate = detail::read_int32(input);
        }
        const std::uint32_t ndim = detail::read_unsigned<std::uint32_t>(input);
        if (ndim < 1 || ndim > 4)
        {
            throw std::runtime_error("tensor ndim must be in [1, 4]");
        }

        std::array<std::size_t, 4> shape{};
        std::uint64_t shape_product = 1;
        for (std::uint32_t dim = 0; dim != ndim; ++dim)
        {
            const std::uint64_t extent = detail::read_unsigned<std::uint64_t>(input);
            if (extent == 0)
            {
                throw std::runtime_error("tensor shape extents must be nonzero");
            }
            if (shape_product > std::numeric_limits<std::uint64_t>::max() / extent)
            {
                throw std::runtime_error("tensor shape product overflows uint64");
            }
            shape_product *= extent;
            shape[dim] = detail::checked_size_t(extent, "tensor shape extent");
        }
        const std::uint64_t value_count = detail::read_unsigned<std::uint64_t>(input);
        if (value_count != shape_product)
        {
            throw std::runtime_error("tensor value count does not match its shape");
        }
        if (value_count > std::numeric_limits<std::uint64_t>::max() / detail::scalar_io<T>::bytes_per_value
            || value_count * detail::scalar_io<T>::bytes_per_value > detail::remaining_bytes(input))
        {
            throw std::runtime_error("truncated EXX compression snapshot tensor data");
        }
        detail::checked_size_t(value_count, "tensor value count");

        RI::Tensor<T> tensor = detail::make_tensor<T>(shape, ndim);
        for (std::uint64_t index = 0; index != value_count; ++index)
        {
            tensor.ptr()[static_cast<std::size_t>(index)] = detail::scalar_io<T>::read(input);
        }

        auto& inner_map = tensor_map[ia1];
        const auto inserted = inner_map.emplace(TAC{ia2, cell}, std::move(tensor));
        if (!inserted.second)
        {
            throw std::runtime_error("duplicate EXX compression snapshot record key");
        }
    }

    if (input.peek() != std::char_traits<char>::eof())
    {
        throw std::runtime_error("trailing data in EXX compression snapshot");
    }
    if (input.bad())
    {
        throw std::runtime_error("failed while reading EXX compression snapshot");
    }
    return tensor_map;
}

} // namespace ExxCompressionIO
