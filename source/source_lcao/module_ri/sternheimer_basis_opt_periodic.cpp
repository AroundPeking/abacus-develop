#include "source_lcao/module_ri/sternheimer_basis_opt_periodic.h"

#include "source_lcao/module_ri/sternheimer_siab_provenance.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace module_ri
{
namespace sternheimer_basis_opt
{
namespace
{

constexpr std::array<char, 16> kMagic{{
    'A',
    'B',
    'A',
    'C',
    'U',
    'S',
    '_',
    'S',
    'T',
    'B',
    'O',
    'P',
    'T',
    '_',
    'V',
    '1',
}};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint64_t kHeaderBytes = 52;
constexpr std::uint64_t kComplexBytes = 16;

bool valid_kind(const ChunkKind kind)
{
    switch (kind)
    {
    case ChunkKind::overlap:
    case ChunkKind::source:
    case ChunkKind::response:
    case ChunkKind::coulomb_metric:
    case ChunkKind::coulomb_whitening:
        return true;
    }
    return false;
}

std::uint64_t payload_count(const PeriodicChunkHeader& header)
{
    if (header.rows == 0 || header.columns == 0
        || header.rows > std::numeric_limits<std::uint64_t>::max() / header.columns)
    {
        throw std::invalid_argument("Periodic basis-optimization chunk has invalid dimensions.");
    }
    const std::uint64_t count = header.rows * header.columns;
    if (count > (std::numeric_limits<std::uint64_t>::max() - kHeaderBytes) / kComplexBytes
        || count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::overflow_error("Periodic basis-optimization chunk payload size overflows.");
    }
    return count;
}

void validate_header(const PeriodicChunkHeader& header)
{
    if (header.magic != kMagic || header.version != kVersion || !valid_kind(header.kind))
    {
        throw std::invalid_argument("Periodic basis-optimization chunk header is not version 1.");
    }
    if (header.iq <= 0 || header.ik <= 0)
    {
        throw std::invalid_argument("Periodic basis-optimization q and k indices are one-based positive values.");
    }
    if (header.kind == ChunkKind::response)
    {
        if (header.ifrequency < 0)
        {
            throw std::invalid_argument("Periodic response chunks require a non-negative frequency index.");
        }
    }
    else if (header.ifrequency != -1)
    {
        throw std::invalid_argument("Frequency-independent periodic chunks require ifrequency=-1.");
    }
    payload_count(header);
}

void append_u32(std::vector<unsigned char>& bytes, const std::uint32_t value)
{
    for (unsigned int shift = 0; shift != 32; shift += 8)
    {
        bytes.push_back(static_cast<unsigned char>((value >> shift) & 0xffU));
    }
}

void append_u64(std::vector<unsigned char>& bytes, const std::uint64_t value)
{
    for (unsigned int shift = 0; shift != 64; shift += 8)
    {
        bytes.push_back(static_cast<unsigned char>((value >> shift) & 0xffU));
    }
}

std::uint32_t read_u32(const unsigned char* bytes)
{
    std::uint32_t value = 0;
    for (unsigned int shift = 0; shift != 32; shift += 8)
    {
        value |= static_cast<std::uint32_t>(bytes[shift / 8]) << shift;
    }
    return value;
}

std::uint64_t read_u64(const unsigned char* bytes)
{
    std::uint64_t value = 0;
    for (unsigned int shift = 0; shift != 64; shift += 8)
    {
        value |= static_cast<std::uint64_t>(bytes[shift / 8]) << shift;
    }
    return value;
}

void append_double(std::vector<unsigned char>& bytes, const double value)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t), "Periodic chunks require IEEE-754 binary64 doubles.");
    std::uint64_t encoded = 0;
    std::memcpy(&encoded, &value, sizeof(encoded));
    append_u64(bytes, encoded);
}

double read_double(const unsigned char* bytes)
{
    const std::uint64_t encoded = read_u64(bytes);
    double value = 0.0;
    std::memcpy(&value, &encoded, sizeof(value));
    return value;
}

std::vector<unsigned char> encode_header(const PeriodicChunkHeader& header)
{
    validate_header(header);
    std::vector<unsigned char> bytes;
    bytes.reserve(static_cast<std::size_t>(kHeaderBytes));
    bytes.insert(bytes.end(), header.magic.begin(), header.magic.end());
    append_u32(bytes, header.version);
    append_u32(bytes, static_cast<std::uint32_t>(header.kind));
    append_u32(bytes, static_cast<std::uint32_t>(header.iq));
    append_u32(bytes, static_cast<std::uint32_t>(header.ik));
    append_u32(bytes, static_cast<std::uint32_t>(header.ifrequency));
    append_u64(bytes, header.rows);
    append_u64(bytes, header.columns);
    if (bytes.size() != kHeaderBytes)
    {
        throw std::logic_error("Periodic basis-optimization chunk header encoding has the wrong size.");
    }
    return bytes;
}

PeriodicChunkHeader decode_header(const std::array<unsigned char, kHeaderBytes>& bytes)
{
    PeriodicChunkHeader header;
    std::copy_n(bytes.begin(), header.magic.size(), header.magic.begin());
    std::size_t offset = header.magic.size();
    header.version = read_u32(bytes.data() + offset);
    offset += 4;
    header.kind = static_cast<ChunkKind>(read_u32(bytes.data() + offset));
    offset += 4;
    header.iq = static_cast<std::int32_t>(read_u32(bytes.data() + offset));
    offset += 4;
    header.ik = static_cast<std::int32_t>(read_u32(bytes.data() + offset));
    offset += 4;
    header.ifrequency = static_cast<std::int32_t>(read_u32(bytes.data() + offset));
    offset += 4;
    header.rows = read_u64(bytes.data() + offset);
    offset += 8;
    header.columns = read_u64(bytes.data() + offset);
    validate_header(header);
    return header;
}

void write_all(std::ofstream& output, const unsigned char* bytes, const std::size_t size)
{
    output.write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(size));
    if (!output)
    {
        throw std::runtime_error("Failed while writing periodic basis-optimization data.");
    }
}

void atomic_rename(const std::string& temporary, const std::string& destination)
{
    if (std::rename(temporary.c_str(), destination.c_str()) != 0)
    {
        const int error = errno;
        std::remove(temporary.c_str());
        throw std::runtime_error("Cannot atomically rename periodic basis-optimization output: "
                                 + std::string(std::strerror(error)));
    }
}

bool valid_hex(const std::string& value, const std::size_t count)
{
    return value.size() == count && std::all_of(value.begin(), value.end(), [](const unsigned char character) {
               return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f')
                      || (character >= 'A' && character <= 'F');
           });
}

void validate_manifest_entry(const ManifestEntry& entry, const Manifest& manifest)
{
    validate_header(entry.header);
    if (entry.header.iq > manifest.q_count || entry.header.ik > manifest.k_count)
    {
        throw std::invalid_argument("Periodic basis-optimization manifest entry index exceeds campaign dimensions.");
    }
    if (entry.header.kind == ChunkKind::response && entry.header.ifrequency >= manifest.frequency_count)
    {
        throw std::invalid_argument("Periodic basis-optimization response frequency exceeds the manifest grid.");
    }
    if (!std::isfinite(entry.q_weight) || entry.q_weight <= 0.0 || !std::isfinite(entry.k_weight)
        || entry.k_weight <= 0.0)
    {
        throw std::invalid_argument("Periodic basis-optimization manifest weights must be finite and positive.");
    }
    if (entry.header.kind == ChunkKind::response)
    {
        if (!std::isfinite(entry.frequency) || entry.frequency < 0.0)
        {
            throw std::invalid_argument("Periodic response manifest entries require a non-negative frequency.");
        }
    }
    else if (entry.frequency != -1.0)
    {
        throw std::invalid_argument("Frequency-independent manifest entries require frequency=-1.");
    }
    if (entry.file_path.empty() || entry.relative_path.empty()
        || entry.relative_path.find_first_of("\t\r\n") != std::string::npos || !valid_hex(entry.sha256, 64))
    {
        throw std::invalid_argument("Periodic basis-optimization manifest entry path or SHA256 is invalid.");
    }
    if (sternheimer_siab::sha256_file(entry.file_path) != entry.sha256)
    {
        throw std::runtime_error("Periodic basis-optimization chunk SHA256 does not match the manifest.");
    }
}

void validate_manifest(const Manifest& manifest)
{
    if ((!valid_hex(manifest.abacus_commit, 40) && !valid_hex(manifest.abacus_commit, 64))
        || !valid_hex(manifest.executable_sha256, 64) || !valid_hex(manifest.orbital_sha256, 64)
        || !valid_hex(manifest.pseudopotential_sha256, 64) || !valid_hex(manifest.auxiliary_basis_sha256, 64)
        || !valid_hex(manifest.physics_hash, 64))
    {
        throw std::invalid_argument("Periodic basis-optimization manifest provenance hashes are invalid.");
    }
    if (manifest.kernel != "full_coulomb")
    {
        throw std::invalid_argument("Periodic basis-optimization output requires the full_coulomb kernel.");
    }
    if (manifest.q_count <= 0 || manifest.k_count <= 0 || manifest.frequency_count <= 0 || manifest.entries.empty())
    {
        throw std::invalid_argument("Periodic basis-optimization manifest dimensions and entries must be positive.");
    }

    std::set<std::tuple<std::uint32_t, std::int32_t, std::int32_t, std::int32_t>> records;
    for (const ManifestEntry& entry: manifest.entries)
    {
        validate_manifest_entry(entry, manifest);
        const auto record = std::make_tuple(static_cast<std::uint32_t>(entry.header.kind),
                                            entry.header.iq,
                                            entry.header.ik,
                                            entry.header.ifrequency);
        if (!records.insert(record).second)
        {
            throw std::invalid_argument("Periodic basis-optimization manifest contains a duplicate record.");
        }
    }
}

} // namespace

PeriodicChunkHeader make_periodic_chunk_header(const ChunkKind kind,
                                               const int iq,
                                               const int ik,
                                               const int ifrequency,
                                               const std::uint64_t rows,
                                               const std::uint64_t columns)
{
    PeriodicChunkHeader header;
    header.magic = kMagic;
    header.version = kVersion;
    header.kind = kind;
    header.iq = iq;
    header.ik = ik;
    header.ifrequency = ifrequency;
    header.rows = rows;
    header.columns = columns;
    validate_header(header);
    return header;
}

void write_periodic_chunk_atomic(const std::string& path,
                                 const PeriodicChunkHeader& header,
                                 const std::vector<std::complex<double>>& values)
{
    if (path.empty())
    {
        throw std::invalid_argument("Periodic basis-optimization chunk path must not be empty.");
    }
    const std::uint64_t count = payload_count(header);
    const std::vector<unsigned char> encoded_header = encode_header(header);
    if (values.size() != static_cast<std::size_t>(count))
    {
        throw std::invalid_argument("Periodic basis-optimization chunk payload has the wrong size.");
    }

    const std::string temporary = path + ".tmp";
    std::remove(temporary.c_str());
    try
    {
        std::ofstream output(temporary.c_str(), std::ios::binary | std::ios::trunc);
        if (!output)
        {
            throw std::runtime_error("Cannot open periodic basis-optimization temporary chunk.");
        }
        write_all(output, encoded_header.data(), encoded_header.size());
        std::vector<unsigned char> encoded_value;
        encoded_value.reserve(static_cast<std::size_t>(kComplexBytes));
        for (const std::complex<double>& value: values)
        {
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
            {
                throw std::invalid_argument("Periodic basis-optimization chunk contains a non-finite value.");
            }
            encoded_value.clear();
            append_double(encoded_value, value.real());
            append_double(encoded_value, value.imag());
            write_all(output, encoded_value.data(), encoded_value.size());
        }
        output.close();
        if (!output)
        {
            throw std::runtime_error("Cannot close periodic basis-optimization temporary chunk.");
        }
        atomic_rename(temporary, path);
    }
    catch (...)
    {
        std::remove(temporary.c_str());
        throw;
    }
}

PeriodicChunk read_periodic_chunk(const std::string& path)
{
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input)
    {
        throw std::runtime_error("Cannot open periodic basis-optimization chunk.");
    }
    const std::streamoff size = input.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) < kHeaderBytes)
    {
        throw std::runtime_error("Periodic basis-optimization chunk is truncated before its header.");
    }
    input.seekg(0, std::ios::beg);
    std::array<unsigned char, kHeaderBytes> encoded_header{};
    input.read(reinterpret_cast<char*>(encoded_header.data()), static_cast<std::streamsize>(encoded_header.size()));
    if (!input)
    {
        throw std::runtime_error("Cannot read periodic basis-optimization chunk header.");
    }
    const PeriodicChunkHeader header = decode_header(encoded_header);
    const std::uint64_t count = payload_count(header);
    const std::uint64_t expected_size = kHeaderBytes + kComplexBytes * count;
    if (static_cast<std::uint64_t>(size) != expected_size)
    {
        throw std::runtime_error("Periodic basis-optimization chunk payload is truncated or has trailing bytes.");
    }

    std::vector<std::complex<double>> values;
    values.reserve(static_cast<std::size_t>(count));
    std::array<unsigned char, kComplexBytes> encoded_value{};
    for (std::uint64_t index = 0; index != count; ++index)
    {
        input.read(reinterpret_cast<char*>(encoded_value.data()), static_cast<std::streamsize>(encoded_value.size()));
        if (!input)
        {
            throw std::runtime_error("Cannot read periodic basis-optimization chunk payload.");
        }
        const double real = read_double(encoded_value.data());
        const double imaginary = read_double(encoded_value.data() + 8);
        if (!std::isfinite(real) || !std::isfinite(imaginary))
        {
            throw std::runtime_error("Periodic basis-optimization chunk contains a non-finite value.");
        }
        values.emplace_back(real, imaginary);
    }
    return {header, values};
}

ManifestEntry make_manifest_entry(const std::string& file_path,
                                  const std::string& relative_path,
                                  const PeriodicChunkHeader& header,
                                  const double q_weight,
                                  const double k_weight,
                                  const double frequency)
{
    ManifestEntry entry;
    entry.file_path = file_path;
    entry.relative_path = relative_path;
    entry.sha256 = sternheimer_siab::sha256_file(file_path);
    entry.header = header;
    entry.q_weight = q_weight;
    entry.k_weight = k_weight;
    entry.frequency = frequency;
    return entry;
}

void write_manifest_atomic(const std::string& path, const Manifest& manifest)
{
    if (path.empty())
    {
        throw std::invalid_argument("Periodic basis-optimization manifest path must not be empty.");
    }
    validate_manifest(manifest);
    std::vector<ManifestEntry> entries = manifest.entries;
    std::sort(entries.begin(), entries.end(), [](const ManifestEntry& left, const ManifestEntry& right) {
        return std::make_tuple(static_cast<std::uint32_t>(left.header.kind),
                               left.header.iq,
                               left.header.ik,
                               left.header.ifrequency,
                               left.relative_path)
               < std::make_tuple(static_cast<std::uint32_t>(right.header.kind),
                                 right.header.iq,
                                 right.header.ik,
                                 right.header.ifrequency,
                                 right.relative_path);
    });

    const std::string temporary = path + ".tmp";
    std::remove(temporary.c_str());
    try
    {
        std::ofstream output(temporary.c_str(), std::ios::trunc);
        if (!output)
        {
            throw std::runtime_error("Cannot open periodic basis-optimization temporary manifest.");
        }
        output << "ABACUS_STERNHEIMER_BASIS_OPT_MANIFEST_V1\n"
               << "abacus_commit " << manifest.abacus_commit << '\n'
               << "executable_sha256 " << manifest.executable_sha256 << '\n'
               << "orbital_sha256 " << manifest.orbital_sha256 << '\n'
               << "pseudopotential_sha256 " << manifest.pseudopotential_sha256 << '\n'
               << "auxiliary_basis_sha256 " << manifest.auxiliary_basis_sha256 << '\n'
               << "physics_hash " << manifest.physics_hash << '\n'
               << "kernel " << manifest.kernel << '\n'
               << "q_count " << manifest.q_count << '\n'
               << "k_count " << manifest.k_count << '\n'
               << "frequency_count " << manifest.frequency_count << '\n'
               << "entry_count " << entries.size() << '\n'
               << std::scientific << std::setprecision(17);
        for (const ManifestEntry& entry: entries)
        {
            output << "entry\t" << static_cast<std::uint32_t>(entry.header.kind) << '\t' << entry.header.iq << '\t'
                   << entry.header.ik << '\t' << entry.header.ifrequency << '\t' << entry.header.rows << '\t'
                   << entry.header.columns << '\t' << entry.q_weight << '\t' << entry.k_weight << '\t'
                   << entry.frequency << '\t' << entry.relative_path << '\t' << entry.sha256 << '\n';
        }
        output.close();
        if (!output)
        {
            throw std::runtime_error("Cannot close periodic basis-optimization temporary manifest.");
        }
        atomic_rename(temporary, path);
    }
    catch (...)
    {
        std::remove(temporary.c_str());
        throw;
    }
}

} // namespace sternheimer_basis_opt
} // namespace module_ri
