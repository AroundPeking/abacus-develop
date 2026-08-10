#pragma once

#include "exx_compression_io.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <mpi.h>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

#if defined(COMMIT_INFO)
#include "commit.h"
#endif

namespace ExxCompressionDump
{

struct ManifestContext
{
    std::string scalar_type;
    std::array<int, 3> period{{0, 0, 0}};
    std::array<std::array<double, 3>, 3> lattice_vectors{};
    std::map<int, std::array<double, 3>> atom_positions;
    double c_threshold = 0.0;
    double v_threshold = 0.0;
    double v_threshold_long = 0.0;
    double d_threshold = 0.0;
};

inline bool enabled(const char* value)
{
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

inline std::string root()
{
    const char* value = std::getenv("ABACUS_EXX_COMPRESSION_DUMP");
    return enabled(value) ? value : std::string{};
}

namespace detail
{

inline std::set<std::string>& initialized_roots()
{
    static std::set<std::string> roots;
    return roots;
}

inline std::mutex& initialized_roots_mutex()
{
    static std::mutex mutex;
    return mutex;
}

inline bool is_initialized_root(const std::string& dump_root)
{
    std::lock_guard<std::mutex> lock(initialized_roots_mutex());
    return initialized_roots().count(dump_root) != 0;
}

inline bool register_initialized_root(const std::string& dump_root)
{
    std::lock_guard<std::mutex> lock(initialized_roots_mutex());
    return initialized_roots().insert(dump_root).second;
}

inline void unregister_initialized_root(const std::string& dump_root)
{
    std::lock_guard<std::mutex> lock(initialized_roots_mutex());
    initialized_roots().erase(dump_root);
}

inline void validate_component(const std::string& component, const char* field)
{
    if (component.empty() || component == "." || component == "..")
    {
        throw std::runtime_error(std::string("invalid EXX compression dump ") + field);
    }
    for (const unsigned char character: component)
    {
        const bool valid = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z')
                           || (character >= '0' && character <= '9') || character == '.' || character == '_'
                           || character == '-';
        if (!valid)
        {
            throw std::runtime_error(std::string("invalid EXX compression dump ") + field + ": " + component);
        }
    }
}

inline std::string rank_string(const int rank)
{
    if (rank < 0)
    {
        throw std::runtime_error("invalid EXX compression dump rank");
    }
    std::ostringstream output;
    output << std::setw(6) << std::setfill('0') << rank;
    return output.str();
}

inline bool is_directory(const std::string& path)
{
    struct stat status;
    return stat(path.c_str(), &status) == 0 && S_ISDIR(status.st_mode);
}

inline bool path_exists(const std::string& path)
{
    struct stat status;
    return lstat(path.c_str(), &status) == 0;
}

inline bool create_directories(const std::string& path)
{
    if (path.empty())
    {
        throw std::runtime_error("empty EXX compression dump root");
    }
    if (is_directory(path))
    {
        return false;
    }
    if (path_exists(path))
    {
        throw std::runtime_error("EXX compression dump root is not a directory: " + path);
    }

    std::string normalized = path;
    while (normalized.size() > 1 && normalized.back() == '/')
    {
        normalized.pop_back();
    }

    std::string partial;
    std::size_t position = 0;
    bool created_final = false;
    if (normalized[0] == '/')
    {
        partial = "/";
        position = 1;
    }
    while (position <= normalized.size())
    {
        const std::size_t slash = normalized.find('/', position);
        const std::string component = normalized.substr(position, slash - position);
        if (!component.empty())
        {
            if (partial.size() > 1 && partial.back() != '/')
            {
                partial += '/';
            }
            partial += component;
            const int mkdir_result = mkdir(partial.c_str(), 0755);
            if (mkdir_result != 0 && errno != EEXIST)
            {
                throw std::runtime_error("failed to create EXX compression dump directory " + partial + ": "
                                         + std::strerror(errno));
            }
            if (!is_directory(partial))
            {
                throw std::runtime_error("EXX compression dump path is not a directory: " + partial);
            }
            if (slash == std::string::npos)
            {
                created_final = mkdir_result == 0;
            }
        }
        if (slash == std::string::npos)
        {
            break;
        }
        position = slash + 1;
    }
    return created_final;
}

inline std::string abacus_commit()
{
#if defined(COMMIT_INFO)
    return COMMIT;
#else
    return "unknown";
#endif
}

inline std::string utc_time()
{
    const std::time_t now = std::time(nullptr);
    std::tm value{};
#if defined(_WIN32)
    gmtime_s(&value, &now);
#else
    gmtime_r(&now, &value);
#endif
    char buffer[32] = {};
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &value) == 0)
    {
        throw std::runtime_error("failed to format EXX compression dump UTC time");
    }
    return buffer;
}

inline std::string manifest_context_text(const ManifestContext& context, const int mpi_size)
{
    if (context.scalar_type.empty())
    {
        throw std::runtime_error("EXX compression dump scalar type is empty");
    }
    std::ostringstream output;
    output << std::setprecision(17);
    output << "EXX_COMPRESSION_MANIFEST 1\n";
    output << "abacus_commit=" << abacus_commit() << '\n';
    output << "scalar_type=" << context.scalar_type << '\n';
    output << "mpi_size=" << mpi_size << '\n';
    output << "period=" << context.period[0] << ' ' << context.period[1] << ' ' << context.period[2] << '\n';
    for (std::size_t row = 0; row != context.lattice_vectors.size(); ++row)
    {
        output << "lattice_vector." << row << '=' << context.lattice_vectors[row][0] << ' '
               << context.lattice_vectors[row][1] << ' ' << context.lattice_vectors[row][2] << '\n';
    }
    for (const auto& atom: context.atom_positions)
    {
        output << "atom_position." << atom.first << '=' << atom.second[0] << ' ' << atom.second[1] << ' '
               << atom.second[2] << '\n';
    }
    output << "C_threshold=" << context.c_threshold << '\n';
    output << "V_threshold=" << context.v_threshold << '\n';
    output << "V_threshold_long=" << context.v_threshold_long << '\n';
    output << "D_threshold=" << context.d_threshold << '\n';
    output << "END_CONTEXT\n";
    return output.str();
}

inline std::string read_manifest_context(std::istream& input)
{
    std::ostringstream output;
    std::string line;
    bool complete = false;
    while (std::getline(input, line))
    {
        output << line << '\n';
        if (line == "END_CONTEXT")
        {
            complete = true;
            break;
        }
    }
    if (!complete)
    {
        throw std::runtime_error("malformed EXX compression dump manifest");
    }
    return output.str();
}

inline bool starts_with(const std::string& value, const char* prefix)
{
    const std::size_t prefix_size = std::strlen(prefix);
    return value.size() > prefix_size && value.compare(0, prefix_size, prefix) == 0;
}

inline std::string require_manifest_value(std::istream& input, const char* prefix)
{
    std::string line;
    if (!std::getline(input, line) || !starts_with(line, prefix))
    {
        throw std::runtime_error("malformed EXX compression dump manifest entry");
    }
    return line.substr(std::strlen(prefix));
}

struct ParsedManifest
{
    std::string context;
    std::string state;
    std::set<std::string> objects;
};

inline ParsedManifest parse_manifest(const std::string& text)
{
    std::istringstream input(text);
    ParsedManifest parsed;
    parsed.context = read_manifest_context(input);
    std::string line;
    if (!std::getline(input, line) || !starts_with(line, "session_state="))
    {
        throw std::runtime_error("EXX compression dump manifest has no session state");
    }
    parsed.state = line.substr(std::strlen("session_state="));
    if (parsed.state != "incomplete" && parsed.state != "complete")
    {
        throw std::runtime_error("invalid EXX compression dump manifest session state");
    }
    if (!std::getline(input, line) || !starts_with(line, "created_utc="))
    {
        throw std::runtime_error("EXX compression dump manifest has no creation time");
    }

    bool saw_completed_utc = false;
    while (std::getline(input, line))
    {
        if (line == "ENTRY")
        {
            require_manifest_value(input, "utc=");
            const std::string object = require_manifest_value(input, "object=");
            validate_component(object, "manifest object");
            require_manifest_value(input, "state=");
            require_manifest_value(input, "spin=");
            const std::string channel = require_manifest_value(input, "channel=");
            validate_component(channel, "manifest channel");
            require_manifest_value(input, "global_block_count=");
            require_manifest_value(input, "global_serialized_bytes=");
            require_manifest_value(input, "counts_for_compression_metrics=");
            if (!std::getline(input, line) || line != "END_ENTRY")
            {
                throw std::runtime_error("malformed EXX compression dump manifest entry");
            }
            parsed.objects.insert(object);
        }
        else if (starts_with(line, "completed_utc="))
        {
            if (saw_completed_utc)
            {
                throw std::runtime_error("duplicate EXX compression dump completion time");
            }
            saw_completed_utc = true;
        }
        else
        {
            throw std::runtime_error("malformed EXX compression dump manifest tail");
        }
    }
    if ((parsed.state == "complete") != saw_completed_utc)
    {
        throw std::runtime_error("EXX compression dump manifest completion state is inconsistent");
    }
    return parsed;
}

inline std::string read_text_file(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("failed to open EXX compression dump manifest: " + path);
    }
    const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (input.bad())
    {
        throw std::runtime_error("failed to read EXX compression dump manifest: " + path);
    }
    return text;
}

template <class Function>
void rank_zero_action(MPI_Comm comm, Function&& function)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    std::string error;
    if (rank == 0)
    {
        try
        {
            function();
        }
        catch (const std::exception& exception)
        {
            error = exception.what();
        }
    }
    int error_size = static_cast<int>(error.size());
    MPI_Bcast(&error_size, 1, MPI_INT, 0, comm);
    std::vector<char> error_buffer(static_cast<std::size_t>(error_size));
    if (rank == 0 && error_size > 0)
    {
        std::copy(error.begin(), error.end(), error_buffer.begin());
    }
    if (error_size > 0)
    {
        MPI_Bcast(error_buffer.data(), error_size, MPI_CHAR, 0, comm);
        throw std::runtime_error(std::string(error_buffer.begin(), error_buffer.end()));
    }
}

inline void collective_require(MPI_Comm comm, const std::string& local_error)
{
    const int local_ok = local_error.empty() ? 1 : 0;
    int all_ok = 0;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
    if (all_ok == 0)
    {
        throw std::runtime_error(local_error.empty() ? "EXX compression dump failed on another MPI rank" : local_error);
    }
}

inline std::string temporary_path(const std::string& final_path, const int rank)
{
    static std::atomic<unsigned long> next_id{0};
    return final_path + ".tmp." + std::to_string(static_cast<unsigned long>(getpid())) + "." + std::to_string(rank)
           + "." + std::to_string(next_id++);
}

class TemporaryFile
{
  public:
    explicit TemporaryFile(std::string path) : path_(std::move(path))
    {
    }

    ~TemporaryFile()
    {
        if (!path_.empty())
        {
            std::remove(path_.c_str());
        }
    }

    TemporaryFile(const TemporaryFile&) = delete;
    TemporaryFile& operator=(const TemporaryFile&) = delete;

    const std::string& path() const
    {
        return path_;
    }

    void release()
    {
        path_.clear();
    }

  private:
    std::string path_;
};

inline void ensure_final_absent(const std::string& final_path)
{
    if (path_exists(final_path))
    {
        throw std::runtime_error("refusing to overwrite EXX compression dump: " + final_path);
    }
}

inline void publish_exclusive(const std::string& temporary_path, const std::string& final_path)
{
    if (link(temporary_path.c_str(), final_path.c_str()) != 0)
    {
        throw std::runtime_error("failed to atomically publish EXX compression dump " + final_path + ": "
                                 + std::strerror(errno));
    }
    if (unlink(temporary_path.c_str()) != 0)
    {
        const int unlink_errno = errno;
        std::remove(final_path.c_str());
        throw std::runtime_error("failed to remove EXX compression dump temporary file: "
                                 + std::string(std::strerror(unlink_errno)));
    }
}

inline void write_text_durable(const std::string& path, const std::string& text)
{
    int descriptor = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (descriptor < 0)
    {
        throw std::runtime_error("failed to create EXX compression dump temporary manifest: "
                                 + std::string(std::strerror(errno)));
    }
    try
    {
        std::size_t written = 0;
        while (written != text.size())
        {
            const ssize_t result = write(descriptor, text.data() + written, text.size() - written);
            if (result < 0 && errno == EINTR)
            {
                continue;
            }
            if (result <= 0)
            {
                throw std::runtime_error("failed to write EXX compression dump manifest: "
                                         + std::string(std::strerror(errno)));
            }
            written += static_cast<std::size_t>(result);
        }
        if (fsync(descriptor) != 0)
        {
            throw std::runtime_error("failed to sync EXX compression dump manifest: "
                                     + std::string(std::strerror(errno)));
        }
        if (close(descriptor) != 0)
        {
            descriptor = -1;
            throw std::runtime_error("failed to close EXX compression dump manifest: "
                                     + std::string(std::strerror(errno)));
        }
        descriptor = -1;
    }
    catch (...)
    {
        if (descriptor >= 0)
        {
            close(descriptor);
        }
        throw;
    }
}

inline void publish_text_exclusive(const std::string& final_path, const std::string& text)
{
    const std::string temp_path = temporary_path(final_path, 0);
    TemporaryFile temporary_file(temp_path);
    write_text_durable(temp_path, text);
    publish_exclusive(temp_path, final_path);
    temporary_file.release();
}

inline void replace_text_atomically(const std::string& final_path, const std::string& text)
{
    const std::string temp_path = temporary_path(final_path, 0);
    TemporaryFile temporary_file(temp_path);
    write_text_durable(temp_path, text);
    if (rename(temp_path.c_str(), final_path.c_str()) != 0)
    {
        throw std::runtime_error("failed to atomically replace EXX compression dump manifest: "
                                 + std::string(std::strerror(errno)));
    }
    temporary_file.release();
}

inline unsigned long long file_size(const std::string& path)
{
    struct stat status;
    if (stat(path.c_str(), &status) != 0 || status.st_size < 0)
    {
        throw std::runtime_error("failed to inspect EXX compression dump file: " + path);
    }
    return static_cast<unsigned long long>(status.st_size);
}

template <class T>
unsigned long long block_count(const ExxCompressionIO::TensorMap<T>& tensor_map)
{
    unsigned long long blocks = 0;
    for (const auto& outer: tensor_map)
    {
        if (outer.second.size() > std::numeric_limits<unsigned long long>::max() - blocks)
        {
            throw std::runtime_error("EXX compression dump block count overflows");
        }
        blocks += static_cast<unsigned long long>(outer.second.size());
    }
    return blocks;
}

inline std::string object_state(const std::string& object)
{
    const std::size_t separator = object.rfind('.');
    return separator == std::string::npos ? "unspecified" : object.substr(separator + 1);
}

inline bool counts_for_compression_metrics(const std::string& object)
{
    return object_state(object) == "active";
}

inline void append_manifest_entry(const std::string& dump_root,
                                  const std::string& object,
                                  const int spin,
                                  const std::string& channel,
                                  const unsigned long long blocks,
                                  const unsigned long long bytes)
{
    const std::string manifest_path = dump_root + "/manifest.txt";
    const std::string old_text = read_text_file(manifest_path);
    const ParsedManifest parsed = parse_manifest(old_text);
    if (parsed.state != "incomplete")
    {
        throw std::runtime_error("cannot append to a completed EXX compression dump manifest");
    }
    std::ostringstream entry;
    entry << "ENTRY\n"
          << "utc=" << utc_time() << '\n'
          << "object=" << object << '\n'
          << "state=" << object_state(object) << '\n'
          << "spin=" << spin << '\n'
          << "channel=" << channel << '\n'
          << "global_block_count=" << blocks << '\n'
          << "global_serialized_bytes=" << bytes << '\n'
          << "counts_for_compression_metrics=" << (counts_for_compression_metrics(object) ? "yes" : "no") << '\n'
          << "END_ENTRY\n";
    replace_text_atomically(manifest_path, old_text + entry.str());
}

template <class T>
double scalar_real(const T& value)
{
    return static_cast<double>(value);
}

template <class T>
double scalar_real(const std::complex<T>& value)
{
    return static_cast<double>(value.real());
}

template <class T>
double scalar_imaginary(const T&)
{
    return 0.0;
}

template <class T>
double scalar_imaginary(const std::complex<T>& value)
{
    return static_cast<double>(value.imag());
}

} // namespace detail

inline std::string map_path(const std::string& dump_root,
                            const std::string& object,
                            const int spin,
                            const std::string& channel,
                            const int rank)
{
    detail::validate_component(object, "object");
    detail::validate_component(channel, "channel");
    return dump_root + "/" + object + ".spin" + std::to_string(spin) + "." + channel + ".rank"
           + detail::rank_string(rank) + ".exxcmp";
}

inline std::string scalar_path(const std::string& dump_root,
                               const std::string& object,
                               const int spin,
                               const std::string& channel,
                               const int rank)
{
    detail::validate_component(object, "object");
    detail::validate_component(channel, "channel");
    return dump_root + "/" + object + ".spin" + std::to_string(spin) + "." + channel + ".rank"
           + detail::rank_string(rank) + ".scalar";
}

template <class T>
std::string scalar_type_name();

template <>
inline std::string scalar_type_name<double>()
{
    return "real64";
}

template <>
inline std::string scalar_type_name<std::complex<double>>()
{
    return "complex128";
}

inline bool initialize(const ManifestContext& context, MPI_Comm comm)
{
    const std::string dump_root = root();
    if (dump_root.empty())
    {
        return false;
    }

    const int local_initialized = detail::is_initialized_root(dump_root) ? 1 : 0;
    int all_initialized = 0;
    int any_initialized = 0;
    MPI_Allreduce(&local_initialized, &all_initialized, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&local_initialized, &any_initialized, 1, MPI_INT, MPI_MAX, comm);
    if (all_initialized != any_initialized)
    {
        throw std::runtime_error("inconsistent per-rank EXX compression dump initialization state");
    }
    if (all_initialized != 0)
    {
        return false;
    }

    int mpi_size = 1;
    MPI_Comm_size(comm, &mpi_size);
    detail::rank_zero_action(comm, [&]() {
        const bool created = detail::create_directories(dump_root);
        const std::string manifest_path = dump_root + "/manifest.txt";
        const std::string expected_context = detail::manifest_context_text(context, mpi_size);
        if (created)
        {
            const std::string manifest_text
                = expected_context + "session_state=incomplete\ncreated_utc=" + detail::utc_time() + '\n';
            detail::publish_text_exclusive(manifest_path, manifest_text);
        }
        else
        {
            if (!detail::path_exists(manifest_path))
            {
                throw std::runtime_error("existing EXX compression dump root has no manifest: " + dump_root);
            }
            const detail::ParsedManifest parsed = detail::parse_manifest(detail::read_text_file(manifest_path));
            if (parsed.context != expected_context)
            {
                throw std::runtime_error("EXX compression dump manifest does not match the current calculation");
            }
            throw std::runtime_error("EXX compression dump root already contains a " + parsed.state
                                     + " snapshot; use a new root");
        }
    });
    MPI_Barrier(comm);

    bool registered = false;
    std::string local_error;
    try
    {
        registered = detail::register_initialized_root(dump_root);
        if (!registered)
        {
            local_error = "EXX compression dump root was registered concurrently";
        }
    }
    catch (const std::exception& exception)
    {
        local_error = exception.what();
    }
    try
    {
        detail::collective_require(comm, local_error);
    }
    catch (...)
    {
        if (registered)
        {
            detail::unregister_initialized_root(dump_root);
        }
        throw;
    }
    return true;
}

inline bool begin_first_electronic_snapshot(MPI_Comm comm)
{
    const std::string dump_root = root();
    if (dump_root.empty())
    {
        return false;
    }
    int dump_this_update = 0;
    detail::rank_zero_action(comm, [&]() {
        const detail::ParsedManifest parsed
            = detail::parse_manifest(detail::read_text_file(dump_root + "/manifest.txt"));
        dump_this_update = parsed.state == "incomplete" ? 1 : 0;
    });
    MPI_Bcast(&dump_this_update, 1, MPI_INT, 0, comm);
    return dump_this_update != 0;
}

inline void mark_complete(MPI_Comm comm)
{
    const std::string dump_root = root();
    if (dump_root.empty())
    {
        return;
    }
    detail::rank_zero_action(comm, [&]() {
        const std::string manifest_path = dump_root + "/manifest.txt";
        const std::string old_text = detail::read_text_file(manifest_path);
        const detail::ParsedManifest parsed = detail::parse_manifest(old_text);
        if (parsed.state != "incomplete")
        {
            throw std::runtime_error("EXX compression dump snapshot is already complete");
        }
        const std::set<std::string> required_objects{"C.active",
                                                     "V.raw",
                                                     "V.active",
                                                     "D.raw",
                                                     "D.active",
                                                     "H.lri",
                                                     "E.lri",
                                                     "H.final",
                                                     "E.final"};
        for (const std::string& object: required_objects)
        {
            if (parsed.objects.count(object) == 0)
            {
                throw std::runtime_error("cannot complete EXX compression dump; missing object " + object);
            }
        }

        std::string completed_text = old_text;
        const std::string incomplete_line = "session_state=incomplete\n";
        const std::size_t state_position = completed_text.find(incomplete_line);
        if (state_position == std::string::npos)
        {
            throw std::runtime_error("malformed EXX compression dump incomplete state");
        }
        completed_text.replace(state_position, incomplete_line.size(), "session_state=complete\n");
        completed_text += "completed_utc=" + detail::utc_time() + '\n';
        detail::replace_text_atomically(manifest_path, completed_text);
    });
}

template <class T>
void write_if_enabled(const std::string& object,
                      const ExxCompressionIO::TensorMap<T>& tensor_map,
                      const int spin,
                      const std::string& channel,
                      MPI_Comm comm)
{
    const std::string dump_root = root();
    if (dump_root.empty())
    {
        return;
    }
    int rank = 0;
    int mpi_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpi_size);
    const std::string final_path = map_path(dump_root, object, spin, channel, rank);
    const std::string temp_path = detail::temporary_path(final_path, rank);
    detail::TemporaryFile temporary_file(temp_path);

    std::string local_error;
    try
    {
        detail::ensure_final_absent(final_path);
        ExxCompressionIO::write_map(temp_path, tensor_map, rank, mpi_size);
    }
    catch (const std::exception& exception)
    {
        local_error = exception.what();
    }
    detail::collective_require(comm, local_error);

    local_error.clear();
    bool published = false;
    try
    {
        detail::publish_exclusive(temp_path, final_path);
        temporary_file.release();
        published = true;
    }
    catch (const std::exception& exception)
    {
        local_error = exception.what();
    }
    try
    {
        detail::collective_require(comm, local_error);
    }
    catch (...)
    {
        if (published)
        {
            std::remove(final_path.c_str());
        }
        throw;
    }

    unsigned long long local_blocks = 0;
    unsigned long long local_bytes = 0;
    local_error.clear();
    try
    {
        local_blocks = detail::block_count(tensor_map);
        local_bytes = detail::file_size(final_path);
    }
    catch (const std::exception& exception)
    {
        local_error = exception.what();
    }
    try
    {
        detail::collective_require(comm, local_error);
    }
    catch (...)
    {
        std::remove(final_path.c_str());
        throw;
    }
    unsigned long long global_blocks = 0;
    unsigned long long global_bytes = 0;
    MPI_Reduce(&local_blocks, &global_blocks, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&local_bytes, &global_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm);
    try
    {
        detail::rank_zero_action(comm, [&]() {
            detail::append_manifest_entry(dump_root, object, spin, channel, global_blocks, global_bytes);
        });
    }
    catch (...)
    {
        std::remove(final_path.c_str());
        throw;
    }
}

template <class T>
void write_scalar_if_enabled(const std::string& object,
                             const T& value,
                             const int spin,
                             const std::string& channel,
                             MPI_Comm comm)
{
    const std::string dump_root = root();
    if (dump_root.empty())
    {
        return;
    }
    const std::string final_path = scalar_path(dump_root, object, spin, channel, 0);
    unsigned long long scalar_bytes = 0;
    detail::rank_zero_action(comm, [&]() {
        detail::ensure_final_absent(final_path);
        const std::string temp_path = detail::temporary_path(final_path, 0);
        detail::TemporaryFile temporary_file(temp_path);
        std::ofstream output(temp_path, std::ios::out | std::ios::trunc);
        output << std::setprecision(17) << detail::scalar_real(value) << ' ' << detail::scalar_imaginary(value) << '\n';
        output.close();
        if (!output)
        {
            throw std::runtime_error("failed to write EXX compression scalar snapshot");
        }
        detail::publish_exclusive(temp_path, final_path);
        temporary_file.release();
        scalar_bytes = detail::file_size(final_path);
        try
        {
            detail::append_manifest_entry(dump_root, object, spin, channel, 0, scalar_bytes);
        }
        catch (...)
        {
            std::remove(final_path.c_str());
            throw;
        }
    });
}

} // namespace ExxCompressionDump
