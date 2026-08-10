#include "../exx_compression_io.h"

#include <RI/physics/Exx.h>
#include <RI/ri/Label.h>
#include <RI/ri/RI_Tools.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <map>
#include <mpi.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace ExxCompressionReplay
{

void replay_file(const std::string& path, MPI_Comm communicator);
bool replay_mode(int argc, char** argv);

namespace
{

using Scalar = std::complex<double>;
using TensorMap = ExxCompressionIO::TensorMap<Scalar>;
using Exx = RI::Exx<int, int, 3, Scalar>;

struct ReplayConfig
{
    std::array<int, 3> period{};
    std::array<std::array<double, 3>, 3> lattice{};
    std::map<int, std::array<double, 3>> atoms;
    std::string C_path;
    std::string V_path;
    std::string D_path;
    std::string D_post_path;
    bool D_raw = false;
    std::string H_out;
    std::string E_out;
    std::string D_full_out;
};

std::string trim(const std::string& value)
{
    const auto first = std::find_if_not(value.begin(), value.end(), [](const unsigned char character) {
        return std::isspace(character) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](const unsigned char character) {
                          return std::isspace(character) != 0;
                      }).base();
    return first < last ? std::string(first, last) : std::string{};
}

template <class T, std::size_t Count>
std::array<T, Count> parse_numbers(const std::string& value, const std::string& key, const int line_number)
{
    std::istringstream input(value);
    std::array<T, Count> result{};
    for (T& item: result)
    {
        if (!(input >> item))
        {
            throw std::runtime_error("config line " + std::to_string(line_number) + ": " + key + " requires exactly "
                                     + std::to_string(Count) + " values");
        }
    }
    std::string trailing;
    if (input >> trailing)
    {
        throw std::runtime_error("config line " + std::to_string(line_number) + ": trailing value for " + key);
    }
    return result;
}

void require_finite(const std::array<double, 3>& values, const std::string& key, const int line_number)
{
    if (!std::all_of(values.begin(), values.end(), [](const double value) { return std::isfinite(value); }))
    {
        throw std::runtime_error("config line " + std::to_string(line_number) + ": " + key + " values must be finite");
    }
}

ReplayConfig parse_config(const std::string& path)
{
    std::ifstream input(path);
    if (!input)
    {
        throw std::runtime_error("cannot open replay config: " + path);
    }

    ReplayConfig config;
    std::set<std::string> seen;
    std::set<int> atom_ids;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line))
    {
        ++line_number;
        const std::string stripped = trim(line);
        if (stripped.empty() || stripped[0] == '#')
        {
            continue;
        }
        const std::size_t separator = stripped.find_first_of(" \t");
        if (separator == std::string::npos)
        {
            throw std::runtime_error("config line " + std::to_string(line_number) + ": missing value");
        }
        const std::string key = stripped.substr(0, separator);
        const std::string value = trim(stripped.substr(separator + 1));
        if (value.empty())
        {
            throw std::runtime_error("config line " + std::to_string(line_number) + ": empty value for " + key);
        }

        if (key == "atom")
        {
            const auto fields = parse_numbers<double, 4>(value, key, line_number);
            if (!std::isfinite(fields[0]) || std::floor(fields[0]) != fields[0]
                || fields[0] < std::numeric_limits<int>::min() || fields[0] > std::numeric_limits<int>::max())
            {
                throw std::runtime_error("config line " + std::to_string(line_number) + ": atom id must be an integer");
            }
            const int atom = static_cast<int>(fields[0]);
            const std::array<double, 3> position{{fields[1], fields[2], fields[3]}};
            require_finite(position, key, line_number);
            if (!atom_ids.insert(atom).second)
            {
                throw std::runtime_error("config line " + std::to_string(line_number) + ": duplicate atom id");
            }
            config.atoms[atom] = position;
            continue;
        }

        const std::set<std::string> allowed{"period",
                                            "lattice_1",
                                            "lattice_2",
                                            "lattice_3",
                                            "C_path",
                                            "V_path",
                                            "D_path",
                                            "D_post_path",
                                            "D_state",
                                            "H_out",
                                            "E_out",
                                            "D_full_out"};
        if (!allowed.count(key))
        {
            throw std::runtime_error("config line " + std::to_string(line_number) + ": unknown key " + key);
        }
        if (!seen.insert(key).second)
        {
            throw std::runtime_error("config line " + std::to_string(line_number) + ": duplicate key " + key);
        }

        if (key == "period")
        {
            config.period = parse_numbers<int, 3>(value, key, line_number);
            if (std::any_of(config.period.begin(), config.period.end(), [](const int item) { return item <= 0; }))
            {
                throw std::runtime_error("config period values must be positive");
            }
        }
        else if (key.compare(0, 8, "lattice_") == 0)
        {
            const int row = key[8] - '1';
            config.lattice[static_cast<std::size_t>(row)] = parse_numbers<double, 3>(value, key, line_number);
            require_finite(config.lattice[static_cast<std::size_t>(row)], key, line_number);
        }
        else if (key == "D_state")
        {
            if (value != "active" && value != "raw")
            {
                throw std::runtime_error("D_state must be active or raw");
            }
            config.D_raw = value == "raw";
        }
        else
        {
            std::string* destination = nullptr;
            if (key == "C_path")
                destination = &config.C_path;
            if (key == "V_path")
                destination = &config.V_path;
            if (key == "D_path")
                destination = &config.D_path;
            if (key == "D_post_path")
                destination = &config.D_post_path;
            if (key == "H_out")
                destination = &config.H_out;
            if (key == "E_out")
                destination = &config.E_out;
            if (key == "D_full_out")
                destination = &config.D_full_out;
            *destination = value;
        }
    }
    if (!input.eof())
    {
        throw std::runtime_error("failed while reading replay config: " + path);
    }

    const std::set<std::string> required{"period",
                                         "lattice_1",
                                         "lattice_2",
                                         "lattice_3",
                                         "C_path",
                                         "V_path",
                                         "D_path",
                                         "D_state",
                                         "H_out",
                                         "E_out"};
    for (const std::string& key: required)
    {
        if (!seen.count(key))
        {
            throw std::runtime_error("missing replay config key: " + key);
        }
    }
    if (config.atoms.empty())
    {
        throw std::runtime_error("replay config requires at least one atom line");
    }
    if (config.D_raw != seen.count("D_full_out"))
    {
        throw std::runtime_error(config.D_raw ? "raw D_state requires D_full_out"
                                              : "active D_state does not accept D_full_out");
    }
    if (config.D_raw == seen.count("D_post_path"))
    {
        throw std::runtime_error(config.D_raw ? "raw D_state does not accept D_post_path"
                                              : "active D_state requires D_post_path");
    }
    std::vector<std::string> paths{config.C_path, config.V_path, config.D_path, config.H_out, config.E_out};
    if (config.D_raw)
        paths.push_back(config.D_full_out);
    else
        paths.push_back(config.D_post_path);
    std::set<std::string> unique_paths;
    for (const std::string& configured_path: paths)
    {
        if (!unique_paths.insert(configured_path).second)
            throw std::runtime_error("all replay input and output paths must be distinct");
    }
    return config;
}

std::uint32_t read_uint32_le(const std::array<unsigned char, 32>& bytes, const std::size_t offset)
{
    std::uint32_t value = 0;
    for (std::size_t byte = 0; byte != 4; ++byte)
    {
        value |= static_cast<std::uint32_t>(bytes[offset + byte]) << (8 * byte);
    }
    return value;
}

void validate_snapshot_header(const std::string& path, const std::string& name)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        throw std::runtime_error("cannot open " + name + " snapshot: " + path);
    }
    std::array<unsigned char, 32> header{};
    input.read(reinterpret_cast<char*>(header.data()), static_cast<std::streamsize>(header.size()));
    if (!input)
    {
        throw std::runtime_error("truncated " + name + " EXXCMP1 header");
    }
    const std::array<unsigned char, 8> magic{{'E', 'X', 'X', 'C', 'M', 'P', '1', 0}};
    if (!std::equal(magic.begin(), magic.end(), header.begin()) || read_uint32_le(header, 8) != 1)
    {
        throw std::runtime_error(name + " snapshot is not EXXCMP1 version 1");
    }
    if (header[12] != 2)
    {
        throw std::runtime_error(name + " snapshot must use complex128 scalars");
    }
    if (header[13] != 0 || header[14] != 0 || header[15] != 0)
    {
        throw std::runtime_error(name + " snapshot has nonzero reserved header bytes");
    }
    if (read_uint32_le(header, 16) != 0 || read_uint32_le(header, 20) != 1)
    {
        throw std::runtime_error(name + " snapshot must be the complete rank 0 of 1, not a distributed shard");
    }
}

class TemporaryOutputs
{
  public:
    explicit TemporaryOutputs(const ReplayConfig& config)
        : H_final(config.H_out), E_final(config.E_out), D_final(config.D_full_out)
    {
        require_absent(H_final);
        require_absent(E_final);
        if (!D_final.empty())
            require_absent(D_final);
        try
        {
            H_temporary = create_temporary(H_final);
            E_temporary = create_temporary(E_final);
            if (config.D_raw)
                D_temporary = create_temporary(D_final);
        }
        catch (...)
        {
            cleanup_temporary_files();
            throw;
        }
    }

    ~TemporaryOutputs()
    {
        cleanup();
    }

    void publish()
    {
        if (!D_temporary.empty())
            publish_one(D_temporary, D_final);
        publish_one(H_temporary, H_final);
        publish_one(E_temporary, E_final);
        cleanup_temporary_files();
        committed_ = true;
    }

    std::string H_final;
    std::string E_final;
    std::string D_final;
    std::string H_temporary;
    std::string E_temporary;
    std::string D_temporary;

  private:
    static std::string create_temporary(const std::string& final_name)
    {
        const std::string pattern = final_name + ".tmp.XXXXXX";
        std::vector<char> writable_pattern(pattern.begin(), pattern.end());
        writable_pattern.push_back('\0');
        const int descriptor = mkstemp(writable_pattern.data());
        if (descriptor < 0)
            throw std::runtime_error("cannot exclusively create replay temporary for " + final_name + ": "
                                     + std::strerror(errno));
        if (close(descriptor) != 0)
        {
            const int close_error = errno;
            std::remove(writable_pattern.data());
            throw std::runtime_error("cannot close replay temporary for " + final_name + ": "
                                     + std::strerror(close_error));
        }
        return writable_pattern.data();
    }

    static void require_absent(const std::string& path)
    {
        struct stat metadata
        {
        };
        if (lstat(path.c_str(), &metadata) == 0)
            throw std::runtime_error("replay output already exists: " + path);
        if (errno != ENOENT)
            throw std::runtime_error("cannot inspect replay output " + path + ": " + std::strerror(errno));
    }

    void publish_one(const std::string& source, const std::string& destination)
    {
        struct stat source_metadata
        {
        };
        if (lstat(source.c_str(), &source_metadata) != 0)
            throw std::runtime_error("cannot inspect replay temporary " + source + ": " + std::strerror(errno));
        if (link(source.c_str(), destination.c_str()) != 0)
        {
            throw std::runtime_error("failed to exclusively publish replay output " + destination + ": "
                                     + std::strerror(errno));
        }
        published_finals_.push_back({source, destination});
    }

    void cleanup()
    {
        if (!committed_)
        {
            for (auto published = published_finals_.rbegin(); published != published_finals_.rend(); ++published)
            {
                struct stat temporary_metadata
                {
                };
                struct stat final_metadata
                {
                };
                if (lstat(published->temporary.c_str(), &temporary_metadata) == 0
                    && lstat(published->final.c_str(), &final_metadata) == 0
                    && final_metadata.st_dev == temporary_metadata.st_dev
                    && final_metadata.st_ino == temporary_metadata.st_ino)
                    std::remove(published->final.c_str());
            }
        }
        cleanup_temporary_files();
    }

    void cleanup_temporary_files()
    {
        if (!H_temporary.empty())
            std::remove(H_temporary.c_str());
        if (!E_temporary.empty())
            std::remove(E_temporary.c_str());
        if (!D_temporary.empty())
            std::remove(D_temporary.c_str());
    }

    struct PublishedFinal
    {
        std::string temporary;
        std::string final;
    };

    bool committed_ = false;
    std::vector<PublishedFinal> published_finals_;
};

void require_finite_map(const TensorMap& tensors, const std::string& name)
{
    for (const auto& outer: tensors)
        for (const auto& inner: outer.second)
            for (std::size_t index = 0; index != inner.second.get_shape_all(); ++index)
            {
                const Scalar value = inner.second.ptr()[index];
                if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                    throw std::runtime_error(name + " snapshot contains a non-finite tensor value");
            }
}

void write_energy(const std::string& path, const Scalar& energy)
{
    std::ofstream output(path, std::ios::trunc);
    if (!output)
    {
        throw std::runtime_error("cannot open temporary replay energy output: " + path);
    }
    output << std::setprecision(17) << energy.real() << ' ' << energy.imag() << '\n';
    output.flush();
    if (!output)
    {
        throw std::runtime_error("failed to write replay energy output: " + path);
    }
}

void replay(const ReplayConfig& config, MPI_Comm communicator)
{
    int world_size = 0;
    if (MPI_Comm_size(communicator, &world_size) != MPI_SUCCESS)
    {
        throw std::runtime_error("MPI_Comm_size failed");
    }
    if (world_size != 1)
    {
        throw std::runtime_error("EXX compression replay requires MPI world size 1");
    }
    validate_snapshot_header(config.C_path, "C");
    validate_snapshot_header(config.V_path, "V");
    validate_snapshot_header(config.D_path, "D");
    if (!config.D_raw)
        validate_snapshot_header(config.D_post_path, "D.post");

    Exx exx;
    exx.set_parallel(communicator, config.atoms, config.lattice, config.period);
    exx.set_symmetry(false, {});
    const std::map<std::string, double> active_flags{{"flag_period", false},
                                                     {"flag_comm", false},
                                                     {"flag_filter", false}};
    {
        const TensorMap tensors = ExxCompressionIO::read_map<Scalar>(config.C_path);
        require_finite_map(tensors, "C");
        exx.lri.set_tensors_map2(tensors, {RI::Label::ab::a, RI::Label::ab::b}, active_flags, "Cs_");
    }
    {
        const TensorMap tensors = ExxCompressionIO::read_map<Scalar>(config.V_path);
        require_finite_map(tensors, "V");
        exx.lri.set_tensors_map2(tensors, {RI::Label::ab::a0b0}, active_flags, "Vs_");
    }
    std::map<std::string, double> density_flags = active_flags;
    density_flags["flag_period"] = config.D_raw;
    {
        const TensorMap tensors = ExxCompressionIO::read_map<Scalar>(config.D_path);
        require_finite_map(tensors, "D");
        exx.lri.set_tensors_map2(tensors,
                                 {RI::Label::ab::a1b1, RI::Label::ab::a1b2, RI::Label::ab::a2b1, RI::Label::ab::a2b2},
                                 density_flags,
                                 "Ds_");
        if (config.D_raw)
            exx.post_2D.saves["Ds_"] = exx.post_2D.set_tensors_map2(tensors);
    }
    if (!config.D_raw)
    {
        const TensorMap tensors = ExxCompressionIO::read_map<Scalar>(config.D_post_path);
        require_finite_map(tensors, "D.post");
        exx.post_2D.saves["Ds_"] = exx.post_2D.set_tensors_map2(tensors);
    }
    exx.flag_finish.Cs = true;
    exx.flag_finish.Vs = true;
    exx.flag_finish.Ds = true;
    exx.flag_finish.Ds_delta = false;
    const TensorMap& loaded_density = exx.lri.data_pool.at("Ds_").Ds_ab;
    require_finite_map(loaded_density, "loaded D");

    TemporaryOutputs outputs(config);
    if (config.D_raw)
    {
        ExxCompressionIO::write_map(outputs.D_temporary, loaded_density, 0, 1);
    }
    exx.cal_Hs({"", "", ""});
    const Scalar energy = exx.energy;
    require_finite_map(exx.Hs, "calculated H");
    if (!std::isfinite(energy.real()) || !std::isfinite(energy.imag()))
        throw std::runtime_error("calculated energy is non-finite");
    ExxCompressionIO::write_map(outputs.H_temporary, exx.Hs, 0, 1);
    write_energy(outputs.E_temporary, energy);
    outputs.publish();
}

} // namespace

void replay_file(const std::string& path, MPI_Comm communicator)
{
    replay(parse_config(path), communicator);
}

bool replay_mode(const int argc, char** argv)
{
    return argc == 2 && argv != nullptr && argv[1] != nullptr && argv[1][0] != '-';
}

} // namespace ExxCompressionReplay

namespace
{

using Scalar = std::complex<double>;
using TensorMap = ExxCompressionIO::TensorMap<Scalar>;
using Exx = RI::Exx<int, int, 3, Scalar>;

class TemporaryDirectory
{
  public:
    TemporaryDirectory()
    {
        static std::atomic<unsigned long> next_id{0};
        path_ = testing::TempDir() + "exx_compression_replay_" + std::to_string(getpid()) + "_"
                + std::to_string(next_id++);
        if (mkdir(path_.c_str(), 0700) != 0)
        {
            throw std::runtime_error("failed to create replay test directory");
        }
    }

    ~TemporaryDirectory()
    {
        for (const char* name: {"C.exxcmp",
                                "V.exxcmp",
                                "D.exxcmp",
                                "D.post.exxcmp",
                                "D.full.exxcmp",
                                "H.exxcmp",
                                "E.scalar",
                                "replay.conf",
                                "bad.conf"})
        {
            std::remove(file(name).c_str());
        }
        rmdir(path_.c_str());
    }

    std::string file(const std::string& name) const
    {
        return path_ + "/" + name;
    }

  private:
    std::string path_;
};

RI::Tensor<Scalar> tensor(const RI::Shape_Vector& shape, const std::initializer_list<Scalar>& values)
{
    RI::Tensor<Scalar> result(shape);
    if (result.get_shape_all() != values.size())
    {
        throw std::runtime_error("invalid replay test tensor initializer");
    }
    std::copy(values.begin(), values.end(), result.ptr());
    return result;
}

TensorMap make_C()
{
    TensorMap result;
    result[0][{0, {0, 0, 0}}] = tensor({1, 2, 2}, {{0.7, 0.1}, {-0.2, 0.3}, {0.4, -0.5}, {0.6, 0.2}});
    result[0][{0, {-1, 0, 0}}] = tensor({1, 2, 2}, {{-0.1, 0.2}, {0.3, 0.4}, {0.2, -0.1}, {-0.5, 0.3}});
    return result;
}

TensorMap make_V()
{
    TensorMap result;
    result[0][{0, {0, 0, 0}}] = tensor({1, 1}, {{1.4, 0.0}});
    result[0][{0, {-1, 0, 0}}] = tensor({1, 1}, {{0.25, 0.0}});
    return result;
}

TensorMap make_D_active()
{
    TensorMap result;
    result[0][{0, {0, 0, 0}}] = tensor({2, 2}, {{0.8, 0.0}, {0.1, 0.2}, {0.1, -0.2}, {0.6, 0.0}});
    result[0][{0, {-1, 0, 0}}] = tensor({2, 2}, {{0.15, 0.0}, {-0.03, 0.04}, {-0.03, -0.04}, {0.1, 0.0}});
    return result;
}

TensorMap make_D_raw()
{
    TensorMap result = make_D_active();
    result[0][{0, {1, 0, 0}}] = tensor({2, 2}, {{0.02, 0.0}, {0.01, -0.005}, {0.01, 0.005}, {0.03, 0.0}});
    return result;
}

TensorMap make_D_post()
{
    TensorMap result = make_D_active();
    result[0][{0, {0, 0, 0}}] = tensor({2, 2}, {{0.45, 0.0}, {-0.08, 0.12}, {-0.08, -0.12}, {0.25, 0.0}});
    result[0][{0, {-1, 0, 0}}] = tensor({2, 2}, {{-0.06, 0.0}, {0.02, -0.03}, {0.02, 0.03}, {0.04, 0.0}});
    return result;
}

void initialize_exx(Exx& exx)
{
    const std::map<int, std::array<double, 3>> atoms{{0, {{0.0, 0.0, 0.0}}}};
    const std::array<std::array<double, 3>, 3> lattice{{{{1.0, 0.0, 0.0}}, {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}}};
    exx.set_parallel(MPI_COMM_WORLD, atoms, lattice, {{2, 1, 1}});
    exx.set_symmetry(false, {});
}

std::pair<TensorMap, Scalar> direct_result(const TensorMap& active_density,
                                           const TensorMap& post_density,
                                           const bool raw)
{
    Exx exx;
    initialize_exx(exx);
    const std::map<std::string, double> inactive_flags{{"flag_period", false},
                                                       {"flag_comm", false},
                                                       {"flag_filter", false}};
    exx.lri.set_tensors_map2(make_C(), {RI::Label::ab::a, RI::Label::ab::b}, inactive_flags, "Cs_");
    exx.lri.set_tensors_map2(make_V(), {RI::Label::ab::a0b0}, inactive_flags, "Vs_");
    auto density_flags = inactive_flags;
    density_flags["flag_period"] = raw;
    exx.lri.set_tensors_map2(active_density,
                             {RI::Label::ab::a1b1, RI::Label::ab::a1b2, RI::Label::ab::a2b1, RI::Label::ab::a2b2},
                             density_flags,
                             "Ds_");
    exx.flag_finish.Cs = exx.flag_finish.Vs = exx.flag_finish.Ds = true;
    exx.flag_finish.Ds_delta = false;
    exx.post_2D.saves["Ds_"] = exx.post_2D.set_tensors_map2(post_density);
    exx.cal_Hs({"", "", ""});
    return {exx.Hs, exx.energy};
}

void write_config(const TemporaryDirectory& directory, const std::string& density_state)
{
    std::ofstream output(directory.file("replay.conf"));
    output << "period 2 1 1\n"
           << "lattice_1 1 0 0\n"
           << "lattice_2 0 1 0\n"
           << "lattice_3 0 0 1\n"
           << "atom 0 0 0 0\n"
           << "C_path " << directory.file("C.exxcmp") << "\n"
           << "V_path " << directory.file("V.exxcmp") << "\n"
           << "D_path " << directory.file("D.exxcmp") << "\n"
           << "D_state " << density_state << "\n"
           << "H_out " << directory.file("H.exxcmp") << "\n"
           << "E_out " << directory.file("E.scalar") << "\n";
    if (density_state == "raw")
    {
        output << "D_full_out " << directory.file("D.full.exxcmp") << "\n";
    }
    else
    {
        output << "D_post_path " << directory.file("D.post.exxcmp") << "\n";
    }
}

Scalar read_energy(const std::string& path)
{
    std::ifstream input(path);
    double real = 0.0;
    double imag = 0.0;
    input >> real >> imag;
    if (!input)
    {
        throw std::runtime_error("invalid replay energy output");
    }
    input >> std::ws;
    if (!input.eof())
        throw std::runtime_error("invalid replay energy output");
    return {real, imag};
}

void expect_maps_near(const TensorMap& actual, const TensorMap& expected, const double tolerance)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (const auto& outer: expected)
    {
        ASSERT_TRUE(actual.count(outer.first));
        ASSERT_EQ(actual.at(outer.first).size(), outer.second.size());
        for (const auto& inner: outer.second)
        {
            ASSERT_TRUE(actual.at(outer.first).count(inner.first));
            const auto& value = actual.at(outer.first).at(inner.first);
            ASSERT_EQ(value.shape.size(), inner.second.shape.size());
            for (std::size_t dimension = 0; dimension != value.shape.size(); ++dimension)
            {
                ASSERT_EQ(value.shape[dimension], inner.second.shape[dimension]);
            }
            for (std::size_t index = 0; index != value.get_shape_all(); ++index)
            {
                EXPECT_LE(std::abs(value.ptr()[index] - inner.second.ptr()[index]), tolerance);
            }
        }
    }
}

TEST(ExxCompressionReplayTest, ActiveSnapshotsMatchDirectLibRI)
{
    TemporaryDirectory directory;
    const TensorMap active_density = make_D_active();
    const TensorMap post_density = make_D_post();
    ExxCompressionIO::write_map(directory.file("C.exxcmp"), make_C(), 0, 1);
    ExxCompressionIO::write_map(directory.file("V.exxcmp"), make_V(), 0, 1);
    ExxCompressionIO::write_map(directory.file("D.exxcmp"), active_density, 0, 1);
    ExxCompressionIO::write_map(directory.file("D.post.exxcmp"), post_density, 0, 1);
    write_config(directory, "active");
    const auto expected = direct_result(active_density, post_density, false);
    const auto wrong_active_post = direct_result(active_density, active_density, false);
    ASSERT_GT(std::abs(expected.second - wrong_active_post.second), 1.0e-4);

    ExxCompressionReplay::replay_file(directory.file("replay.conf"), MPI_COMM_WORLD);

    expect_maps_near(ExxCompressionIO::read_map<Scalar>(directory.file("H.exxcmp")), expected.first, 1.0e-13);
    EXPECT_LE(std::abs(read_energy(directory.file("E.scalar")) - expected.second), 1.0e-13);
    EXPECT_FALSE(std::ifstream(directory.file("D.full.exxcmp")).good());
}

TEST(ExxCompressionReplayTest, RawDensityWritesActuallyPeriodizedMapAndMatchesDirectLibRI)
{
    TemporaryDirectory directory;
    const TensorMap density = make_D_raw();
    ExxCompressionIO::write_map(directory.file("C.exxcmp"), make_C(), 0, 1);
    ExxCompressionIO::write_map(directory.file("V.exxcmp"), make_V(), 0, 1);
    ExxCompressionIO::write_map(directory.file("D.exxcmp"), density, 0, 1);
    write_config(directory, "raw");
    const auto expected = direct_result(density, density, true);

    ExxCompressionReplay::replay_file(directory.file("replay.conf"), MPI_COMM_WORLD);

    const TensorMap full_density = ExxCompressionIO::read_map<Scalar>(directory.file("D.full.exxcmp"));
    expect_maps_near(full_density, RI::RI_Tools::cal_period(density, std::array<int, 3>{{2, 1, 1}}), 1.0e-13);
    expect_maps_near(ExxCompressionIO::read_map<Scalar>(directory.file("H.exxcmp")), expected.first, 1.0e-13);
    EXPECT_LE(std::abs(read_energy(directory.file("E.scalar")) - expected.second), 1.0e-13);
}

TEST(ExxCompressionReplayTest, ConfigRejectsDuplicateUnknownAndMissingKeys)
{
    TemporaryDirectory directory;
    for (const std::string& invalid_tail: {"period 2 1 1\n", "unknown value\n", ""})
    {
        std::ofstream output(directory.file("bad.conf"));
        output << "period 2 1 1\n"
               << "lattice_1 1 0 0\n"
               << "lattice_2 0 1 0\n"
               << "lattice_3 0 0 1\n"
               << "atom 0 0 0 0\n"
               << "C_path C\nV_path V\nD_path D\nD_state active\nH_out H\n"
               << invalid_tail;
        output.close();
        EXPECT_THROW(ExxCompressionReplay::replay_file(directory.file("bad.conf"), MPI_COMM_WORLD), std::runtime_error);
    }
}

TEST(ExxCompressionReplayTest, ConfigEnforcesPostDensityStateAndPathSeparation)
{
    TemporaryDirectory directory;
    const auto write_bad_config = [&](const std::string& state_lines, const std::string& H_path) {
        std::ofstream output(directory.file("bad.conf"));
        output << "period 2 1 1\n"
               << "lattice_1 1 0 0\n"
               << "lattice_2 0 1 0\n"
               << "lattice_3 0 0 1\n"
               << "atom 0 0 0 0\n"
               << "C_path C\nV_path V\nD_path D\n"
               << state_lines << "H_out " << H_path << "\nE_out E\n";
    };

    write_bad_config("D_state active\n", "H");
    EXPECT_THROW(ExxCompressionReplay::parse_config(directory.file("bad.conf")), std::runtime_error);

    write_bad_config("D_state raw\nD_post_path D.post\nD_full_out D.full\n", "H");
    EXPECT_THROW(ExxCompressionReplay::parse_config(directory.file("bad.conf")), std::runtime_error);

    write_bad_config("D_state active\nD_post_path D.post\n", "C");
    EXPECT_THROW(ExxCompressionReplay::parse_config(directory.file("bad.conf")), std::runtime_error);
}

TEST(ExxCompressionReplayTest, RejectsDistributedShardBeforeCreatingOutputs)
{
    TemporaryDirectory directory;
    ExxCompressionIO::write_map(directory.file("C.exxcmp"), make_C(), 0, 2);
    ExxCompressionIO::write_map(directory.file("V.exxcmp"), make_V(), 0, 1);
    ExxCompressionIO::write_map(directory.file("D.exxcmp"), make_D_active(), 0, 1);
    ExxCompressionIO::write_map(directory.file("D.post.exxcmp"), make_D_raw(), 0, 1);
    write_config(directory, "active");

    EXPECT_THROW(ExxCompressionReplay::replay_file(directory.file("replay.conf"), MPI_COMM_WORLD), std::runtime_error);
    EXPECT_FALSE(std::ifstream(directory.file("H.exxcmp")).good());
    EXPECT_FALSE(std::ifstream(directory.file("E.scalar")).good());
}

TEST(ExxCompressionReplayTest, RejectsDistributedPostDensityHeader)
{
    TemporaryDirectory directory;
    ExxCompressionIO::write_map(directory.file("C.exxcmp"), make_C(), 0, 1);
    ExxCompressionIO::write_map(directory.file("V.exxcmp"), make_V(), 0, 1);
    ExxCompressionIO::write_map(directory.file("D.exxcmp"), make_D_active(), 0, 1);
    ExxCompressionIO::write_map(directory.file("D.post.exxcmp"), make_D_raw(), 0, 2);
    write_config(directory, "active");

    EXPECT_THROW(ExxCompressionReplay::replay_file(directory.file("replay.conf"), MPI_COMM_WORLD), std::runtime_error);
    EXPECT_FALSE(std::ifstream(directory.file("H.exxcmp")).good());
    EXPECT_FALSE(std::ifstream(directory.file("E.scalar")).good());
}

TEST(ExxCompressionReplayTest, RejectsNonfiniteInputBeforeCreatingOutputs)
{
    TemporaryDirectory directory;
    TensorMap coefficient = make_C();
    coefficient.begin()->second.begin()->second.ptr()[0] = Scalar{std::numeric_limits<double>::quiet_NaN(), 0.0};
    ExxCompressionIO::write_map(directory.file("C.exxcmp"), coefficient, 0, 1);
    ExxCompressionIO::write_map(directory.file("V.exxcmp"), make_V(), 0, 1);
    ExxCompressionIO::write_map(directory.file("D.exxcmp"), make_D_active(), 0, 1);
    ExxCompressionIO::write_map(directory.file("D.post.exxcmp"), make_D_raw(), 0, 1);
    write_config(directory, "active");

    EXPECT_THROW(ExxCompressionReplay::replay_file(directory.file("replay.conf"), MPI_COMM_WORLD), std::runtime_error);
    EXPECT_FALSE(std::ifstream(directory.file("H.exxcmp")).good());
    EXPECT_FALSE(std::ifstream(directory.file("E.scalar")).good());
}

TEST(ExxCompressionReplayTest, RejectsExistingFinalWithoutOverwritingIt)
{
    TemporaryDirectory directory;
    ExxCompressionIO::write_map(directory.file("C.exxcmp"), make_C(), 0, 1);
    ExxCompressionIO::write_map(directory.file("V.exxcmp"), make_V(), 0, 1);
    ExxCompressionIO::write_map(directory.file("D.exxcmp"), make_D_active(), 0, 1);
    ExxCompressionIO::write_map(directory.file("D.post.exxcmp"), make_D_raw(), 0, 1);
    write_config(directory, "active");
    {
        std::ofstream existing(directory.file("H.exxcmp"));
        existing << "owned by another run\n";
    }

    EXPECT_THROW(ExxCompressionReplay::replay_file(directory.file("replay.conf"), MPI_COMM_WORLD), std::runtime_error);
    std::ifstream preserved(directory.file("H.exxcmp"));
    std::string contents;
    std::getline(preserved, contents);
    EXPECT_EQ(contents, "owned by another run");
    EXPECT_FALSE(std::ifstream(directory.file("E.scalar")).good());
}

TEST(ExxCompressionReplayTest, PublishFailureRollsBackOnlyOutputsCreatedByThisRun)
{
    TemporaryDirectory directory;
    ExxCompressionReplay::ReplayConfig config;
    config.D_raw = true;
    config.D_full_out = directory.file("D.full.exxcmp");
    config.H_out = directory.file("H.exxcmp");
    config.E_out = directory.file("E.scalar");
    {
        ExxCompressionReplay::TemporaryOutputs outputs(config);
        std::ofstream(outputs.D_temporary) << "new D\n";
        std::ofstream(outputs.H_temporary) << "new H\n";
        std::ofstream(outputs.E_temporary) << "new E\n";
        std::ofstream(config.H_out) << "external H\n";
        EXPECT_THROW(outputs.publish(), std::runtime_error);
        EXPECT_TRUE(std::ifstream(outputs.D_temporary).good());
        EXPECT_TRUE(std::ifstream(outputs.H_temporary).good());
        EXPECT_TRUE(std::ifstream(outputs.E_temporary).good());
        EXPECT_TRUE(std::ifstream(config.D_full_out).good());
    }
    EXPECT_FALSE(std::ifstream(config.D_full_out).good());
    std::ifstream preserved(config.H_out);
    std::string contents;
    std::getline(preserved, contents);
    EXPECT_EQ(contents, "external H");
    EXPECT_FALSE(std::ifstream(config.E_out).good());
}

TEST(ExxCompressionReplayTest, TemporaryOutputsAreExclusiveAndDistinct)
{
    TemporaryDirectory directory;
    ExxCompressionReplay::ReplayConfig config;
    config.D_raw = true;
    config.D_full_out = directory.file("D.full.exxcmp");
    config.H_out = directory.file("H.exxcmp");
    config.E_out = directory.file("E.scalar");
    {
        ExxCompressionReplay::TemporaryOutputs first(config);
        ExxCompressionReplay::TemporaryOutputs second(config);
        EXPECT_NE(first.D_temporary, second.D_temporary);
        EXPECT_NE(first.H_temporary, second.H_temporary);
        EXPECT_NE(first.E_temporary, second.E_temporary);
        for (const std::string& path: {first.D_temporary,
                                       first.H_temporary,
                                       first.E_temporary,
                                       second.D_temporary,
                                       second.H_temporary,
                                       second.E_temporary})
            EXPECT_TRUE(std::ifstream(path).good());
    }
}

TEST(ExxCompressionReplayTest, GoogleTestFlagsRemainInTestMode)
{
    char executable[] = "replay";
    char gtest_filter[] = "--gtest_filter=ExxCompressionReplayTest.*";
    char config[] = "replay.conf";
    char* test_arguments[] = {executable, gtest_filter};
    char* replay_arguments[] = {executable, config};
    EXPECT_FALSE(ExxCompressionReplay::replay_mode(2, test_arguments));
    EXPECT_TRUE(ExxCompressionReplay::replay_mode(2, replay_arguments));
}

} // namespace

int main(int argc, char** argv)
{
    int initialized = 0;
    int finalized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS || MPI_Finalized(&finalized) != MPI_SUCCESS || finalized != 0)
    {
        std::fprintf(stderr, "error: MPI runtime is unavailable or already finalized\n");
        return 2;
    }
    const bool owns_mpi = initialized == 0;
    if (owns_mpi && MPI_Init(&argc, &argv) != MPI_SUCCESS)
    {
        std::fprintf(stderr, "error: MPI_Init failed\n");
        return 2;
    }

    int status = 0;
    try
    {
        if (argc == 1 || (argc >= 2 && std::string(argv[1]).compare(0, 8, "--gtest_") == 0))
        {
            testing::InitGoogleTest(&argc, argv);
            status = RUN_ALL_TESTS();
        }
        else if (ExxCompressionReplay::replay_mode(argc, argv))
        {
            ExxCompressionReplay::replay_file(argv[1], MPI_COMM_WORLD);
        }
        else
        {
            throw std::runtime_error("usage: exx_compression_replay [replay.conf]");
        }
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "error: %s\n", error.what());
        status = 2;
    }

    if (owns_mpi && MPI_Finalize() != MPI_SUCCESS && status == 0)
    {
        std::fprintf(stderr, "error: MPI_Finalize failed\n");
        status = 2;
    }
    return status;
}
