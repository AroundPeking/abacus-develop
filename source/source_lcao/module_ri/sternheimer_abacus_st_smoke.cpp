#include "source_lcao/module_ri/sternheimer_abacus_st_smoke.h"

#include "source_basis/module_ao/ORB_read.h"
#include "source_basis/module_ao/element_basis_index-ORB.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/unitcell.h"
#include "source_estate/elecstate.h"
#include "source_estate/module_pot/potential_new.h"
#include "source_hamilt/module_xc/exx_info.h"
#include "source_io/module_bessel/numerical_basis.h"
#include "source_io/module_parameter/parameter.h"
#include "source_lcao/module_gint/gint_atom.h"
#include "source_lcao/module_ri/Matrix_Orbs11.h"
#include "source_lcao/module_ri/conv_coulomb_pot_k.h"
#include "source_lcao/module_ri/exx_abfs-construct_orbs.h"
#include "source_lcao/module_ri/exx_abfs-io.h"
#include "source_lcao/module_ri/sternheimer_abfs_perturbation.h"
#include "source_lcao/module_ri/sternheimer_channel_parallel.h"
#include "source_lcao/module_ri/sternheimer_channel_resources.h"
#include "source_lcao/module_ri/sternheimer_chi0_mpi.h"
#include "source_lcao/module_ri/sternheimer_coulomb_whitening.h"
#include "source_lcao/module_ri/sternheimer_delta.h"
#include "source_lcao/module_ri/sternheimer_fd_solver.h"
#include "source_lcao/module_ri/sternheimer_rpa.h"
#include "source_lcao/module_ri/sternheimer_siab_mpi.h"
#include "source_lcao/module_ri/sternheimer_siab_fixed_ao.h"
#include "source_lcao/module_ri/sternheimer_siab_overlap.h"
#include "source_lcao/module_ri/sternheimer_siab_primitive_galerkin.h"
#include "source_lcao/module_ri/sternheimer_siab_provenance.h"
#include "source_lcao/module_ri/sternheimer_siab_writer.h"
#include "source_pw/module_pwdft/structure_factor.h"
#include "source_base/module_external/blas_connector.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <map>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef COMMIT_INFO
#include "commit.h"
#endif

#ifdef __MPI
#include <mpi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ModuleRI
{
namespace
{

namespace siab = ::module_ri::sternheimer_siab;
namespace sternheimer_chi0 = ::module_ri::sternheimer_chi0;

constexpr const char* kSmokeEnv = "ABACUS_STERNHEIMER_FD_ST_SMOKE";
constexpr const char* kOutputEnv = "ABACUS_STERNHEIMER_FD_ST_OUT";
constexpr const char* kBandsEnv = "ABACUS_STERNHEIMER_FD_ST_BANDS";
constexpr const char* kChannelsEnv = "ABACUS_STERNHEIMER_FD_ST_CHANNELS";
constexpr const char* kMaxDenseEnv = "ABACUS_STERNHEIMER_FD_ST_MAX_DENSE";
constexpr const char* kLanczosSubspaceEnv = "ABACUS_STERNHEIMER_FD_ST_LANCZOS_SUBSPACE";
constexpr const char* kOmegaEnv = "ABACUS_STERNHEIMER_FD_ST_OMEGA";
constexpr const char* kSolverToleranceEnv = "ABACUS_STERNHEIMER_FD_ST_SOLVER_TOL";
constexpr const char* kSolverMaxIterEnv = "ABACUS_STERNHEIMER_FD_ST_MAX_ITER";
constexpr const char* kPCAThresholdEnv = "ABACUS_STERNHEIMER_FD_ST_PCA_THRESHOLD";
constexpr const char* kCCPRmeshTimesEnv = "ABACUS_STERNHEIMER_FD_ST_CCP_RMESH_TIMES";
constexpr const char* kOrbitalDirEnv = "ABACUS_STERNHEIMER_FD_ST_ORBITAL_DIR";
constexpr const char* kOrbitalFilesEnv = "ABACUS_STERNHEIMER_FD_ST_ORBITAL_FILES";
constexpr const char* kFrequencyRankShiftEnv = "ABACUS_STERNHEIMER_FD_ST_FREQ_RANK_SHIFT";
constexpr const char* kChannelMaxWorkersEnv = "ABACUS_STERNHEIMER_CHANNEL_MAX_WORKERS";
constexpr const char* kLCAOVirtualSourceEnv = "ABACUS_STERNHEIMER_DELTA_VIRTUAL_SOURCE";
constexpr double kHartreeToRydberg = 2.0;

void hash_u64(siab::Sha256& digest, const std::uint64_t value)
{
    std::array<unsigned char, 8> bytes;
    for (std::size_t index = 0; index != bytes.size(); ++index)
    {
        bytes[index] = static_cast<unsigned char>((value >> (56U - 8U * index)) & 0xffU);
    }
    digest.update(bytes.data(), bytes.size());
}

void hash_int(siab::Sha256& digest, const int value)
{
    hash_u64(digest, static_cast<std::uint64_t>(static_cast<std::int64_t>(value)));
}

void hash_double(siab::Sha256& digest, const double value)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t), "Sternheimer SIAB provenance requires binary64 doubles.");
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    hash_u64(digest, bits);
}

void hash_string(siab::Sha256& digest, const std::string& value)
{
    hash_u64(digest, static_cast<std::uint64_t>(value.size()));
    digest.update(reinterpret_cast<const unsigned char*>(value.data()), value.size());
}

std::string hash_effective_auxiliary_channels(const std::vector<SternheimerABFGridChannel>& channels)
{
    if (channels.empty())
    {
        throw std::runtime_error("Cannot hash an empty generated Sternheimer ABFS channel basis.");
    }
    siab::Sha256 digest;
    const std::string marker = "ABACUS_STERNHEIMER_GENERATED_ABFS_CHANNELS_V1";
    digest.update(reinterpret_cast<const unsigned char*>(marker.data()), marker.size());
    hash_u64(digest, static_cast<std::uint64_t>(channels.size()));
    for (const SternheimerABFGridChannel& channel: channels)
    {
        hash_int(digest, channel.channel_index);
        hash_int(digest, channel.atom_index);
        hash_int(digest, channel.atom_local_index);
        hash_int(digest, channel.type_index);
        hash_int(digest, channel.angular_momentum);
        hash_int(digest, channel.radial_index);
        hash_int(digest, channel.magnetic_index);
        hash_string(digest, channel.label);
        hash_u64(digest, static_cast<std::uint64_t>(channel.potential_r.size()));
        for (const double value: channel.potential_r)
        {
            if (!std::isfinite(value))
            {
                throw std::runtime_error("Generated Sternheimer ABFS channel contains a non-finite grid value.");
            }
            hash_double(digest, value);
        }
    }
    return digest.finish();
}

std::string hash_coulomb_whitening_transform(const SternheimerCoulombWhitening& whitening)
{
    siab::Sha256 digest;
    const std::string marker = "ABACUS_STERNHEIMER_COULOMB_WHITENING_V1";
    digest.update(reinterpret_cast<const unsigned char*>(marker.data()), marker.size());
    hash_int(digest, whitening.raw_dimension);
    hash_int(digest, whitening.retained_rank);
    hash_double(digest, whitening.relative_threshold);
    for (const double value: whitening.transform)
    {
        if (!std::isfinite(value))
        {
            throw std::runtime_error("Sternheimer Coulomb whitening transform contains a non-finite value.");
        }
        hash_double(digest, value);
    }
    return digest.finish();
}

std::string compiled_commit_metadata()
{
#ifdef COMMIT_INFO
    return COMMIT;
#else
    return std::string();
#endif
}

std::vector<double> cell_vectors_bohr(const UnitCell& ucell)
{
    return {ucell.lat0 * ucell.latvec.e11,
            ucell.lat0 * ucell.latvec.e12,
            ucell.lat0 * ucell.latvec.e13,
            ucell.lat0 * ucell.latvec.e21,
            ucell.lat0 * ucell.latvec.e22,
            ucell.lat0 * ucell.latvec.e23,
            ucell.lat0 * ucell.latvec.e31,
            ucell.lat0 * ucell.latvec.e32,
            ucell.lat0 * ucell.latvec.e33};
}

std::string lower_string(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool env_is_true(const char* name)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr)
    {
        return false;
    }
    const std::string value = lower_string(raw);
    return !(value.empty() || value == "0" || value == "false" || value == "off" || value == "no");
}

int positive_int_from_env(const char* name, const int default_value)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr)
    {
        return default_value;
    }
    std::size_t parsed = 0;
    const int value = std::stoi(raw, &parsed);
    if (raw[parsed] != '\0' || value <= 0)
    {
        throw std::invalid_argument(std::string("Invalid positive integer in ") + name + ".");
    }
    return value;
}

int int_from_env(const char* name, const int default_value)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr)
    {
        return default_value;
    }
    std::size_t parsed = 0;
    const int value = std::stoi(raw, &parsed);
    if (raw[parsed] != '\0')
    {
        throw std::invalid_argument(std::string("Invalid integer in ") + name + ".");
    }
    return value;
}

int sternheimer_channel_openmp_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

double positive_double_from_env(const char* name, const double default_value)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr)
    {
        return default_value;
    }
    std::size_t parsed = 0;
    const double value = std::stod(raw, &parsed);
    if (raw[parsed] != '\0' || value <= 0.0)
    {
        throw std::invalid_argument(std::string("Invalid positive floating-point value in ") + name + ".");
    }
    return value;
}

double nonnegative_double_from_env(const char* name, const double default_value)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr)
    {
        return default_value;
    }
    std::size_t parsed = 0;
    const double value = std::stod(raw, &parsed);
    if (raw[parsed] != '\0' || value < 0.0)
    {
        throw std::invalid_argument(std::string("Invalid non-negative floating-point value in ") + name + ".");
    }
    return value;
}

std::uint64_t slurm_memory_limit_bytes()
{
    const char* raw = std::getenv("SLURM_MEM_PER_NODE");
    if (raw == nullptr || raw[0] == '\0')
    {
        return 0;
    }
    char* end = nullptr;
    const unsigned long long memory_mib = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || memory_mib == 0
        || memory_mib > std::numeric_limits<std::uint64_t>::max() / (1024U * 1024U))
    {
        return 0;
    }
    return static_cast<std::uint64_t>(memory_mib) * 1024U * 1024U;
}

std::string default_report_path(const std::string& output_dir)
{
    const char* explicit_path = std::getenv(kOutputEnv);
    if (explicit_path != nullptr && explicit_path[0] != '\0')
    {
        return explicit_path;
    }
    if (output_dir.empty())
    {
        return "STERNHEIMER_FD_ST.dat";
    }
    if (output_dir.back() == '/')
    {
        return output_dir + "STERNHEIMER_FD_ST.dat";
    }
    return output_dir + "/STERNHEIMER_FD_ST.dat";
}

std::string join_output_path(const std::string& output_dir, const std::string& filename)
{
    if (output_dir.empty())
    {
        return filename;
    }
    if (output_dir.back() == '/')
    {
        return output_dir + filename;
    }
    return output_dir + "/" + filename;
}

std::string chi0_status_path(const std::string& output_dir)
{
    return join_output_path(output_dir,
                            PARAM.inp.out_sternheimer_siab ? "STERNHEIMER_SIAB_STATUS.dat"
                                                          : "STERNHEIMER_CHI0.dat");
}

std::string chi0_v1_filename(const int iq, const int ifrequency, const int rank = GlobalV::MY_RANK)
{
    std::ostringstream out;
    out << "v1_sternheimer_chi0_iq_" << iq << "_ifreq_" << ifrequency << "_rank" << rank << ".dat";
    return out.str();
}

std::string chi0_progress_filename(const int rank = GlobalV::MY_RANK)
{
    std::ostringstream out;
    out << (PARAM.inp.out_sternheimer_siab ? "STERNHEIMER_SIAB_PROGRESS_rank"
                                          : "STERNHEIMER_CHI0_PROGRESS_rank")
        << rank << ".dat";
    return out.str();
}

double elapsed_seconds_since(const std::chrono::steady_clock::time_point& start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void reset_chi0_progress_file()
{
    std::ofstream out(chi0_progress_filename().c_str());
    out << std::setprecision(16);
    out << "event rank ifrequency owner_rank band channel solved_equations converged iterations "
           "solver_relative_residual equation_residual_norm elapsed_s note\n";
}

void append_chi0_progress_event(const std::string& event,
                                const int ifrequency,
                                const int owner_rank,
                                const int band,
                                const int channel,
                                const int solved_equations,
                                const SternheimerRPA::SolverResult* solver_result,
                                const double equation_residual_norm,
                                const double elapsed_seconds,
                                const std::string& note)
{
    std::ofstream out(chi0_progress_filename().c_str(), std::ios::app);
    out << std::setprecision(16);
    out << event << ' ' << GlobalV::MY_RANK << ' ' << ifrequency << ' ' << owner_rank << ' ' << band << ' ' << channel
        << ' ' << solved_equations << ' ';
    if (solver_result == nullptr)
    {
        out << "- -1 -1 ";
    }
    else
    {
        out << (solver_result->converged ? "yes" : "no") << ' ' << solver_result->iterations << ' '
            << solver_result->relative_residual << ' ';
    }
    out << equation_residual_norm << ' ' << elapsed_seconds << ' ' << note << '\n';
}

int occupied_band_count(const elecstate::ElecState& elec_state, const int k_index)
{
    int count = 0;
    for (int ib = 0; ib != elec_state.wg.nc; ++ib)
    {
        if (elec_state.wg(k_index, ib) > 1.0e-8)
        {
            count = ib + 1;
        }
    }
    return count;
}

std::vector<double> eigenvalues_ry_from_elec_state(const elecstate::ElecState& elec_state, const int k_index)
{
    std::vector<double> eigenvalues;
    eigenvalues.reserve(static_cast<std::size_t>(elec_state.ekb.nc));
    for (int ib = 0; ib != elec_state.ekb.nc; ++ib)
    {
        eigenvalues.push_back(elec_state.ekb(k_index, ib));
    }
    return eigenvalues;
}

std::vector<double> occupations_from_elec_state(const elecstate::ElecState& elec_state, const int k_index)
{
    std::vector<double> occupations;
    occupations.reserve(static_cast<std::size_t>(elec_state.wg.nc));
    for (int ib = 0; ib != elec_state.wg.nc; ++ib)
    {
        occupations.push_back(elec_state.wg(k_index, ib));
    }
    return occupations;
}

std::vector<std::string> split_orbital_file_list(const std::string& raw)
{
    std::vector<std::string> files;
    std::string current;
    for (const char ch: raw)
    {
        if (ch == ',' || ch == ':')
        {
            if (!current.empty())
            {
                files.push_back(current);
                current.clear();
            }
        }
        else
        {
            current.push_back(ch);
        }
    }
    if (!current.empty())
    {
        files.push_back(current);
    }
    return files;
}

std::string join_path(const std::string& dir, const std::string& file)
{
    if (file.empty())
    {
        return file;
    }
    if (!file.empty() && file.front() == '/')
    {
        return file;
    }
    if (dir.empty() || dir.back() == '/')
    {
        return dir + file;
    }
    return dir + "/" + file;
}

void validate_orbital_files(const std::string& orbital_dir, const std::vector<std::string>& orbital_files)
{
    if (orbital_files.empty())
    {
        throw std::runtime_error("No numerical orbital files are available. Set NUMERICAL_ORBITAL in STRU or "
                                 "ABACUS_STERNHEIMER_FD_ST_ORBITAL_FILES.");
    }
    for (const std::string& file: orbital_files)
    {
        const std::string path = join_path(orbital_dir, file);
        std::ifstream in(path.c_str());
        if (!in)
        {
            throw std::runtime_error("Cannot open Sternheimer ST orbital file: " + path + ".");
        }
    }
}

std::vector<std::string> orbital_files_from_env_or_cell(const UnitCell& ucell)
{
    const char* raw = std::getenv(kOrbitalFilesEnv);
    if (raw != nullptr && raw[0] != '\0')
    {
        return split_orbital_file_list(raw);
    }
    return ucell.orbital_fn;
}

std::string orbital_dir_from_env_or_input()
{
    const char* raw = std::getenv(kOrbitalDirEnv);
    if (raw != nullptr)
    {
        return raw;
    }
    return PARAM.inp.orbital_dir;
}

void read_sternheimer_orbitals(const UnitCell& ucell, LCAO_Orbitals& orb)
{
    std::vector<std::string> orbital_files = orbital_files_from_env_or_cell(ucell);
    if (static_cast<int>(orbital_files.size()) != ucell.ntype)
    {
        throw std::runtime_error("Sternheimer ST requires one numerical orbital file per atom type.");
    }

    std::string orbital_dir = orbital_dir_from_env_or_input();
    const bool all_absolute = std::all_of(orbital_files.begin(), orbital_files.end(), [](const std::string& file) {
        return !file.empty() && file.front() == '/';
    });
    if (all_absolute)
    {
        orbital_dir.clear();
    }
    validate_orbital_files(orbital_dir, orbital_files);

    const double lcao_ecut = PARAM.inp.lcao_ecut > 0.0 ? PARAM.inp.lcao_ecut : PARAM.inp.ecutwfc;
    if (lcao_ecut <= 0.0)
    {
        throw std::runtime_error("Sternheimer ST requires positive lcao_ecut or ecutwfc to read orbitals.");
    }

    orb.init(GlobalV::ofs_running,
             ucell.ntype,
             orbital_dir,
             orbital_files.data(),
             ucell.descriptor_file,
             PARAM.inp.lmaxmax,
             lcao_ecut,
             PARAM.inp.lcao_dk,
             PARAM.inp.lcao_dr,
             PARAM.inp.lcao_rmax,
             PARAM.globalv.deepks_setorb,
             PARAM.inp.out_mat_r,
             PARAM.inp.cal_force,
             GlobalV::MY_RANK);
}

std::vector<std::string> resolved_sternheimer_response_orbital_files(const UnitCell& ucell)
{
    if (PARAM.inp.sternheimer_response_orbital_files.size()
        != static_cast<std::size_t>(ucell.ntype))
    {
        throw std::runtime_error("Sternheimer response space requires one numerical orbital file per atom type.");
    }
    return siab::resolve_required_input_files(PARAM.inp.orbital_dir,
                                               PARAM.inp.sternheimer_response_orbital_files,
                                               "response orbital");
}

void read_sternheimer_response_orbitals(const UnitCell& ucell, LCAO_Orbitals& orb)
{
    std::vector<std::string> orbital_files = resolved_sternheimer_response_orbital_files(ucell);
    const double lcao_ecut = PARAM.inp.lcao_ecut > 0.0 ? PARAM.inp.lcao_ecut : PARAM.inp.ecutwfc;
    if (lcao_ecut <= 0.0)
    {
        throw std::runtime_error("Sternheimer response orbitals require positive lcao_ecut or ecutwfc.");
    }
    orb.init(GlobalV::ofs_running,
             ucell.ntype,
             "",
             orbital_files.data(),
             ucell.descriptor_file,
             PARAM.inp.lmaxmax,
             lcao_ecut,
             PARAM.inp.lcao_dk,
             PARAM.inp.lcao_dr,
             PARAM.inp.lcao_rmax,
             PARAM.globalv.deepks_setorb,
             PARAM.inp.out_mat_r,
             PARAM.inp.cal_force,
             GlobalV::MY_RANK);
}

std::map<Conv_Coulomb_Pot_K::Coulomb_Type, std::vector<std::map<std::string, std::string>>>
    make_fock_hartree_coulomb_param()
{
    return {{Conv_Coulomb_Pot_K::Coulomb_Type::Fock, {{{"alpha", "1"}, {"singularity_correction", "limits"}}}}};
}

struct SternheimerABFBuildData
{
    std::vector<SternheimerABFGridChannel> channels;
    std::vector<double> full_coulomb_metric;
    std::vector<std::vector<SternheimerRadialPerturbation>> radials_by_type;
    std::vector<int> atom_types;
    std::vector<ModuleBase::Vector3<double>> atom_positions;
};

std::vector<double> build_molecular_coulomb_metric(
    const UnitCell& ucell,
    const LCAO_Orbitals& orb,
    const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs_ccp,
    const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs,
    const std::vector<SternheimerABFGridChannel>& channels)
{
    const ModuleBase::Element_Basis_Index::Range range_ccp
        = ModuleBase::Element_Basis_Index::construct_range(abfs_ccp);
    const ModuleBase::Element_Basis_Index::Range range_abfs
        = ModuleBase::Element_Basis_Index::construct_range(abfs);
    const ModuleBase::Element_Basis_Index::IndexLNM index_ccp
        = ModuleBase::Element_Basis_Index::construct_index(range_ccp);
    const ModuleBase::Element_Basis_Index::IndexLNM index_abfs
        = ModuleBase::Element_Basis_Index::construct_index(range_abfs);
    if (index_ccp.size() != index_abfs.size())
    {
        throw std::runtime_error("Sternheimer Coulomb metric ABFS type counts do not match.");
    }
    for (std::size_t type = 0; type != index_ccp.size(); ++type)
    {
        if (index_ccp[type].count_size != index_abfs[type].count_size)
        {
            throw std::runtime_error("Sternheimer Coulomb metric ABFS channel counts do not match.");
        }
    }

    std::vector<std::size_t> atom_offsets;
    std::vector<int> atom_types;
    std::vector<int> atom_local_indices;
    atom_offsets.reserve(static_cast<std::size_t>(ucell.nat));
    atom_types.reserve(static_cast<std::size_t>(ucell.nat));
    atom_local_indices.reserve(static_cast<std::size_t>(ucell.nat));
    std::size_t total_dimension = 0;
    for (int type = 0; type != ucell.ntype; ++type)
    {
        for (int atom = 0; atom != ucell.atoms[type].na; ++atom)
        {
            atom_offsets.push_back(total_dimension);
            atom_types.push_back(type);
            atom_local_indices.push_back(atom);
            total_dimension += index_abfs[static_cast<std::size_t>(type)].count_size;
        }
    }
    if (total_dimension != channels.size())
    {
        throw std::runtime_error("Sternheimer Coulomb metric dimension does not match sampled ABFS channels.");
    }
    for (const SternheimerABFGridChannel& channel: channels)
    {
        const std::size_t atom = static_cast<std::size_t>(channel.atom_index);
        if (atom >= atom_offsets.size()
            || atom_offsets[atom] + static_cast<std::size_t>(channel.atom_local_index)
                   != static_cast<std::size_t>(channel.channel_index))
        {
            throw std::runtime_error("Sternheimer Coulomb metric channel ordering is inconsistent with ABFS indices.");
        }
    }

    Matrix_Orbs11 metric_builder;
    metric_builder.init(abfs_ccp, abfs, ucell, orb, GlobalC::exx_info.info_ri.kmesh_times);
    metric_builder.init_radial_table();
    std::vector<double> metric(total_dimension * total_dimension, 0.0);
    for (std::size_t atom_a = 0; atom_a != atom_offsets.size(); ++atom_a)
    {
        const std::size_t type_a = static_cast<std::size_t>(atom_types[atom_a]);
        const auto& tau_a = ucell.atoms[type_a].tau[static_cast<std::size_t>(atom_local_indices[atom_a])];
        const std::size_t count_a = index_ccp[type_a].count_size;
        for (std::size_t atom_b = 0; atom_b != atom_offsets.size(); ++atom_b)
        {
            const std::size_t type_b = static_cast<std::size_t>(atom_types[atom_b]);
            const auto& tau_b = ucell.atoms[type_b].tau[static_cast<std::size_t>(atom_local_indices[atom_b])];
            const std::size_t count_b = index_abfs[type_b].count_size;
            const RI::Tensor<double> block
                = metric_builder.cal_overlap_matrix<double>(type_a,
                                                             type_b,
                                                             tau_a,
                                                             tau_b,
                                                             index_ccp,
                                                             index_abfs,
                                                             Matrix_Orbs11::Matrix_Order::AB);
            for (std::size_t local_a = 0; local_a != count_a; ++local_a)
            {
                for (std::size_t local_b = 0; local_b != count_b; ++local_b)
                {
                    metric[(atom_offsets[atom_a] + local_a) * total_dimension
                           + atom_offsets[atom_b] + local_b]
                        = block(local_a, local_b);
                }
            }
        }
    }
    return metric;
}

SternheimerABFBuildData build_abfs_ccp_data(const UnitCell& ucell,
                                             const SternheimerFDHamiltonian::Grid& grid,
                                             const int max_channels,
                                             const double pca_threshold,
                                             const double ccp_rmesh_times,
                                             const bool build_coulomb_metric)
{
    LCAO_Orbitals orb;
    read_sternheimer_orbitals(ucell, orb);
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs;
    if (sternheimer_builds_product_pca_auxiliary_basis(GlobalC::exx_info.info_ri.files_abfs))
    {
        auto lcaos = Exx_Abfs::Construct_Orbs::change_orbs(orb, GlobalC::exx_info.info_ri.kmesh_times);
        Exx_Abfs::Construct_Orbs::filter_empty_orbs(lcaos);
        abfs = Exx_Abfs::Construct_Orbs::abfs_same_atom(ucell,
                                                        orb,
                                                        lcaos,
                                                        GlobalC::exx_info.info_ri.kmesh_times,
                                                        pca_threshold);
    }
    else
    {
        abfs = Exx_Abfs::IO::construct_abfs(orb,
                                            GlobalC::exx_info.info_ri.files_abfs,
                                            GlobalC::exx_info.info_ri.kmesh_times);
    }
    Exx_Abfs::Construct_Orbs::filter_empty_orbs(abfs);
    const auto abfs_ccp = Conv_Coulomb_Pot_K::cal_orbs_ccp(abfs, make_fock_hartree_coulomb_param(), ccp_rmesh_times);
    auto radials_by_type = make_sternheimer_radial_perturbations_from_orbitals(abfs_ccp);

    std::vector<int> atom_types;
    std::vector<ModuleBase::Vector3<double>> atom_positions;
    atom_types.reserve(ucell.nat);
    atom_positions.reserve(ucell.nat);
    for (int it = 0; it != ucell.ntype; ++it)
    {
        const Atom& atom = ucell.atoms[it];
        for (int ia = 0; ia != atom.na; ++ia)
        {
            atom_types.push_back(it);
            atom_positions.push_back(atom.tau[ia] * ucell.lat0);
        }
    }

    SternheimerABFBuildData result;
    result.radials_by_type = std::move(radials_by_type);
    result.atom_types = std::move(atom_types);
    result.atom_positions = std::move(atom_positions);
    result.channels = build_coulomb_metric
                          ? describe_sternheimer_abf_grid_channels(result.radials_by_type,
                                                                   result.atom_types,
                                                                   result.atom_positions,
                                                                   max_channels)
                          : sample_sternheimer_abf_grid_channels(result.radials_by_type,
                                                                 result.atom_types,
                                                                 result.atom_positions,
                                                                 grid,
                                                                 max_channels);
    if (build_coulomb_metric)
    {
        if (max_channels > 0)
        {
            throw std::invalid_argument("A full Sternheimer Coulomb metric cannot be built for truncated channels.");
        }
        result.full_coulomb_metric = build_molecular_coulomb_metric(ucell, orb, abfs_ccp, abfs, result.channels);
    }
    return result;
}

std::vector<SternheimerABFGridChannel> build_abfs_ccp_grid_channels(const UnitCell& ucell,
                                                                    const SternheimerFDHamiltonian::Grid& grid,
                                                                    const int max_channels,
                                                                    const double pca_threshold,
                                                                    const double ccp_rmesh_times)
{
    return build_abfs_ccp_data(ucell, grid, max_channels, pca_threshold, ccp_rmesh_times, false).channels;
}

struct SIABPrimitiveExportData
{
    std::vector<siab::PrimitiveBlock> blocks;
    std::vector<std::vector<std::complex<double>>> full_values;
    std::vector<std::complex<double>> reciprocal_matrix;
    std::vector<std::complex<double>> overlap_s;
    std::unique_ptr<ModulePW::PW_Basis_K> serial_pw_basis;
    int primitive_count = 0;
    int reciprocal_count = 0;
};

SIABPrimitiveExportData build_siab_primitive_export_data(const ModulePW::PW_Basis& response_pw_basis,
                                                         const Structure_Factor& structure_factor,
                                                         const UnitCell& ucell,
                                                         const bool include_full_grid_values)
{
    siab::require_single_primitive_rcut(PARAM.inp.bessel_nao_rcuts);
    SIABPrimitiveExportData result;
    result.serial_pw_basis.reset(new ModulePW::PW_Basis_K("cpu", "double"));
#ifdef __MPI
    result.serial_pw_basis->initmpi(1, 0, MPI_COMM_SELF);
#endif
    result.serial_pw_basis->initgrids(ucell.lat0,
                                      ucell.latvec,
                                      response_pw_basis.nx,
                                      response_pw_basis.ny,
                                      response_pw_basis.nz);
    const ModuleBase::Vector3<double> gamma(0.0, 0.0, 0.0);
    result.serial_pw_basis->initparameters(false, PARAM.inp.ecutwfc, 1, &gamma);
    result.serial_pw_basis->fft_bundle.initfftmode(PARAM.inp.fft_mode);
    result.serial_pw_basis->setuptransform();
    result.serial_pw_basis->collect_local_pw(PARAM.inp.erf_ecut, PARAM.inp.erf_height, PARAM.inp.erf_sigma);
    if (result.serial_pw_basis->nxyz != response_pw_basis.nxyz
        || result.serial_pw_basis->nrxx != response_pw_basis.nxyz || result.serial_pw_basis->nks != 1)
    {
        throw std::runtime_error("Sternheimer SIAB serial primitive FFT basis does not cover the complete response grid.");
    }
    const Numerical_Basis::SIABPrimitiveParameters parameters
        = Numerical_Basis::siab_parameters_from_input(0, PARAM.inp.sternheimer_siab_lmax);
    Numerical_Basis numerical_basis;
    const auto reciprocal_blocks = numerical_basis.siab_primitive_reciprocal_values(
        0, result.serial_pw_basis.get(), structure_factor, ucell, parameters);
    if (reciprocal_blocks.empty())
    {
        throw std::runtime_error("Sternheimer SIAB primitive construction returned no blocks.");
    }
    for (const auto& block: reciprocal_blocks)
    {
        siab::PrimitiveBlock output_block;
        output_block.element = block.element;
        output_block.atom_index = block.atom_index;
        output_block.l = block.l;
        output_block.m = block.m;
        output_block.n_primitive = block.n_primitive;
        output_block.offset = block.offset;
        result.blocks.push_back(std::move(output_block));
        for (const auto& primitive: block.values)
        {
            if (result.reciprocal_count == 0)
            {
                result.reciprocal_count = static_cast<int>(primitive.size());
            }
            if (primitive.size() != static_cast<std::size_t>(result.reciprocal_count))
            {
                throw std::runtime_error("Sternheimer SIAB reciprocal primitive sizes are inconsistent.");
            }
            result.reciprocal_matrix.insert(result.reciprocal_matrix.end(), primitive.begin(), primitive.end());
            ++result.primitive_count;
        }
    }
    if (result.primitive_count <= 0 || result.reciprocal_count != result.serial_pw_basis->npwk[0])
    {
        throw std::runtime_error("Sternheimer SIAB reciprocal primitive matrix is empty or incomplete.");
    }
    if (GlobalV::MY_RANK == 0)
    {
        result.overlap_s.assign(static_cast<std::size_t>(result.primitive_count)
                                    * static_cast<std::size_t>(result.primitive_count),
                                ModuleBase::ZERO);
        BlasConnector::gemm('N',
                            'C',
                            result.primitive_count,
                            result.primitive_count,
                            result.reciprocal_count,
                            std::complex<double>(1.0, 0.0),
                            result.reciprocal_matrix.data(),
                            result.reciprocal_count,
                            result.reciprocal_matrix.data(),
                            result.reciprocal_count,
                            std::complex<double>(0.0, 0.0),
                            result.overlap_s.data(),
                            result.primitive_count);
    }

    if (include_full_grid_values)
    {
        std::unique_ptr<ModulePW::PW_Basis_K> primitive_pw_basis
            = Numerical_Basis::siab_complete_gamma_pw_basis(response_pw_basis, parameters.ecut_ry);
        if (primitive_pw_basis->nks <= 0
            || primitive_pw_basis->nx != response_pw_basis.nx
            || primitive_pw_basis->ny != response_pw_basis.ny
            || primitive_pw_basis->nz != response_pw_basis.nz
            || primitive_pw_basis->nxy != response_pw_basis.nxy
            || primitive_pw_basis->nxyz != response_pw_basis.nxyz
            || primitive_pw_basis->startz_current != 0
            || primitive_pw_basis->nplane != primitive_pw_basis->nz
            || primitive_pw_basis->nrxx != primitive_pw_basis->nxyz)
        {
            throw std::runtime_error(
                "Single-rank Sternheimer SIAB primitive basis must cover the complete uniform grid.");
        }
        auto grid_blocks = numerical_basis.siab_primitive_grid_values(
            0, primitive_pw_basis.get(), structure_factor, ucell, parameters);
        if (grid_blocks.size() != result.blocks.size())
        {
            throw std::runtime_error("Sternheimer SIAB reciprocal and grid primitive block counts differ.");
        }
        for (std::size_t iblock = 0; iblock != grid_blocks.size(); ++iblock)
        {
            auto& block = grid_blocks[iblock];
            const auto& metadata = result.blocks[iblock];
            if (block.atom_index != metadata.atom_index || block.l != metadata.l || block.m != metadata.m
                || block.n_primitive != metadata.n_primitive || block.offset != metadata.offset)
            {
                throw std::runtime_error("Sternheimer SIAB reciprocal and grid primitive metadata differ.");
            }
            for (auto& primitive: block.values)
            {
                result.full_values.push_back(std::move(primitive));
            }
        }
        if (result.full_values.size() != static_cast<std::size_t>(result.primitive_count))
        {
            throw std::runtime_error("Sternheimer SIAB full-grid primitive matrix is incomplete.");
        }
    }
    return result;
}

std::vector<std::complex<double>> project_siab_response_to_primitives(
    const std::vector<std::complex<double>>& complete_response,
    const UnitCell& ucell,
    const SIABPrimitiveExportData& primitives)
{
    if (!primitives.serial_pw_basis
        || complete_response.size() != static_cast<std::size_t>(primitives.serial_pw_basis->nrxx))
    {
        throw std::invalid_argument("Sternheimer SIAB response does not match its serial primitive FFT basis.");
    }
    std::vector<std::complex<double>> response_coefficients(
        static_cast<std::size_t>(primitives.serial_pw_basis->npwk[0]), ModuleBase::ZERO);
    // PW_Basis_K::real2recip reuses FFT scratch buffers stored in the basis.
#ifdef _OPENMP
#pragma omp critical(sternheimer_siab_real2recip)
#endif
    {
        primitives.serial_pw_basis->real2recip(complete_response.data(), response_coefficients.data(), 0);
    }
    const double coefficient_scale = std::sqrt(ucell.omega);
    for (std::complex<double>& value: response_coefficients)
    {
        value *= coefficient_scale;
    }
    return siab::overlap_q_reciprocal_contiguous(response_coefficients,
                                                 primitives.reciprocal_matrix,
                                                 primitives.primitive_count,
                                                 primitives.reciprocal_count);
}

siab::Provenance make_siab_production_provenance(const UnitCell& ucell,
                                                 const std::string& auxiliary_basis_sha256,
                                                 const SternheimerRPA::FrequencyGrid& frequency_grid,
                                                 const double pca_threshold,
                                                 const SternheimerCoulombWhitening& whitening)
{
    siab::Provenance provenance;
    provenance.abacus_commit = siab::require_source_commit(compiled_commit_metadata());
    provenance.auxiliary_basis_sha256 = auxiliary_basis_sha256;
    provenance.cell_bohr = cell_vectors_bohr(ucell);
    provenance.ecut_ry = PARAM.inp.ecutwfc;
    provenance.kernel = "full_coulomb";
    const std::vector<std::string> orbital_files
        = siab::resolve_required_input_files(orbital_dir_from_env_or_input(),
                                             orbital_files_from_env_or_cell(ucell),
                                             "initial orbital");
    const std::vector<std::string> pseudopotential_files
        = siab::resolve_required_input_files(PARAM.inp.pseudo_dir, ucell.pseudo_fn, "pseudopotential");
    provenance.orbital_sha256 = siab::sha256_file_manifest(orbital_files);
    provenance.pseudopotential_sha256 = siab::sha256_file_manifest(pseudopotential_files);
    provenance.spin_convention = "spin_resolved_occupation_in_reference_rows";
    provenance.executable_sha256 = siab::sha256_file(siab::resolve_executable_path());
    provenance.exx_pca_thr = pca_threshold;
    provenance.sternheimer_nfreq = static_cast<int>(frequency_grid.omega_ha.size());
    provenance.frequency_ha = frequency_grid.omega_ha;
    provenance.frequency_weights_ha = frequency_grid.weights_ha;
    provenance.mpi_ranks = GlobalV::NPROC;
    provenance.omp_threads = PARAM.globalv.nthread_per_proc;
    provenance.auxiliary_whitening = "global_full_coulomb_v1";
    provenance.raw_auxiliary_dimension = whitening.raw_dimension;
    provenance.whitened_auxiliary_rank = whitening.retained_rank;
    provenance.discarded_auxiliary_rank = whitening.discarded_rank;
    provenance.coulomb_relative_threshold = whitening.relative_threshold;
    provenance.coulomb_eigenvalues = whitening.eigenvalues;
    provenance.coulomb_max_orthonormality_error = whitening.max_orthonormality_error;
    provenance.coulomb_transform_sha256 = hash_coulomb_whitening_transform(whitening);
    return provenance;
}

siab::Provenance make_siab_production_provenance(
    const UnitCell& ucell,
    const std::vector<SternheimerABFGridChannel>& channels,
    const SternheimerRPA::FrequencyGrid& frequency_grid,
    const double pca_threshold)
{
    siab::Provenance provenance;
    provenance.abacus_commit = siab::require_source_commit(compiled_commit_metadata());
    provenance.auxiliary_basis_sha256 = hash_effective_auxiliary_channels(channels);
    provenance.cell_bohr = cell_vectors_bohr(ucell);
    provenance.ecut_ry = PARAM.inp.ecutwfc;
    provenance.kernel = "full_coulomb";
    const std::vector<std::string> orbital_files
        = siab::resolve_required_input_files(orbital_dir_from_env_or_input(),
                                             orbital_files_from_env_or_cell(ucell),
                                             "initial orbital");
    const std::vector<std::string> pseudopotential_files
        = siab::resolve_required_input_files(PARAM.inp.pseudo_dir, ucell.pseudo_fn, "pseudopotential");
    provenance.orbital_sha256 = siab::sha256_file_manifest(orbital_files);
    provenance.pseudopotential_sha256 = siab::sha256_file_manifest(pseudopotential_files);
    provenance.spin_convention = "spin_resolved_occupation_in_reference_rows";
    provenance.executable_sha256 = siab::sha256_file(siab::resolve_executable_path());
    provenance.exx_pca_thr = pca_threshold;
    provenance.sternheimer_nfreq = static_cast<int>(frequency_grid.omega_ha.size());
    provenance.frequency_ha = frequency_grid.omega_ha;
    provenance.frequency_weights_ha = frequency_grid.weights_ha;
    provenance.mpi_ranks = GlobalV::NPROC;
    provenance.omp_threads = PARAM.globalv.nthread_per_proc;
    return provenance;
}

std::vector<SternheimerDeltaGridFunction> build_lcao_candidate_grid_functions(
    const UnitCell& ucell,
    const SternheimerFDHamiltonian::Grid& grid,
    const LCAO_Orbitals* provided_orbitals = nullptr)
{
    LCAO_Orbitals loaded_orbitals;
    if (provided_orbitals == nullptr)
    {
        read_sternheimer_orbitals(ucell, loaded_orbitals);
        provided_orbitals = &loaded_orbitals;
    }
    const LCAO_Orbitals& orb = *provided_orbitals;
    const int grid_size = grid.size();
    constexpr int sample_chunk_size = 8192;
    const double lx = grid.nx * grid.hx;
    const double ly = grid.ny * grid.hy;
    const double lz = grid.nz * grid.hz;

    std::vector<SternheimerDeltaGridFunction> candidates;
    int atom_index = 0;
    for (int type = 0; type != ucell.ntype; ++type)
    {
        const Atom& unitcell_atom = ucell.atoms[type];
        const Numerical_Orbital& atom_orbitals = orb.Phi[type];
        Atom sampling_atom;
        sampling_atom.nwl = atom_orbitals.getLmax();
        sampling_atom.l_nchi.resize(static_cast<std::size_t>(sampling_atom.nwl + 1), 0);
        for (int angular_momentum = 0; angular_momentum <= sampling_atom.nwl; ++angular_momentum)
        {
            sampling_atom.l_nchi[static_cast<std::size_t>(angular_momentum)] = atom_orbitals.getNchi(angular_momentum);
            sampling_atom.nw
                += (2 * angular_momentum + 1) * sampling_atom.l_nchi[static_cast<std::size_t>(angular_momentum)];
        }
        sampling_atom.set_index();

        for (int ia = 0; ia != unitcell_atom.na; ++ia, ++atom_index)
        {
            const int orbital_count = sampling_atom.nw;
            const std::size_t candidate_begin = candidates.size();
            candidates.resize(candidate_begin + static_cast<std::size_t>(orbital_count));
            for (int iw = 0; iw != orbital_count; ++iw)
            {
                SternheimerDeltaGridFunction& candidate = candidates[candidate_begin + static_cast<std::size_t>(iw)];
                candidate.values.assign(static_cast<std::size_t>(grid_size),
                                        SternheimerFDHamiltonian::Complex(0.0, 0.0));
                for (SternheimerFDHamiltonian::Vector& gradient: candidate.gradients)
                {
                    gradient.assign(static_cast<std::size_t>(grid_size), SternheimerFDHamiltonian::Complex(0.0, 0.0));
                }
            }

            ModuleGint::GintAtom sampler(&sampling_atom,
                                         type,
                                         ia,
                                         atom_index,
                                         ModuleGint::Vec3i(0, 0, 0),
                                         ModuleGint::Vec3i(0, 0, 0),
                                         ModuleGint::Vec3d(0.0, 0.0, 0.0),
                                         &orb.Phi[type],
                                         &ucell);
            const ModuleBase::Vector3<double> atom_position = unitcell_atom.tau[ia] * ucell.lat0;
            const std::vector<std::array<int, 3>> periodic_images
                = enumerate_delta_sternheimer_periodic_images(grid,
                                                              {atom_position.x, atom_position.y, atom_position.z},
                                                              atom_orbitals.getRcut());
            for (int first = 0; first < grid_size; first += sample_chunk_size)
            {
                const int chunk_size = std::min(sample_chunk_size, grid_size - first);
                std::vector<ModuleGint::Vec3d> coordinates(static_cast<std::size_t>(chunk_size));
                const std::size_t buffer_size
                    = static_cast<std::size_t>(chunk_size) * static_cast<std::size_t>(orbital_count);
                std::vector<double> values(buffer_size, 0.0);
                std::vector<double> gradient_x(buffer_size, 0.0);
                std::vector<double> gradient_y(buffer_size, 0.0);
                std::vector<double> gradient_z(buffer_size, 0.0);
                for (const std::array<int, 3>& image: periodic_images)
                {
                    for (int local = 0; local != chunk_size; ++local)
                    {
                        const int linear = first + local;
                        const int ix = linear / (grid.ny * grid.nz);
                        const int remainder = linear % (grid.ny * grid.nz);
                        const int iy = remainder / grid.nz;
                        const int iz = remainder % grid.nz;
                        coordinates[static_cast<std::size_t>(local)]
                            = ModuleGint::Vec3d(ix * grid.hx - atom_position.x - image[0] * lx,
                                                iy * grid.hy - atom_position.y - image[1] * ly,
                                                iz * grid.hz - atom_position.z - image[2] * lz);
                    }
                    sampler.set_phi_dphi(coordinates,
                                         orbital_count,
                                         values.data(),
                                         gradient_x.data(),
                                         gradient_y.data(),
                                         gradient_z.data());
                    for (int local = 0; local != chunk_size; ++local)
                    {
                        const std::size_t grid_index = static_cast<std::size_t>(first + local);
                        for (int iw = 0; iw != orbital_count; ++iw)
                        {
                            const std::size_t buffer_index
                                = static_cast<std::size_t>(local) * static_cast<std::size_t>(orbital_count)
                                  + static_cast<std::size_t>(iw);
                            SternheimerDeltaGridFunction& candidate
                                = candidates[candidate_begin + static_cast<std::size_t>(iw)];
                            candidate.values[grid_index]
                                += SternheimerFDHamiltonian::Complex(values[buffer_index], 0.0);
                            candidate.gradients[0][grid_index]
                                += SternheimerFDHamiltonian::Complex(gradient_x[buffer_index], 0.0);
                            candidate.gradients[1][grid_index]
                                += SternheimerFDHamiltonian::Complex(gradient_y[buffer_index], 0.0);
                            candidate.gradients[2][grid_index]
                                += SternheimerFDHamiltonian::Complex(gradient_z[buffer_index], 0.0);
                        }
                    }
                }
            }
        }
    }
    return candidates;
}

std::vector<SternheimerResponseOrbitalAtomSpec> make_response_orbital_atom_specs(
    const UnitCell& ucell,
    const LCAO_Orbitals& response_orbitals)
{
    std::vector<SternheimerResponseOrbitalAtomSpec> atoms;
    atoms.reserve(static_cast<std::size_t>(ucell.nat));
    int atom_index = 0;
    for (int type = 0; type != ucell.ntype; ++type)
    {
        const Numerical_Orbital& type_orbitals = response_orbitals.Phi[type];
        std::vector<int> radial_counts(static_cast<std::size_t>(type_orbitals.getLmax() + 1), 0);
        for (int l = 0; l <= type_orbitals.getLmax(); ++l)
        {
            radial_counts[static_cast<std::size_t>(l)] = type_orbitals.getNchi(l);
        }
        for (int atom_local = 0; atom_local != ucell.atoms[type].na; ++atom_local, ++atom_index)
        {
            atoms.push_back(SternheimerResponseOrbitalAtomSpec{
                ucell.atoms[type].label,
                atom_index,
                radial_counts});
        }
    }
    return atoms;
}

std::vector<SternheimerDeltaGridFunction> reorder_response_orbital_grid_functions(
    std::vector<SternheimerDeltaGridFunction> sampled_functions,
    const SternheimerResponseOrbitalLayout& layout)
{
    if (sampled_functions.size() != layout.sampled_indices.size())
    {
        throw std::runtime_error("Sternheimer response-orbital layout does not match sampled AO count.");
    }
    std::vector<SternheimerDeltaGridFunction> ordered;
    ordered.reserve(sampled_functions.size());
    for (const std::size_t sampled_index: layout.sampled_indices)
    {
        if (sampled_index >= sampled_functions.size())
        {
            throw std::runtime_error("Sternheimer response-orbital layout contains an invalid sampled AO index.");
        }
        ordered.push_back(std::move(sampled_functions[sampled_index]));
    }
    return ordered;
}

std::vector<siab::AuxiliaryChannelMetadata> make_siab_auxiliary_metadata(
    const std::vector<SternheimerABFGridChannel>& channels)
{
    std::vector<siab::AuxiliaryChannelMetadata> metadata;
    metadata.reserve(channels.size());
    for (const SternheimerABFGridChannel& channel: channels)
    {
        metadata.push_back(siab::AuxiliaryChannelMetadata{
            channel.channel_index,
            channel.atom_index,
            channel.angular_momentum,
            channel.radial_index,
            sternheimer_physical_magnetic_index(channel.angular_momentum,
                                                 channel.magnetic_index),
            channel.label});
    }
    return metadata;
}

std::string write_sternheimer_fixed_ao_sidecar(
    const std::string& output_dir,
    const UnitCell& ucell,
    const std::vector<SternheimerABFGridChannel>& channels,
    const SternheimerRPA::FrequencyGrid& frequency_grid,
    const double pca_threshold,
    const SternheimerABACUSFDGridData& grid_data,
    const std::vector<SternheimerDeltaGridFunction>& sampled_ao_functions,
    const std::vector<std::vector<double>>& potentials,
    const SternheimerLCAOFixedAOMatrices& fixed_ao_matrices)
{
    const std::vector<siab::AuxiliaryChannelMetadata> auxiliary_metadata
        = make_siab_auxiliary_metadata(channels);
    std::vector<std::vector<std::complex<double>>> basis_values;
    basis_values.reserve(sampled_ao_functions.size());
    for (const SternheimerDeltaGridFunction& function: sampled_ao_functions)
    {
        basis_values.push_back(function.values);
    }
    const siab::FixedAOData fixed_ao_data
        = siab::build_fixed_ao_data(fixed_ao_matrices.n_basis,
                                    fixed_ao_matrices.overlap_s,
                                    fixed_ao_matrices.spins,
                                    auxiliary_metadata,
                                    basis_values,
                                    potentials,
                                    frequency_grid.omega_ha,
                                    frequency_grid.weights_ha,
                                    grid_data.volume_element);
    const siab::Provenance provenance
        = make_siab_production_provenance(ucell, channels, frequency_grid, pca_threshold);
    const std::string path = join_output_path(output_dir, "sternheimer_galerkin_fixed_ao.dat");
    siab::write_fixed_ao_v1(path, fixed_ao_data, provenance);
    GlobalV::ofs_running << " Sternheimer SIAB fixed-AO v1 output: " << path << std::endl;
    return path;
}

std::string write_sternheimer_primitive_galerkin_sidecar(
    const std::string& output_dir,
    const UnitCell& ucell,
    const std::vector<SternheimerABFGridChannel>& channels,
    const SternheimerRPA::FrequencyGrid& frequency_grid,
    const double pca_threshold,
    const SternheimerABACUSFDGridData& grid_data,
    const SIABPrimitiveExportData& primitives,
    const std::vector<SternheimerDeltaGridFunction>& sampled_ao_functions,
    const std::vector<std::vector<double>>& potentials,
    const SternheimerLCAOFixedAOMatrices& fixed_ao_matrices,
    const std::vector<SternheimerFDHamiltonian>& hamiltonians_ry)
{
    std::vector<std::vector<std::complex<double>>> fixed_ao_values;
    fixed_ao_values.reserve(sampled_ao_functions.size());
    for (const SternheimerDeltaGridFunction& function: sampled_ao_functions)
    {
        fixed_ao_values.push_back(function.values);
    }
    const siab::PrimitiveGalerkinData data = siab::build_primitive_galerkin_data(
        primitives.blocks,
        make_siab_auxiliary_metadata(channels),
        primitives.full_values,
        fixed_ao_values,
        fixed_ao_matrices.spins,
        hamiltonians_ry,
        potentials,
        frequency_grid.omega_ha,
        frequency_grid.weights_ha,
        grid_data.volume_element);
    const siab::Provenance provenance
        = make_siab_production_provenance(ucell,
                                          channels,
                                          frequency_grid,
                                          pca_threshold);
    const std::string path
        = join_output_path(output_dir, "sternheimer_galerkin_primitive.dat");
    siab::write_primitive_galerkin_v1(path, data, provenance);
    GlobalV::ofs_running << " Sternheimer SIAB primitive Galerkin v1 output: "
                         << path << std::endl;
    return path;
}

std::string write_sternheimer_response_galerkin_sidecar(
    const std::string& output_dir,
    const UnitCell& ucell,
    const std::vector<SternheimerABFGridChannel>& channels,
    const SternheimerRPA::FrequencyGrid& frequency_grid,
    const double pca_threshold,
    const SternheimerABACUSFDGridData& grid_data,
    const std::vector<siab::PrimitiveBlock>& response_blocks,
    const std::vector<SternheimerDeltaGridFunction>& response_functions,
    const std::vector<SternheimerDeltaGridFunction>& sampled_ao_functions,
    const std::vector<std::vector<double>>& potentials,
    const SternheimerLCAOFixedAOMatrices& fixed_ao_matrices,
    const std::vector<SternheimerFDHamiltonian>& hamiltonians_ry)
{
    std::vector<std::vector<std::complex<double>>> response_values;
    response_values.reserve(response_functions.size());
    for (const SternheimerDeltaGridFunction& function: response_functions)
    {
        response_values.push_back(function.values);
    }
    std::vector<std::vector<std::complex<double>>> fixed_ao_values;
    fixed_ao_values.reserve(sampled_ao_functions.size());
    for (const SternheimerDeltaGridFunction& function: sampled_ao_functions)
    {
        fixed_ao_values.push_back(function.values);
    }

    const siab::PrimitiveGalerkinData data = siab::build_primitive_galerkin_data(
        response_blocks,
        make_siab_auxiliary_metadata(channels),
        response_values,
        fixed_ao_values,
        fixed_ao_matrices.spins,
        hamiltonians_ry,
        potentials,
        frequency_grid.omega_ha,
        frequency_grid.weights_ha,
        grid_data.volume_element);
    siab::Provenance provenance
        = make_siab_production_provenance(ucell, channels, frequency_grid, pca_threshold);
    provenance.response_orbital_sha256
        = siab::sha256_file_manifest(resolved_sternheimer_response_orbital_files(ucell));
    const std::string path
        = join_output_path(output_dir, "sternheimer_galerkin_response.dat");
    siab::write_response_galerkin_v1(path, data, provenance);
    GlobalV::ofs_running << " Sternheimer response-orbital Galerkin v1 output: "
                         << path << std::endl;
    return path;
}

std::vector<std::vector<double>> transform_channel_potentials(
    SternheimerABFBuildData& abfs_data,
    const SternheimerFDHamiltonian::Grid& grid,
    const SternheimerCoulombWhitening& whitening)
{
    if (abfs_data.channels.size() != static_cast<std::size_t>(whitening.raw_dimension)
        || abfs_data.channels.empty())
    {
        throw std::invalid_argument("Sternheimer Coulomb transform does not match the raw ABFS channel count.");
    }
    return sample_sternheimer_abf_grid_channel_transform(abfs_data.radials_by_type,
                                                          abfs_data.atom_types,
                                                          abfs_data.atom_positions,
                                                          grid,
                                                          abfs_data.channels,
                                                          whitening.transform,
                                                          whitening.retained_rank);
}

std::vector<double> scale_potential(const std::vector<double>& potential, const double factor)
{
    std::vector<double> scaled = potential;
    for (double& value: scaled)
    {
        value *= factor;
    }
    return scaled;
}

std::vector<std::vector<double>> scale_potentials(const std::vector<std::vector<double>>& potentials,
                                                  const double factor)
{
    std::vector<std::vector<double>> scaled = potentials;
    for (std::vector<double>& potential: scaled)
    {
        for (double& value: potential)
        {
            value *= factor;
        }
    }
    return scaled;
}

void scale_potentials_in_place(std::vector<std::vector<double>>& potentials, const double factor)
{
    for (std::vector<double>& potential: potentials)
    {
        for (double& value: potential)
        {
            value *= factor;
        }
    }
}

void write_abfs_channel_diagnostic(const std::string& filename, const std::vector<SternheimerABFGridChannel>& channels)
{
    std::ofstream out(filename.c_str(), std::ios::out | std::ios::trunc);
    if (!out)
    {
        throw std::runtime_error("Failed to open Sternheimer ABFS channel diagnostic file: " + filename);
    }
    out << std::setprecision(16);
    out << "# ABACUS Sternheimer ABFS channel diagnostic\n";
    out << "# channel atom atom_local type l radial m label max_abs\n";
    for (const SternheimerABFGridChannel& channel: channels)
    {
        out << channel.channel_index << ' ' << channel.atom_index << ' ' << channel.atom_local_index << ' '
            << channel.type_index << ' ' << channel.angular_momentum << ' ' << channel.radial_index << ' '
            << channel.magnetic_index << ' ' << channel.label << ' ' << channel.max_abs << '\n';
    }
}

void write_coulomb_whitening_diagnostic(const std::string& filename,
                                         const SternheimerCoulombWhitening& whitening)
{
    std::ofstream out(filename.c_str(), std::ios::out | std::ios::trunc);
    if (!out)
    {
        throw std::runtime_error("Failed to open Sternheimer Coulomb whitening diagnostic file: " + filename);
    }
    out << std::setprecision(16);
    out << "# ABACUS Sternheimer global full-Coulomb whitening\n";
    out << "raw_auxiliary_dimension " << whitening.raw_dimension << '\n';
    out << "whitened_auxiliary_rank " << whitening.retained_rank << '\n';
    out << "discarded_auxiliary_rank " << whitening.discarded_rank << '\n';
    out << "relative_threshold " << whitening.relative_threshold << '\n';
    out << "largest_eigenvalue " << whitening.largest_eigenvalue << '\n';
    out << "smallest_retained_eigenvalue " << whitening.smallest_retained_eigenvalue << '\n';
    out << "max_orthonormality_error " << whitening.max_orthonormality_error << '\n';
    out << "transform_sha256 " << hash_coulomb_whitening_transform(whitening) << '\n';
    out << "# index eigenvalue retained\n";
    const int first_retained = whitening.discarded_rank;
    for (int index = 0; index != whitening.raw_dimension; ++index)
    {
        out << index << ' ' << whitening.eigenvalues[static_cast<std::size_t>(index)] << ' '
            << (index >= first_retained ? "yes" : "no") << '\n';
    }
}

SternheimerRPA::Chi0V1Metadata make_chi0_v1_metadata(const UnitCell& ucell,
                                                     const std::vector<SternheimerABFGridChannel>& channels,
                                                     const int ifrequency,
                                                     const double omega_ha,
                                                     const double weight_ha)
{
    SternheimerRPA::Chi0V1Metadata metadata;
    metadata.iq = 1;
    metadata.ifrequency = ifrequency;
    metadata.omega = omega_ha;
    metadata.weight = weight_ha;
    metadata.atom_naux.assign(static_cast<std::size_t>(ucell.nat), 0);
    for (const SternheimerABFGridChannel& channel: channels)
    {
        if (channel.atom_index < 0 || channel.atom_index >= ucell.nat)
        {
            throw std::runtime_error("Sternheimer chi0 found an ABFS channel with invalid atom index.");
        }
        if (channel.atom_local_index < 0)
        {
            throw std::runtime_error("Sternheimer chi0 found an ABFS channel with invalid local index.");
        }
        int& atom_naux = metadata.atom_naux[static_cast<std::size_t>(channel.atom_index)];
        atom_naux = std::max(atom_naux, channel.atom_local_index + 1);
    }
    return metadata;
}

std::vector<SternheimerRPA::AuxiliaryChannel> make_chi0_auxiliary_channels(
    const std::vector<SternheimerABFGridChannel>& channels)
{
    std::vector<SternheimerRPA::AuxiliaryChannel> auxiliary_channels;
    auxiliary_channels.reserve(channels.size());
    for (const SternheimerABFGridChannel& channel: channels)
    {
        SternheimerRPA::AuxiliaryChannel auxiliary_channel;
        auxiliary_channel.channel_index = channel.channel_index;
        auxiliary_channel.atom_index = channel.atom_index;
        auxiliary_channel.atom_local_index = channel.atom_local_index;
        auxiliary_channels.push_back(auxiliary_channel);
    }
    return auxiliary_channels;
}

void write_chi0_index_file(const std::string& filename,
                           const std::vector<std::pair<std::string, SternheimerRPA::Chi0V1Metadata>>& entries)
{
    std::ofstream out(filename.c_str(), std::ios::out | std::ios::trunc);
    if (!out)
    {
        throw std::runtime_error("Failed to open Sternheimer chi0 index file: " + filename);
    }
    out << std::setprecision(16);
    out << "# ABACUS Sternheimer chi0 v1 index\n";
    out << "format_version 1\n";
    out << "marker " << SternheimerRPA::chi0_v1_marker() << '\n';
    out << "nq 1\n";
    out << "nfreq " << entries.size() << '\n';
    out << "iq ifrequency omega_Ha weight_Ha filename\n";
    for (const auto& entry: entries)
    {
        const SternheimerRPA::Chi0V1Metadata& metadata = entry.second;
        out << metadata.iq << ' ' << metadata.ifrequency << ' ' << metadata.omega << ' ' << metadata.weight << ' '
            << entry.first << '\n';
    }
}

SternheimerFDZeroOrderStates solve_fd_zero_order_auto(const SternheimerFDHamiltonian& hamiltonian,
                                                      const int num_bands,
                                                      const double volume_element,
                                                      const int max_dense_size,
                                                      const int lanczos_max_subspace_size)
{
    if (hamiltonian.grid().size() <= max_dense_size)
    {
        return solve_sternheimer_fd_zero_order_dense(hamiltonian, num_bands, volume_element, max_dense_size);
    }

    SternheimerFDLanczosOptions options;
    options.max_subspace_size = lanczos_max_subspace_size;
    options.residual_tolerance = 1.0e-8;
    options.initial_seed = 1;
    return solve_sternheimer_fd_zero_order_lanczos(hamiltonian, num_bands, volume_element, options);
}

void broadcast_zero_order_states(SternheimerFDZeroOrderStates& states, const int grid_size, const bool enabled)
{
#ifdef __MPI
    if (!enabled || GlobalV::NPROC <= 1)
    {
        return;
    }

    int num_states = GlobalV::MY_RANK == 0 ? static_cast<int>(states.wavefunctions.size()) : 0;
    MPI_Bcast(&num_states, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (num_states < 0)
    {
        throw std::runtime_error("Invalid Sternheimer zero-order state count broadcast.");
    }

    if (GlobalV::MY_RANK != 0)
    {
        states.eigenvalues.assign(static_cast<std::size_t>(num_states), 0.0);
        states.residual_norms.assign(static_cast<std::size_t>(num_states), 0.0);
        states.wavefunctions.assign(static_cast<std::size_t>(num_states),
                                    SternheimerFDHamiltonian::Vector(static_cast<std::size_t>(grid_size),
                                                                     SternheimerFDHamiltonian::Complex(0.0, 0.0)));
    }

    if (num_states > 0)
    {
        MPI_Bcast(states.eigenvalues.data(), num_states, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(states.residual_norms.data(), num_states, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    for (int istate = 0; istate != num_states; ++istate)
    {
        if (static_cast<int>(states.wavefunctions[static_cast<std::size_t>(istate)].size()) != grid_size)
        {
            throw std::runtime_error("Sternheimer zero-order wavefunction size does not match the full grid.");
        }
        MPI_Bcast(reinterpret_cast<double*>(states.wavefunctions[static_cast<std::size_t>(istate)].data()),
                  2 * grid_size,
                  MPI_DOUBLE,
                  0,
                  MPI_COMM_WORLD);
    }
#else
    (void)states;
    (void)grid_size;
    (void)enabled;
#endif
}

void reduce_chi0_output_stats(bool& all_converged,
                              int& solved_equations,
                              double& max_solver_relative_residual,
                              double& max_equation_residual_norm,
                              const bool enabled)
{
#ifdef __MPI
    if (!enabled || GlobalV::NPROC <= 1)
    {
        return;
    }

    int converged_flag = all_converged ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &converged_flag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    all_converged = converged_flag != 0;

    MPI_Allreduce(MPI_IN_PLACE, &solved_equations, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    double residuals[2] = {max_solver_relative_residual, max_equation_residual_norm};
    MPI_Allreduce(MPI_IN_PLACE, residuals, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    max_solver_relative_residual = residuals[0];
    max_equation_residual_norm = residuals[1];
#else
    (void)all_converged;
    (void)solved_equations;
    (void)max_solver_relative_residual;
    (void)max_equation_residual_norm;
    (void)enabled;
#endif
}

std::vector<SternheimerFDHamiltonian::Vector> occupied_wavefunctions_from_states(
    const SternheimerFDZeroOrderStates& states,
    const elecstate::ElecState& elec_state,
    const int k_index)
{
    std::vector<SternheimerFDHamiltonian::Vector> occupied;
    for (int ib = 0; ib != static_cast<int>(states.wavefunctions.size()); ++ib)
    {
        if (elec_state.wg(k_index, ib) > 1.0e-8)
        {
            occupied.push_back(states.wavefunctions[ib]);
        }
    }
    return occupied;
}

void write_failure_report(std::ofstream& out, const std::string& reason)
{
    out << "# ABACUS Sternheimer FD linear-response smoke test\n";
    out << "status failed\n";
    out << "reason " << reason << '\n';
}

} // namespace

bool sternheimer_abacus_st_smoke_enabled()
{
    return env_is_true(kSmokeEnv);
}

void run_sternheimer_abacus_st_smoke(const elecstate::Potential& potential,
                                     const ModulePW::PW_Basis& pw_basis,
                                     const UnitCell& ucell,
                                     const elecstate::ElecState& elec_state,
                                     const std::string& output_dir)
{
    if (!sternheimer_abacus_st_smoke_enabled() || GlobalV::MY_RANK != 0)
    {
        return;
    }

    const std::string report_path = default_report_path(output_dir);
    std::ofstream out(report_path.c_str(), std::ios::out | std::ios::trunc);
    if (!out)
    {
        GlobalV::ofs_running << " Sternheimer FD ST smoke: failed to open " << report_path << std::endl;
        return;
    }

    try
    {
        if (GlobalV::NPROC != 1)
        {
            throw std::runtime_error(
                "The current FD ST smoke test requires a single MPI rank so pw_basis.nrxx is the full grid.");
        }
        if (elec_state.ekb.nc <= 0 || elec_state.wg.nc <= 0)
        {
            throw std::runtime_error("ABACUS DFT eigenvalues or occupations are not available.");
        }

        const int requested_bands = positive_int_from_env(kBandsEnv, 1);
        const int num_bands = std::min(requested_bands, elec_state.ekb.nc);
        const int max_channels = positive_int_from_env(kChannelsEnv, 1);
        const int max_dense_size = positive_int_from_env(kMaxDenseEnv, 4096);
        const int lanczos_max_subspace_size = positive_int_from_env(kLanczosSubspaceEnv, 320);
        const double omega = nonnegative_double_from_env(kOmegaEnv, 0.5);
        const double solver_tolerance
            = positive_double_from_env(kSolverToleranceEnv, default_sternheimer_solver_tolerance());
        const int solver_max_iter = positive_int_from_env(kSolverMaxIterEnv, 300);
        const double pca_threshold = nonnegative_double_from_env(kPCAThresholdEnv, PARAM.inp.exx_pca_threshold);
        const double ccp_rmesh_times = positive_double_from_env(kCCPRmeshTimesEnv, PARAM.inp.rpa_ccp_rmesh_times);

        SternheimerABACUSSTSmokeResult result;
        result.grid_data = make_sternheimer_fd_grid(pw_basis);
        result.omega = omega;
        result.pca_threshold = pca_threshold;
        result.ccp_rmesh_times = ccp_rmesh_times;
        result.perturbation_source = "abfs_ccp";

        const SternheimerFDHamiltonian hamiltonian
            = make_sternheimer_fd_hamiltonian(potential, pw_basis, ucell, 0, 1.0);
        const SternheimerFDZeroOrderStates states = solve_fd_zero_order_auto(hamiltonian,
                                                                             num_bands,
                                                                             result.grid_data.volume_element,
                                                                             max_dense_size,
                                                                             lanczos_max_subspace_size);
        const std::vector<SternheimerFDHamiltonian::Vector> occupied
            = occupied_wavefunctions_from_states(states, elec_state, 0);
        if (occupied.empty())
        {
            throw std::runtime_error("No occupied FD zero-order states are available for Sternheimer ST.");
        }

        const std::vector<SternheimerABFGridChannel> channels
            = build_abfs_ccp_grid_channels(ucell, result.grid_data.grid, max_channels, pca_threshold, ccp_rmesh_times);
        result.num_available_channels = static_cast<int>(channels.size());
        if (channels.empty())
        {
            throw std::runtime_error("No ABFS CCP perturbation channels were generated.");
        }

        auto dot = [&result](const SternheimerFDHamiltonian::Vector& lhs, const SternheimerFDHamiltonian::Vector& rhs) {
            return sternheimer_fd_grid_dot(lhs, rhs, result.grid_data.volume_element);
        };

        SternheimerRPA::SolverOptions solver_options;
        solver_options.max_iter = solver_max_iter;
        solver_options.residual_tol = solver_tolerance;

        for (int ib = 0; ib != static_cast<int>(states.wavefunctions.size()); ++ib)
        {
            const double occupation = elec_state.wg(0, ib);
            if (occupation <= 1.0e-8)
            {
                continue;
            }

            for (const SternheimerABFGridChannel& channel: channels)
            {
                const std::vector<double> perturbation_ry = scale_potential(channel.potential_r, kHartreeToRydberg);
                SternheimerFDHamiltonian::Vector rhs;
                SternheimerRPA::build_rhs_from_hartree_perturbation(perturbation_ry, states.wavefunctions[ib], rhs);
                SternheimerFDHamiltonian::Vector projected_rhs = rhs;
                SternheimerRPA::project_out_subspace(occupied, dot, projected_rhs);

                const SternheimerFDLinearResponse response
                    = solve_sternheimer_fd_linear_response(hamiltonian,
                                                           occupied,
                                                           states.eigenvalues[ib],
                                                           rhs,
                                                           omega,
                                                           result.grid_data.volume_element,
                                                           solver_options);

                SternheimerABACUSSTChannelResult channel_result;
                channel_result.band_index = ib;
                channel_result.channel_index = channel.channel_index;
                channel_result.atom_index = channel.atom_index;
                channel_result.angular_momentum = channel.angular_momentum;
                channel_result.radial_index = channel.radial_index;
                channel_result.magnetic_index = channel.magnetic_index;
                channel_result.fd_eigenvalue = states.eigenvalues[ib];
                channel_result.occupation = occupation;
                channel_result.rhs_norm = sternheimer_fd_grid_norm(rhs, result.grid_data.volume_element);
                channel_result.projected_rhs_norm
                    = sternheimer_fd_grid_norm(projected_rhs, result.grid_data.volume_element);
                channel_result.solver_converged = response.solver.converged;
                channel_result.solver_iterations = response.solver.iterations;
                channel_result.solver_relative_residual = response.solver.relative_residual;
                channel_result.equation_residual_norm = response.residual_norm;
                channel_result.polarizability
                    = SternheimerRPA::accumulate_polarizability_grid_element(channel.potential_r,
                                                                             states.wavefunctions[ib],
                                                                             response.delta_wavefunction,
                                                                             result.grid_data.volume_element);
                result.channels.push_back(channel_result);
            }
        }

        out << format_sternheimer_abacus_st_report(result);
        GlobalV::ofs_running << " Sternheimer FD ST smoke report: " << report_path << std::endl;
    }
    catch (const std::exception& error)
    {
        write_failure_report(out, error.what());
        GlobalV::ofs_running << " Sternheimer FD ST smoke failed: " << error.what() << std::endl;
        GlobalV::ofs_running << " Sternheimer FD ST smoke report: " << report_path << std::endl;
    }
}

void run_sternheimer_abacus_chi0_output_impl(const elecstate::Potential& potential,
                                             const ModulePW::PW_Basis& pw_basis,
                                             const UnitCell& ucell,
                                             const elecstate::ElecState& elec_state,
                                             const std::string& output_dir,
                                             const LCAO_Orbitals* lcao_orbitals,
                                             const std::vector<SternheimerLCAOOccupiedChannel>* lcao_occupied_channels,
                                             const SternheimerLCAOFixedAOMatrices* fixed_ao_matrices,
                                             const ModulePW::PW_Basis_K* siab_pw_wfc,
                                             const Structure_Factor* siab_structure_factor)
{
    const SternheimerOutputMode output_mode
        = select_sternheimer_output_mode(PARAM.inp.out_sternheimer_librpa,
                                         PARAM.inp.out_sternheimer_siab,
                                         PARAM.inp.out_sternheimer_galerkin,
                                         PARAM.inp.out_sternheimer_galerkin_primitive,
                                         PARAM.inp.out_sternheimer_galerkin_response);
    if (!output_mode.run)
    {
        return;
    }

    const bool write_librpa = output_mode.write_librpa;
    const bool write_siab = output_mode.write_siab_targets;
    const bool write_fixed_ao = output_mode.write_fixed_ao;
    const bool write_primitive = output_mode.write_primitive;
    const bool write_response_orbitals = output_mode.write_response_orbitals;
    const bool use_frequency_mpi = !output_mode.fixed_ao_only && PARAM.inp.sternheimer_frequency_mpi;
    const bool use_channel_mpi = !output_mode.fixed_ao_only && PARAM.inp.sternheimer_channel_mpi;
    const std::string mpi_layout = PARAM.inp.sternheimer_mpi_layout;
    const bool use_global_equation_mpi = mpi_layout == "global_equation";
    const bool use_distributed_mpi = use_frequency_mpi || use_channel_mpi;
    const int nfreq = PARAM.inp.sternheimer_nfreq;
    if (!use_distributed_mpi && GlobalV::MY_RANK != 0)
    {
        return;
    }

    const std::string status_path = output_mode.fixed_ao_only
                                        ? join_output_path(output_dir, "STERNHEIMER_GALERKIN.dat")
                                        : chi0_status_path(output_dir);
    const std::string output_label = output_mode.fixed_ao_only ? "Sternheimer Galerkin output"
                                                               : "Sternheimer chi0 output";
    std::ofstream out;
    if (GlobalV::MY_RANK == 0)
    {
        out.open(status_path.c_str(), std::ios::out | std::ios::trunc);
        if (!out)
        {
            GlobalV::ofs_running << ' ' << output_label << ": failed to open " << status_path << std::endl;
#ifdef __MPI
            if (use_distributed_mpi && GlobalV::NPROC > 1)
            {
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
#endif
            return;
        }
        out << std::setprecision(16);
        out << (output_mode.fixed_ao_only ? "# ABACUS fixed-AO Sternheimer Galerkin output\n"
                  : write_siab ? "# ABACUS Coulomb-whitened Sternheimer target output for SIAB\n"
                               : "# ABACUS Sternheimer chi0 output for LibRPA\n");
    }

    try
    {
        const auto chi0_start_time = std::chrono::steady_clock::now();
        reset_chi0_progress_file();
        append_chi0_progress_event("enter",
                                   0,
                                   -1,
                                   -1,
                                   -1,
                                   0,
                                   nullptr,
                                   -1.0,
                                   elapsed_seconds_since(chi0_start_time),
                                   std::string("frequency_mpi=") + (use_frequency_mpi ? "yes" : "no")
                                       + " channel_mpi=" + (use_channel_mpi ? "yes" : "no")
                                       + " mpi_layout=" + mpi_layout);
        if (write_librpa && PARAM.inp.out_librpa_reader_version != 1)
        {
            throw std::runtime_error("out_sternheimer_librpa requires out_librpa_reader_version=1.");
        }
        if (!output_mode.fixed_ao_only)
        {
            SternheimerRPA::validate_mpi_layout(mpi_layout,
                                                use_frequency_mpi,
                                                use_channel_mpi,
                                                write_siab,
                                                write_librpa,
                                                nfreq,
                                                GlobalV::NPROC);
        }
        if (!output_mode.fixed_ao_only && GlobalV::NPROC != 1 && !use_frequency_mpi)
        {
            throw std::runtime_error(
                "Sternheimer chi0 output with multiple MPI ranks requires sternheimer_frequency_mpi=true.");
        }
        if (use_frequency_mpi && GlobalV::NPROC > 1 && pw_basis.poolnproc != GlobalV::NPROC)
        {
            throw std::runtime_error(
                "sternheimer_frequency_mpi currently requires all MPI ranks to be in the same real-space pool.");
        }
        if (elec_state.ekb.nc <= 0 || elec_state.wg.nc <= 0)
        {
            throw std::runtime_error("ABACUS DFT eigenvalues or occupations are not available.");
        }

        const bool use_lcao_zero_order = lcao_occupied_channels != nullptr;
        if ((lcao_orbitals == nullptr) != (lcao_occupied_channels == nullptr))
        {
            throw std::runtime_error("Sternheimer LCAO zero-order input is incomplete.");
        }
        if (write_fixed_ao && (!use_lcao_zero_order || fixed_ao_matrices == nullptr))
        {
            throw std::runtime_error("Fixed-AO Sternheimer Galerkin output requires LCAO Delta-ST matrices.");
        }
        if (write_siab
            && (siab_pw_wfc == nullptr || siab_structure_factor == nullptr))
        {
            throw std::runtime_error(
                "Sternheimer SIAB target output requires the PW FFT basis and the structure factor.");
        }
        if (write_primitive && siab_structure_factor == nullptr)
        {
            throw std::runtime_error(
                "Sternheimer SIAB primitive output requires the structure factor.");
        }
        if (write_primitive && GlobalV::NPROC != 1)
        {
            throw std::runtime_error(
                "out_sternheimer_galerkin_primitive currently requires one MPI rank; use OpenMP within one node.");
        }
        if (write_response_orbitals && GlobalV::NPROC != 1)
        {
            throw std::runtime_error(
                "out_sternheimer_galerkin_response currently requires one MPI rank; use OpenMP within one node.");
        }

        std::vector<int> response_spin_indices;
        if (use_lcao_zero_order)
        {
            const int physical_spin_channel_count
                = sternheimer_lcao_physical_spin_channel_count(PARAM.inp.nspin);
            if (elec_state.ekb.nr != physical_spin_channel_count
                || elec_state.wg.nr != physical_spin_channel_count)
            {
                throw std::runtime_error(
                    "Sternheimer LCAO response requires one Gamma-point eigenvalue/occupation row per physical spin "
                    "channel.");
            }
            validate_sternheimer_lcao_occupied_channels(*lcao_occupied_channels,
                                                        physical_spin_channel_count,
                                                        PARAM.globalv.nlocal);
            if (write_fixed_ao)
            {
                validate_sternheimer_lcao_fixed_ao_matrices(*fixed_ao_matrices,
                                                            physical_spin_channel_count,
                                                            PARAM.globalv.nlocal);
            }
            if (lcao_occupied_channels->empty())
            {
                throw std::runtime_error("Sternheimer LCAO output found no occupied spin channels.");
            }
            response_spin_indices = sternheimer_lcao_spin_indices(*lcao_occupied_channels);
        }
        else
        {
            response_spin_indices.push_back(0);
        }

        std::vector<int> occupied_band_counts;
        occupied_band_counts.reserve(response_spin_indices.size());
        for (const int spin_index: response_spin_indices)
        {
            const int occupied_count = occupied_band_count(elec_state, spin_index);
            if (occupied_count <= 0)
            {
                throw std::runtime_error("No occupied DFT bands are available for a Sternheimer spin channel.");
            }
            occupied_band_counts.push_back(occupied_count);
        }
        const int requested_bands = positive_int_from_env(kBandsEnv, occupied_band_counts.front());
        const int fd_num_bands = std::min(requested_bands, elec_state.ekb.nc);
        if (!use_lcao_zero_order && fd_num_bands < occupied_band_counts.front())
        {
            throw std::runtime_error("Sternheimer chi0 output requires all occupied bands; use the smoke test for "
                                     "band-limited debugging.");
        }

        const int max_dense_size = positive_int_from_env(kMaxDenseEnv, 4096);
        const int lanczos_max_subspace_size = positive_int_from_env(kLanczosSubspaceEnv, 320);
        const double solver_tolerance
            = positive_double_from_env(kSolverToleranceEnv, default_sternheimer_solver_tolerance());
        const int solver_max_iter = positive_int_from_env(kSolverMaxIterEnv, 300);
        const double pca_threshold = nonnegative_double_from_env(kPCAThresholdEnv, PARAM.inp.exx_pca_threshold);
        const double ccp_rmesh_times = positive_double_from_env(kCCPRmeshTimesEnv, PARAM.inp.rpa_ccp_rmesh_times);
        const int channel_worker_user_cap = int_from_env(kChannelMaxWorkersEnv, 0);
        if (channel_worker_user_cap < 0)
        {
            throw std::invalid_argument(std::string("Invalid non-negative integer in ") + kChannelMaxWorkersEnv + ".");
        }
        const int default_frequency_rank_shift = use_frequency_mpi && GlobalV::NPROC > 1 ? 1 : 0;
        const int frequency_rank_shift = int_from_env(kFrequencyRankShiftEnv, default_frequency_rank_shift);
        const auto frequency_assignment = [&](const int ifrequency) {
            if (use_global_equation_mpi)
            {
                SternheimerRPA::FrequencyMPIAssignment assignment;
                assignment.owns_frequency = true;
                assignment.frequency_leader_rank = 0;
                assignment.frequency_group_size = GlobalV::NPROC;
                assignment.frequency_group_local_rank = GlobalV::MY_RANK;
                return assignment;
            }
            if (!use_frequency_mpi)
            {
                return SternheimerRPA::frequency_mpi_assignment(ifrequency, nfreq, 1, 0, 0, false);
            }
            return SternheimerRPA::frequency_mpi_assignment(ifrequency,
                                                             nfreq,
                                                             GlobalV::NPROC,
                                                             GlobalV::MY_RANK,
                                                             frequency_rank_shift,
                                                             use_channel_mpi);
        };
        const int frequency_group_size = frequency_assignment(0).frequency_group_size;
#ifdef __MPI
        MPI_Comm chi0_frequency_group_communicator = MPI_COMM_NULL;
        bool owns_chi0_frequency_group_communicator = false;
        if (write_librpa && use_channel_mpi && frequency_group_size > 1)
        {
            if (use_global_equation_mpi)
            {
                chi0_frequency_group_communicator = MPI_COMM_WORLD;
            }
            else
            {
                const int frequency_group_color = GlobalV::MY_RANK / frequency_group_size;
                if (MPI_Comm_split(MPI_COMM_WORLD,
                                   frequency_group_color,
                                   GlobalV::MY_RANK,
                                   &chi0_frequency_group_communicator)
                    != MPI_SUCCESS)
                {
                    throw std::runtime_error("Failed to create the Sternheimer chi0 frequency-group communicator.");
                }
                owns_chi0_frequency_group_communicator = true;
            }
        }
#endif
        const std::string frequency_grid_file = PARAM.inp.sternheimer_frequency_grid_file;
        const bool use_frequency_grid_file = !frequency_grid_file.empty();
        const bool use_delta_sternheimer = PARAM.inp.sternheimer_delta;
        const char* lcao_virtual_source_raw = std::getenv(kLCAOVirtualSourceEnv);
        const SternheimerLCAOVirtualSource lcao_virtual_source = parse_sternheimer_lcao_virtual_source(
            lcao_virtual_source_raw == nullptr ? PARAM.inp.sternheimer_delta_virtual_source
                                               : lcao_virtual_source_raw);
        if (use_lcao_zero_order)
        {
            if (!use_delta_sternheimer)
            {
                throw std::runtime_error(
                    "Sternheimer LCAO zero-order input currently requires sternheimer_delta=true.");
            }
            if (PARAM.inp.nspin != 1 && PARAM.inp.nspin != 2)
            {
                throw std::runtime_error("Sternheimer LCAO zero-order input currently supports Gamma-point nspin=1 or "
                                         "nspin=2 calculations.");
            }
            for (std::size_t ispin = 0; ispin != lcao_occupied_channels->size(); ++ispin)
            {
                if ((*lcao_occupied_channels)[ispin].coefficients.size()
                    != static_cast<std::size_t>(occupied_band_counts[ispin]))
                {
                    throw std::runtime_error("Sternheimer LCAO occupied coefficient count does not match occupations.");
                }
                if (lcao_virtual_source == SternheimerLCAOVirtualSource::KSBands)
                {
                    const auto& channel = (*lcao_occupied_channels)[ispin];
                    const std::size_t available_bands
                        = channel.coefficients.size() + channel.unoccupied_coefficients.size();
                    if (available_bands != static_cast<std::size_t>(PARAM.globalv.nlocal))
                    {
                        throw std::runtime_error(
                            "Sternheimer ks_bands virtual source requires nbands=nlocal so the LCAO virtual space is "
                            "complete.");
                    }
                    if (channel.unoccupied_coefficients.empty())
                    {
                        throw std::runtime_error("Sternheimer ks_bands virtual source found no unoccupied KS bands.");
                    }
                }
            }
        }

        std::vector<SternheimerRPA::TransitionEnergyWindow> spin_transition_windows;
        spin_transition_windows.reserve(response_spin_indices.size());
        bool transition_window_available = true;
        for (const int spin_index: response_spin_indices)
        {
            const std::vector<double> eigenvalues_ry = eigenvalues_ry_from_elec_state(elec_state, spin_index);
            const std::vector<double> occupations = occupations_from_elec_state(elec_state, spin_index);
            SternheimerRPA::TransitionEnergyWindow spin_window;
            if (SternheimerRPA::try_transition_energy_window_from_eigenvalues_ry(eigenvalues_ry,
                                                                                 occupations,
                                                                                 spin_window))
            {
                spin_transition_windows.push_back(spin_window);
            }
            else
            {
                transition_window_available = false;
            }
        }
        if (!use_frequency_grid_file && !transition_window_available)
        {
            throw std::runtime_error("Sternheimer GreenX minimax grid requires a positive occupied-to-empty "
                                     "transition in every response spin channel.");
        }
        SternheimerRPA::TransitionEnergyWindow transition_window;
        if (transition_window_available)
        {
            transition_window = SternheimerRPA::merge_transition_energy_windows(spin_transition_windows);
        }
        const std::string frequency_grid_source = use_frequency_grid_file ? "file" : "greenx_minimax";
        const SternheimerRPA::FrequencyGrid frequency_grid
            = use_frequency_grid_file
                  ? SternheimerRPA::read_frequency_grid_file(frequency_grid_file, nfreq)
                  : SternheimerRPA::generate_greenx_minimax_frequency_grid(nfreq,
                                                                           transition_window.emin_ha,
                                                                           transition_window.emax_ha);
        append_chi0_progress_event("frequency_grid",
                                   0,
                                   -1,
                                   -1,
                                   -1,
                                   0,
                                   nullptr,
                                   -1.0,
                                   elapsed_seconds_since(chi0_start_time),
                                   "rank_shift=" + std::to_string(frequency_rank_shift));

        const bool use_full_grid = use_frequency_mpi || write_fixed_ao;
        const SternheimerABACUSFDGridData grid_data
            = use_full_grid ? make_sternheimer_fd_full_grid(pw_basis) : make_sternheimer_fd_grid(pw_basis);
        SternheimerABFBuildData abfs_data
            = build_abfs_ccp_data(ucell, grid_data.grid, -1, pca_threshold, ccp_rmesh_times, write_siab);
        std::vector<SternheimerABFGridChannel>& channels = abfs_data.channels;
        const std::string abfs_source
            = sternheimer_abfs_perturbation_source(GlobalC::exx_info.info_ri.files_abfs);
        if (channels.empty())
        {
            throw std::runtime_error("No ABFS CCP perturbation channels were generated.");
        }

        const int raw_num_channels = static_cast<int>(channels.size());
        if (write_siab && GlobalC::exx_info.info_ri.files_abfs.empty())
        {
            throw std::runtime_error(
                "out_sternheimer_siab requires explicit ABFS_ORBITAL files for immutable target provenance.");
        }
        const std::string auxiliary_basis_sha256
            = write_siab && GlobalV::MY_RANK == 0
                  ? siab::sha256_unique_file_manifest(GlobalC::exx_info.info_ri.files_abfs)
                  : std::string();
        SternheimerCoulombWhitening coulomb_whitening;
        if (write_siab)
        {
            coulomb_whitening = make_sternheimer_coulomb_whitening(
                abfs_data.full_coulomb_metric,
                raw_num_channels,
                PARAM.inp.sternheimer_siab_coulomb_threshold);
        }
        const int num_channels = write_siab ? coulomb_whitening.retained_rank : raw_num_channels;
        append_chi0_progress_event("channels_ready",
                                   0,
                                   -1,
                                   -1,
                                   -1,
                                   0,
                                   nullptr,
                                   -1.0,
                                   elapsed_seconds_since(chi0_start_time),
                                   "raw_channels=" + std::to_string(raw_num_channels)
                                       + " response_channels=" + std::to_string(num_channels)
                                       + " user_cap=" + std::to_string(channel_worker_user_cap));

        SIABPrimitiveExportData siab_primitives;
        SternheimerSIABMemoryEstimate siab_memory_estimate;
        std::uint64_t siab_slurm_memory_bytes = 0;
        if (write_siab || write_primitive)
        {
            siab_primitives = build_siab_primitive_export_data(pw_basis,
                                                               *siab_structure_factor,
                                                               ucell,
                                                               write_primitive);
            append_chi0_progress_event("siab_primitives_ready",
                                       0,
                                       -1,
                                       -1,
                                       -1,
                                       0,
                                       nullptr,
                                       -1.0,
                                       elapsed_seconds_since(chi0_start_time),
                                       "nprimitive=" + std::to_string(siab_primitives.primitive_count));
        }
        if (write_siab)
        {
            int occupied_total = 0;
            for (const int count: occupied_band_counts)
            {
                occupied_total += count;
            }
            siab_memory_estimate
                = estimate_sternheimer_siab_dense_memory(grid_data.grid.size(),
                                                         raw_num_channels,
                                                         num_channels,
                                                         siab_primitives.primitive_count,
                                                         siab_primitives.reciprocal_count,
                                                         occupied_total,
                                                         nfreq);
            siab_slurm_memory_bytes = slurm_memory_limit_bytes();
            constexpr std::uint64_t solver_reserve_bytes = 16ULL * 1024ULL * 1024ULL * 1024ULL;
            if (siab_slurm_memory_bytes > 0
                && (siab_memory_estimate.total_bytes > siab_slurm_memory_bytes
                    || solver_reserve_bytes > siab_slurm_memory_bytes - siab_memory_estimate.total_bytes))
            {
                throw std::runtime_error(
                    "Sternheimer SIAB estimated dense allocation plus the 16-GiB solver reserve exceeds "
                    "SLURM_MEM_PER_NODE.");
            }
            if (GlobalV::MY_RANK == 0)
            {
                std::ofstream memory("STERNHEIMER_SIAB_MEMORY.dat");
                if (!memory)
                {
                    throw std::runtime_error("Failed to open Sternheimer SIAB memory diagnostic file.");
                }
                memory << "# ABACUS Sternheimer SIAB conservative dense-memory estimate\n";
                memory << "representation streamed_raw_hartree_and_serial_reciprocal_primitives_v1\n";
                memory << "coulomb_metric_bytes " << siab_memory_estimate.coulomb_metric_bytes << '\n';
                memory << "transformed_potential_bytes " << siab_memory_estimate.transformed_potential_bytes << '\n';
                memory << "channel_transform_workspace_bytes "
                       << siab_memory_estimate.channel_transform_workspace_bytes << '\n';
                memory << "reciprocal_primitive_bytes " << siab_memory_estimate.reciprocal_primitive_bytes << '\n';
                memory << "primitive_overlap_bytes " << siab_memory_estimate.primitive_overlap_bytes << '\n';
                memory << "gathered_reference_row_bytes " << siab_memory_estimate.gathered_reference_row_bytes << '\n';
                memory << "estimated_dense_total_bytes " << siab_memory_estimate.total_bytes << '\n';
                memory << "solver_reserve_bytes " << solver_reserve_bytes << '\n';
                memory << "slurm_memory_per_node_bytes " << siab_slurm_memory_bytes << '\n';
            }
            append_chi0_progress_event("siab_memory_ready",
                                       0,
                                       -1,
                                       -1,
                                       -1,
                                       0,
                                       nullptr,
                                       -1.0,
                                       elapsed_seconds_since(chi0_start_time),
                                       "dense_bytes=" + std::to_string(siab_memory_estimate.total_bytes));
        }
        if (GlobalV::MY_RANK == 0)
        {
            write_abfs_channel_diagnostic("STERNHEIMER_ABFS_CHANNELS.dat", channels);
            if (write_siab)
            {
                write_coulomb_whitening_diagnostic("STERNHEIMER_SIAB_COULOMB_WHITENING.dat",
                                                    coulomb_whitening);
            }
            if (sternheimer_abfs_diag_only_enabled())
            {
                out << "status abfs_diag_only\n";
                out << "grid " << grid_data.grid.nx << ' ' << grid_data.grid.ny << ' ' << grid_data.grid.nz << " size "
                    << grid_data.grid.size() << " dV " << grid_data.volume_element << '\n';
                out << "frequency_grid_source " << frequency_grid_source << '\n';
                out << "nfreq " << nfreq << '\n';
                out << "abfs_channels " << raw_num_channels << '\n';
                out << "response_channels " << num_channels << '\n';
                out << "abfs_source " << abfs_source << '\n';
            }
        }
        if (sternheimer_abfs_diag_only_enabled())
        {
            return;
        }

        std::vector<std::vector<double>> potentials;
        std::vector<std::vector<double>> perturbations_ry;
        if (!write_siab)
        {
            potentials = take_sternheimer_channel_potentials(channels);
            if (!output_mode.fixed_ao_only)
            {
                // Raw ABFS Coulomb potentials are retained in Ha for M=V chi0 V output.
                perturbations_ry = scale_potentials(potentials, kHartreeToRydberg);
            }
        }
        else
        {
            perturbations_ry = transform_channel_potentials(abfs_data, grid_data.grid, coulomb_whitening);
            scale_potentials_in_place(perturbations_ry, kHartreeToRydberg);
            if (GlobalV::MY_RANK == 0)
            {
                write_abfs_channel_diagnostic("STERNHEIMER_ABFS_CHANNELS.dat", channels);
            }
            for (SternheimerABFGridChannel& channel: channels)
            {
                channel.potential_r.clear();
                channel.potential_r.shrink_to_fit();
            }
        }

        std::vector<SternheimerDeltaGridFunction> sampled_ao_functions;
        if (use_lcao_zero_order)
        {
            sampled_ao_functions = build_lcao_candidate_grid_functions(ucell, grid_data.grid, lcao_orbitals);
            if (sampled_ao_functions.empty())
            {
                throw std::runtime_error("Sternheimer LCAO zero-order input found no sampled AO functions.");
            }
        }

        SternheimerResponseOrbitalLayout response_layout;
        std::vector<SternheimerDeltaGridFunction> response_orbital_functions;
        if (write_response_orbitals)
        {
            LCAO_Orbitals response_orbitals;
            read_sternheimer_response_orbitals(ucell, response_orbitals);
            response_layout = build_sternheimer_response_orbital_layout(
                make_response_orbital_atom_specs(ucell, response_orbitals));
            response_orbital_functions = reorder_response_orbital_grid_functions(
                build_lcao_candidate_grid_functions(ucell, grid_data.grid, &response_orbitals),
                response_layout);
        }

        std::string fixed_ao_path;
        if (write_fixed_ao && GlobalV::MY_RANK == 0)
        {
            fixed_ao_path = write_sternheimer_fixed_ao_sidecar(output_dir,
                                                               ucell,
                                                               channels,
                                                               frequency_grid,
                                                               pca_threshold,
                                                               grid_data,
                                                               sampled_ao_functions,
                                                               potentials,
                                                               *fixed_ao_matrices);
        }
        std::vector<SternheimerFDHamiltonian> hamiltonians_ry;
        if ((write_primitive || write_response_orbitals) && GlobalV::MY_RANK == 0)
        {
            hamiltonians_ry.reserve(fixed_ao_matrices->spins.size());
            for (const siab::FixedAOSpinInput& spin: fixed_ao_matrices->spins)
            {
                hamiltonians_ry.push_back(make_sternheimer_fd_hamiltonian(
                    potential, pw_basis, ucell, spin.spin_index, 1.0));
            }
        }
        std::string primitive_path;
        if (write_primitive && GlobalV::MY_RANK == 0)
        {
            primitive_path = write_sternheimer_primitive_galerkin_sidecar(
                output_dir,
                ucell,
                channels,
                frequency_grid,
                pca_threshold,
                grid_data,
                siab_primitives,
                sampled_ao_functions,
                potentials,
                *fixed_ao_matrices,
                hamiltonians_ry);
        }
        std::string response_path;
        if (write_response_orbitals && GlobalV::MY_RANK == 0)
        {
            response_path = write_sternheimer_response_galerkin_sidecar(
                output_dir,
                ucell,
                channels,
                frequency_grid,
                pca_threshold,
                grid_data,
                response_layout.blocks,
                response_orbital_functions,
                sampled_ao_functions,
                potentials,
                *fixed_ao_matrices,
                hamiltonians_ry);
        }
        if (output_mode.fixed_ao_only)
        {
            if (GlobalV::MY_RANK == 0)
            {
                out << "status success\n";
                out << "format v1\n";
                out << "fixed_ao_file " << fixed_ao_path << '\n';
                if (write_primitive)
                {
                    out << "primitive_file " << primitive_path << '\n';
                }
                if (write_response_orbitals)
                {
                    out << "response_file " << response_path << '\n';
                }
                out << "grid " << grid_data.grid.nx << ' ' << grid_data.grid.ny << ' ' << grid_data.grid.nz
                    << " size " << grid_data.grid.size() << " dV " << grid_data.volume_element << '\n';
                out << "nfreq " << nfreq << '\n';
                out << "frequency_grid_source " << frequency_grid_source << '\n';
                if (use_frequency_grid_file)
                {
                    out << "frequency_grid_file " << frequency_grid_file << '\n';
                }
                out << "pca_threshold " << pca_threshold << '\n';
                out << "ccp_rmesh_times " << ccp_rmesh_times << '\n';
                out << "abfs_channels " << num_channels << '\n';
                out << "linear_solve skipped\n";
                out << "solved_equations 0\n";
                GlobalV::ofs_running << " Sternheimer Galerkin status: " << status_path << std::endl;
            }
            return;
        }

        SternheimerRPA::SolverOptions solver_options;
        solver_options.max_iter = solver_max_iter;
        solver_options.residual_tol = solver_tolerance;

        bool all_converged = true;
        int solved_equations = 0;
        std::int64_t local_iteration_sum = 0;
        double max_solver_relative_residual = 0.0;
        double max_equation_residual_norm = 0.0;
        const std::vector<SternheimerRPA::AuxiliaryChannel> auxiliary_channels
            = write_librpa ? make_chi0_auxiliary_channels(channels)
                           : std::vector<SternheimerRPA::AuxiliaryChannel>();

        const std::size_t response_matrix_size
            = write_librpa ? static_cast<std::size_t>(num_channels) * static_cast<std::size_t>(num_channels) : 0;
        std::vector<std::vector<SternheimerRPA::Complex>> chi0_branches(static_cast<std::size_t>(nfreq));
        std::vector<std::chrono::steady_clock::time_point> frequency_start_times(static_cast<std::size_t>(nfreq));
        for (int ifrequency = 0; ifrequency != nfreq; ++ifrequency)
        {
            const SternheimerRPA::FrequencyMPIAssignment assignment = frequency_assignment(ifrequency);
            const int owner_rank = assignment.frequency_leader_rank;
            if (!assignment.owns_frequency)
            {
                continue;
            }
            if (write_librpa)
            {
                chi0_branches[static_cast<std::size_t>(ifrequency)].assign(response_matrix_size,
                                                                           SternheimerRPA::Complex(0.0, 0.0));
            }
            frequency_start_times[static_cast<std::size_t>(ifrequency)] = std::chrono::steady_clock::now();
            const double omega_ha = frequency_grid.omega_ha[static_cast<std::size_t>(ifrequency)];
            append_chi0_progress_event("frequency_start",
                                       ifrequency + 1,
                                       owner_rank,
                                       -1,
                                       -1,
                                       solved_equations,
                                       nullptr,
                                       -1.0,
                                       elapsed_seconds_since(chi0_start_time),
                                       "omega_Ha=" + std::to_string(omega_ha)
                                           + " group_size=" + std::to_string(assignment.frequency_group_size)
                                           + " group_local_rank="
                                           + std::to_string(assignment.frequency_group_local_rank));
        }

        struct SpinResponseDiagnostics
        {
            int spin_index = -1;
            int occupied_bands = 0;
            int occupied_projector_dimension = 0;
            int delta_virtual_states = 0;
            int delta_accepted_candidates = 0;
            int delta_discarded_candidates = 0;
        };
        std::vector<SpinResponseDiagnostics> spin_diagnostics;
        spin_diagnostics.reserve(response_spin_indices.size());
        std::vector<siab::ReferenceRow> local_siab_rows;
        int occupied_state_offset = 0;

        for (std::size_t ispin = 0; ispin != response_spin_indices.size(); ++ispin)
        {
            const int spin_index = response_spin_indices[ispin];
            const int occupied_count = occupied_band_counts[ispin];
            const int num_bands = use_lcao_zero_order ? occupied_count : fd_num_bands;
            const SternheimerLCAOOccupiedChannel* lcao_channel
                = use_lcao_zero_order ? &(*lcao_occupied_channels)[ispin] : nullptr;

            const SternheimerFDHamiltonian hamiltonian
                = use_frequency_mpi ? make_sternheimer_fd_full_hamiltonian(potential, pw_basis, ucell, spin_index, 1.0)
                                    : make_sternheimer_fd_hamiltonian(potential, pw_basis, ucell, spin_index, 1.0);
            append_chi0_progress_event("hamiltonian_ready",
                                       0,
                                       -1,
                                       -1,
                                       -1,
                                       solved_equations,
                                       nullptr,
                                       -1.0,
                                       elapsed_seconds_since(chi0_start_time),
                                       "spin=" + std::to_string(spin_index + 1)
                                           + " grid_size=" + std::to_string(grid_data.grid.size()));

            SternheimerFDZeroOrderStates states;
            std::vector<SternheimerDeltaGridFunction> lcao_occupied_functions;
            std::vector<SternheimerDeltaGridFunction> lcao_occupied_projector_functions;
            append_chi0_progress_event(
                "zero_order_start",
                0,
                -1,
                -1,
                -1,
                solved_equations,
                nullptr,
                -1.0,
                elapsed_seconds_since(chi0_start_time),
                "spin=" + std::to_string(spin_index + 1) + " source="
                    + (use_lcao_zero_order ? "lcao_sample" : (GlobalV::MY_RANK == 0 ? "fd_solve" : "wait")));
            if (use_lcao_zero_order)
            {
                lcao_occupied_functions.reserve(static_cast<std::size_t>(occupied_count));
                states.eigenvalues.reserve(static_cast<std::size_t>(occupied_count));
                states.wavefunctions.reserve(static_cast<std::size_t>(occupied_count));
                states.residual_norms.reserve(static_cast<std::size_t>(occupied_count));
                for (int ib = 0; ib != occupied_count; ++ib)
                {
                    const auto& coefficients = lcao_channel->coefficients[static_cast<std::size_t>(ib)];
                    if (coefficients.size() != sampled_ao_functions.size())
                    {
                        throw std::runtime_error(
                            "Sternheimer LCAO coefficient basis size does not match sampled AO functions.");
                    }
                    SternheimerDeltaGridFunction occupied_function
                        = linear_combination_delta_sternheimer_grid_functions(sampled_ao_functions, coefficients);
                    const double norm = sternheimer_fd_grid_norm(occupied_function.values, grid_data.volume_element);
                    if (norm <= PARAM.inp.sternheimer_delta_norm_tol)
                    {
                        throw std::runtime_error("Sternheimer sampled LCAO occupied function has zero norm.");
                    }
                    const SternheimerFDHamiltonian::Complex inverse_norm(1.0 / norm, 0.0);
                    for (auto& value: occupied_function.values)
                    {
                        value *= inverse_norm;
                    }
                    for (auto& gradient: occupied_function.gradients)
                    {
                        for (auto& value: gradient)
                        {
                            value *= inverse_norm;
                        }
                    }
                    states.eigenvalues.push_back(elec_state.ekb(spin_index, ib));
                    states.wavefunctions.push_back(occupied_function.values);
                    states.residual_norms.push_back(0.0);
                    lcao_occupied_functions.push_back(std::move(occupied_function));
                }
                lcao_occupied_projector_functions
                    = orthonormalize_delta_sternheimer_grid_functions(lcao_occupied_functions,
                                                                      grid_data.volume_element,
                                                                      PARAM.inp.sternheimer_delta_norm_tol);
            }
            else
            {
                if (!use_frequency_mpi || GlobalV::MY_RANK == 0)
                {
                    states = solve_fd_zero_order_auto(hamiltonian,
                                                      num_bands,
                                                      grid_data.volume_element,
                                                      max_dense_size,
                                                      lanczos_max_subspace_size);
                }
                broadcast_zero_order_states(states, grid_data.grid.size(), use_frequency_mpi);
            }
            append_chi0_progress_event("zero_order_ready",
                                       0,
                                       -1,
                                       -1,
                                       -1,
                                       solved_equations,
                                       nullptr,
                                       -1.0,
                                       elapsed_seconds_since(chi0_start_time),
                                       "spin=" + std::to_string(spin_index + 1)
                                           + " source=" + (use_lcao_zero_order ? "lcao_ks" : "fd_grid")
                                           + " nstates=" + std::to_string(states.wavefunctions.size()));

            const std::vector<SternheimerFDHamiltonian::Vector> occupied
                = occupied_wavefunctions_from_states(states, elec_state, spin_index);
            if (occupied.empty())
            {
                throw std::runtime_error("No occupied zero-order states are available for a Sternheimer spin channel.");
            }
            std::vector<SternheimerFDHamiltonian::Vector> occupied_projector = occupied;
            if (use_lcao_zero_order)
            {
                occupied_projector.clear();
                occupied_projector.reserve(lcao_occupied_projector_functions.size());
                for (const SternheimerDeltaGridFunction& function: lcao_occupied_projector_functions)
                {
                    occupied_projector.push_back(function.values);
                }
            }

            SternheimerDeltaSubspace delta_subspace;
            SternheimerDeltaFixedSubspace delta_fixed_subspace;
            if (use_delta_sternheimer)
            {
                append_chi0_progress_event("delta_subspace_start",
                                           0,
                                           -1,
                                           -1,
                                           -1,
                                           solved_equations,
                                           nullptr,
                                           -1.0,
                                           elapsed_seconds_since(chi0_start_time),
                                           "spin=" + std::to_string(spin_index + 1));
                std::vector<SternheimerDeltaGridFunction> loaded_candidate_functions;
                const std::vector<SternheimerDeltaGridFunction>* candidate_functions = &sampled_ao_functions;
                if (use_lcao_zero_order && lcao_virtual_source == SternheimerLCAOVirtualSource::KSBands)
                {
                    loaded_candidate_functions.reserve(lcao_channel->unoccupied_coefficients.size());
                    for (const auto& coefficients: lcao_channel->unoccupied_coefficients)
                    {
                        loaded_candidate_functions.push_back(
                            linear_combination_delta_sternheimer_grid_functions(sampled_ao_functions, coefficients));
                    }
                    candidate_functions = &loaded_candidate_functions;
                }
                else if (candidate_functions->empty())
                {
                    loaded_candidate_functions = build_lcao_candidate_grid_functions(ucell, grid_data.grid);
                    candidate_functions = &loaded_candidate_functions;
                }
                if (candidate_functions->empty())
                {
                    throw std::runtime_error("Sternheimer delta mode found no sampled LCAO candidate orbitals.");
                }

                std::vector<SternheimerDeltaGridFunction> fd_occupied_functions;
                const std::vector<SternheimerDeltaGridFunction>* occupied_functions
                    = &lcao_occupied_projector_functions;
                if (occupied_functions->empty())
                {
                    fd_occupied_functions.reserve(occupied.size());
                    for (const SternheimerFDHamiltonian::Vector& occupied_wavefunction: occupied)
                    {
                        fd_occupied_functions.push_back(
                            make_delta_sternheimer_grid_function_with_fd_gradients(occupied_wavefunction,
                                                                                   grid_data.grid));
                    }
                    occupied_functions = &fd_occupied_functions;
                }

                SternheimerDeltaSubspaceOptions delta_options;
                delta_options.max_virtual_states = PARAM.inp.sternheimer_delta_max_states;
                delta_options.norm_tolerance = PARAM.inp.sternheimer_delta_norm_tol;
                delta_subspace = build_reference_delta_sternheimer_subspace(hamiltonian,
                                                                            *occupied_functions,
                                                                            *candidate_functions,
                                                                            grid_data.volume_element,
                                                                            delta_options);
                if (delta_subspace.virtual_states.empty())
                {
                    throw std::runtime_error("Sternheimer delta mode produced no fixed virtual states.");
                }
                if (use_lcao_zero_order && lcao_virtual_source == SternheimerLCAOVirtualSource::KSBands)
                {
                    const int expected_virtual_states = expected_sternheimer_ks_virtual_states(
                        static_cast<int>(lcao_channel->unoccupied_coefficients.size()),
                        PARAM.inp.sternheimer_delta_max_states);
                    validate_sternheimer_ks_virtual_subspace(spin_index + 1,
                                                             expected_virtual_states,
                                                             delta_subspace.accepted_candidates,
                                                             static_cast<int>(delta_subspace.virtual_states.size()));
                }
                delta_fixed_subspace
                    = build_delta_sternheimer_fixed_subspace(occupied_projector, delta_subspace.virtual_states);
                append_chi0_progress_event("delta_subspace_ready",
                                           0,
                                           -1,
                                           -1,
                                           -1,
                                           solved_equations,
                                           nullptr,
                                           -1.0,
                                           elapsed_seconds_since(chi0_start_time),
                                           "spin=" + std::to_string(spin_index + 1)
                                               + " nvirtual=" + std::to_string(delta_subspace.virtual_states.size()));
            }

            const SternheimerMemorySnapshot channel_memory = detect_sternheimer_memory_snapshot();
            const SternheimerChannelWorkerPlan channel_worker_plan
                = plan_sternheimer_channel_workers(num_channels,
                                                   sternheimer_channel_openmp_threads(),
                                                   grid_data.grid.size(),
                                                   channel_worker_user_cap,
                                                   channel_memory);
            append_chi0_progress_event(
                "channel_workers_ready",
                0,
                -1,
                -1,
                -1,
                solved_equations,
                nullptr,
                -1.0,
                elapsed_seconds_since(chi0_start_time),
                "spin=" + std::to_string(spin_index + 1) + " "
                    + format_sternheimer_channel_worker_diagnostic(channel_memory,
                                                                   channel_worker_plan,
                                                                   grid_data.grid.size(),
                                                                   channel_worker_user_cap));

            for (int ifrequency = 0; ifrequency != nfreq; ++ifrequency)
            {
                const SternheimerRPA::FrequencyMPIAssignment assignment = frequency_assignment(ifrequency);
                const int owner_rank = assignment.frequency_leader_rank;
                if (!assignment.owns_frequency)
                {
                    continue;
                }
                const double omega_ha = frequency_grid.omega_ha[static_cast<std::size_t>(ifrequency)];
                const double omega_ry = 2.0 * omega_ha;
                std::vector<SternheimerRPA::Complex>* chi0_branch
                    = write_librpa ? &chi0_branches[static_cast<std::size_t>(ifrequency)] : nullptr;

                for (int ib = 0; ib != static_cast<int>(states.wavefunctions.size()); ++ib)
                {
                    const double occupation = elec_state.wg(spin_index, ib);
                    if (occupation <= 1.0e-8)
                    {
                        continue;
                    }

                    struct ChannelEquationResult
                    {
                        int channel_index = -1;
                        int owner_rank = -1;
                        SternheimerRPA::SolverResult solver;
                        double equation_residual_norm = 0.0;
                        bool has_siab_row = false;
                        siab::ReferenceRow siab_row;
                    };

                    std::vector<int> owned_channels;
                    std::vector<int> equation_owner_ranks;
                    owned_channels.reserve(static_cast<std::size_t>(num_channels));
                    equation_owner_ranks.reserve(static_cast<std::size_t>(num_channels));
                    for (int ichannel = 0; ichannel != num_channels; ++ichannel)
                    {
                        int equation_owner_rank = owner_rank;
                        if (use_global_equation_mpi)
                        {
                            equation_owner_rank = SternheimerRPA::global_equation_owner(occupied_state_offset + ib,
                                                                                        ifrequency,
                                                                                        ichannel,
                                                                                        nfreq,
                                                                                        num_channels,
                                                                                        GlobalV::NPROC,
                                                                                        frequency_rank_shift);
                            if (equation_owner_rank != GlobalV::MY_RANK)
                            {
                                continue;
                            }
                        }
                        else if (use_channel_mpi)
                        {
                            const int group_owner = SternheimerRPA::channel_group_owner(occupied_state_offset + ib,
                                                                                       ichannel,
                                                                                       num_channels,
                                                                                       assignment.frequency_group_size);
                            equation_owner_rank = assignment.frequency_leader_rank + group_owner;
                            if (group_owner != assignment.frequency_group_local_rank)
                            {
                                continue;
                            }
                        }
                        owned_channels.push_back(ichannel);
                        equation_owner_ranks.push_back(equation_owner_rank);
                    }

                    std::vector<ChannelEquationResult> channel_results
                        = run_sternheimer_channel_tasks<ChannelEquationResult>(
                            static_cast<int>(owned_channels.size()),
                            [&](const int local_task) {
                              const int ichannel = owned_channels[static_cast<std::size_t>(local_task)];
                              const std::size_t channel_index = static_cast<std::size_t>(ichannel);
                              SternheimerFDHamiltonian::Vector rhs;
                              SternheimerRPA::build_rhs_from_hartree_perturbation(perturbations_ry[channel_index],
                                                                                  states.wavefunctions[ib],
                                                                                  rhs);
                              SternheimerFDHamiltonian::Vector delta_wavefunction;
                              ChannelEquationResult result;
                              result.channel_index = ichannel;
                              result.owner_rank = equation_owner_ranks[static_cast<std::size_t>(local_task)];
                              if (use_delta_sternheimer)
                              {
                                  const std::vector<SternheimerFDHamiltonian::Complex> perturbation_matrix_elements
                                      = delta_sternheimer_perturbation_matrix_elements(delta_subspace.virtual_states,
                                                                                       perturbations_ry[channel_index],
                                                                                       states.wavefunctions[ib],
                                                                                       grid_data.volume_element);
                                  const SternheimerDeltaLinearResponse response
                                      = solve_delta_sternheimer_linear_response(hamiltonian,
                                                                                delta_fixed_subspace,
                                                                                states.eigenvalues[ib],
                                                                                rhs,
                                                                                delta_subspace.virtual_states,
                                                                                perturbation_matrix_elements,
                                                                                omega_ry,
                                                                                grid_data.volume_element,
                                                                                solver_options);
                                  delta_wavefunction = response.response.reconstructed_wavefunction;
                                  result.solver = response.solver;
                                  result.equation_residual_norm = response.residual_norm;
                                  if (write_siab)
                                  {
                                      const auto& complete_response = response.response.reconstructed_wavefunction;
                                      if (complete_response.size() != static_cast<std::size_t>(grid_data.grid.size()))
                                      {
                                          throw std::runtime_error(
                                              "Sternheimer SIAB requires each equation owner to hold a complete response grid.");
                                      }
                                      result.has_siab_row = true;
                                      result.siab_row.occupied_state = occupied_state_offset + ib;
                                      result.siab_row.auxiliary_channel = ichannel;
                                      result.siab_row.frequency_index = ifrequency;
                                      result.siab_row.frequency_ha = omega_ha;
                                      result.siab_row.occupation = occupation;
                                      result.siab_row.frequency_weight
                                          = frequency_grid.weights_ha[static_cast<std::size_t>(ifrequency)];
                                      result.siab_row.norm = siab::norm(complete_response, grid_data.volume_element);
                                      result.siab_row.q = project_siab_response_to_primitives(complete_response,
                                                                                              ucell,
                                                                                              siab_primitives);
                                  }
                              }
                              else
                              {
                                  const SternheimerFDLinearResponse response
                                      = solve_sternheimer_fd_linear_response(hamiltonian,
                                                                             occupied,
                                                                             states.eigenvalues[ib],
                                                                             rhs,
                                                                             omega_ry,
                                                                             grid_data.volume_element,
                                                                             solver_options);
                                  delta_wavefunction = response.delta_wavefunction;
                                  result.solver = response.solver;
                                  result.equation_residual_norm = response.residual_norm;
                              }
                              if (write_librpa)
                              {
                                  SternheimerRPA::accumulate_chi0_branch_column(potentials,
                                                                                states.wavefunctions[ib],
                                                                                delta_wavefunction,
                                                                                grid_data.volume_element,
                                                                                occupation,
                                                                                ichannel,
                                                                                *chi0_branch);
                              }
                              return result;
                            },
                            channel_worker_plan.effective_workers);

                    for (ChannelEquationResult& result: channel_results)
                    {
                        if (result.has_siab_row)
                        {
                            local_siab_rows.push_back(std::move(result.siab_row));
                        }
                        all_converged = all_converged && result.solver.converged;
                        ++solved_equations;
                        local_iteration_sum += result.solver.iterations;
                        max_solver_relative_residual
                            = std::max(max_solver_relative_residual, result.solver.relative_residual);
                        max_equation_residual_norm
                            = std::max(max_equation_residual_norm, result.equation_residual_norm);
                        append_chi0_progress_event("equation",
                                                   ifrequency + 1,
                                                   result.owner_rank,
                                                   ib,
                                                   result.channel_index,
                                                   solved_equations,
                                                   &result.solver,
                                                   result.equation_residual_norm,
                                                   elapsed_seconds_since(chi0_start_time),
                                                   "spin=" + std::to_string(spin_index + 1)
                                                       + " mode=" + (use_delta_sternheimer ? "delta" : "standard"));
                    }
                }
            }

            SpinResponseDiagnostics diagnostics;
            diagnostics.spin_index = spin_index;
            diagnostics.occupied_bands = static_cast<int>(occupied.size());
            diagnostics.occupied_projector_dimension = static_cast<int>(occupied_projector.size());
            diagnostics.delta_virtual_states = static_cast<int>(delta_subspace.virtual_states.size());
            diagnostics.delta_accepted_candidates = delta_subspace.accepted_candidates;
            diagnostics.delta_discarded_candidates = delta_subspace.discarded_candidates;
            spin_diagnostics.push_back(diagnostics);
            occupied_state_offset += static_cast<int>(states.wavefunctions.size());
        }

        const int local_solved_equations = solved_equations;
        int rank_local_equations_min = local_solved_equations;
        int rank_local_equations_max = local_solved_equations;
        std::int64_t rank_local_iterations_min = local_iteration_sum;
        std::int64_t rank_local_iterations_max = local_iteration_sum;
#ifdef __MPI
        if (use_distributed_mpi && GlobalV::NPROC > 1)
        {
            MPI_Allreduce(&local_solved_equations,
                          &rank_local_equations_min,
                          1,
                          MPI_INT,
                          MPI_MIN,
                          MPI_COMM_WORLD);
            MPI_Allreduce(&local_solved_equations,
                          &rank_local_equations_max,
                          1,
                          MPI_INT,
                          MPI_MAX,
                          MPI_COMM_WORLD);
            MPI_Allreduce(&local_iteration_sum,
                          &rank_local_iterations_min,
                          1,
                          MPI_INT64_T,
                          MPI_MIN,
                          MPI_COMM_WORLD);
            MPI_Allreduce(&local_iteration_sum,
                          &rank_local_iterations_max,
                          1,
                          MPI_INT64_T,
                          MPI_MAX,
                          MPI_COMM_WORLD);
        }
#endif

        std::vector<siab::ReferenceRow> global_siab_rows;
        if (write_siab)
        {
#ifdef __MPI
            global_siab_rows = siab::gather_reference_rows_to_root(local_siab_rows,
                                                                   static_cast<std::size_t>(siab_primitives.primitive_count),
                                                                   0,
                                                                   MPI_COMM_WORLD);
#else
            global_siab_rows = siab::gather_reference_rows_to_root(local_siab_rows,
                                                                   static_cast<std::size_t>(siab_primitives.primitive_count),
                                                                   0);
#endif
            if (GlobalV::MY_RANK == 0)
            {
                std::size_t occupied_total = 0;
                for (const int occupied_count: occupied_band_counts)
                {
                    occupied_total += static_cast<std::size_t>(occupied_count);
                }
                const std::size_t expected_rows = occupied_total * static_cast<std::size_t>(num_channels)
                                                  * static_cast<std::size_t>(nfreq);
                if (global_siab_rows.size() != expected_rows)
                {
                    throw std::runtime_error("Sternheimer SIAB global row assembly has missing or duplicate rows.");
                }
                const siab::Provenance provenance = make_siab_production_provenance(ucell,
                                                                                    auxiliary_basis_sha256,
                                                                                    frequency_grid,
                                                                                    pca_threshold,
                                                                                    coulomb_whitening);
                const std::string siab_path = join_output_path(output_dir, "sternheimer_matrix.dat");
                siab::write_v1(siab_path,
                               grid_data.volume_element,
                               siab_primitives.blocks,
                               global_siab_rows,
                               siab_primitives.overlap_s,
                               provenance);
                GlobalV::ofs_running << " Sternheimer SIAB v1 output: " << siab_path << std::endl;
            }
        }

        for (int ifrequency = 0; ifrequency != nfreq; ++ifrequency)
        {
            const SternheimerRPA::FrequencyMPIAssignment assignment = frequency_assignment(ifrequency);
            const int owner_rank = assignment.frequency_leader_rank;
            if (!assignment.owns_frequency)
            {
                continue;
            }
            const double omega_ha = frequency_grid.omega_ha[static_cast<std::size_t>(ifrequency)];
            if (write_librpa)
            {
#ifdef __MPI
                if (use_channel_mpi && assignment.frequency_group_size > 1)
                {
                    sternheimer_chi0::reduce_branch_to_root(
                        chi0_branches[static_cast<std::size_t>(ifrequency)],
                        0,
                        chi0_frequency_group_communicator);
                }
#endif
                const bool writes_frequency
                    = !use_channel_mpi || assignment.frequency_group_local_rank == 0;
                if (!writes_frequency)
                {
                    append_chi0_progress_event("frequency_finish",
                                               ifrequency + 1,
                                               owner_rank,
                                               -1,
                                               -1,
                                               solved_equations,
                                               nullptr,
                                               -1.0,
                                               elapsed_seconds_since(chi0_start_time),
                                               "chi0_reduced_to_leader=yes");
                    continue;
                }
                const std::vector<SternheimerRPA::Complex> chi0
                    = SternheimerRPA::symmetrize_chi0_imaginary_frequency(
                        chi0_branches[static_cast<std::size_t>(ifrequency)],
                        num_channels);
                const SternheimerRPA::Chi0V1Metadata metadata
                    = make_chi0_v1_metadata(ucell,
                                            channels,
                                            ifrequency + 1,
                                            omega_ha,
                                            frequency_grid.weights_ha[static_cast<std::size_t>(ifrequency)]);
                const std::string data_file = chi0_v1_filename(metadata.iq, metadata.ifrequency);
                SternheimerRPA::write_chi0_v1_file(data_file, metadata, auxiliary_channels, chi0);
                GlobalV::ofs_running << " Sternheimer chi0 v1 output: " << data_file << std::endl;
            }
            append_chi0_progress_event("frequency_finish",
                                       ifrequency + 1,
                                       owner_rank,
                                       -1,
                                       -1,
                                       solved_equations,
                                       nullptr,
                                       -1.0,
                                       elapsed_seconds_since(chi0_start_time),
                                       "elapsed_freq_s="
                                           + std::to_string(elapsed_seconds_since(
                                               frequency_start_times[static_cast<std::size_t>(ifrequency)])));
        }

#ifdef __MPI
        if (owns_chi0_frequency_group_communicator)
        {
            MPI_Comm_free(&chi0_frequency_group_communicator);
        }
#endif

        reduce_chi0_output_stats(all_converged,
                                 solved_equations,
                                 max_solver_relative_residual,
                                 max_equation_residual_norm,
                                 use_frequency_mpi);

#ifdef __MPI
        if (use_frequency_mpi && GlobalV::NPROC > 1)
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }
#endif

        std::vector<std::pair<std::string, SternheimerRPA::Chi0V1Metadata>> index_entries;
        if (write_librpa && GlobalV::MY_RANK == 0)
        {
            index_entries.reserve(frequency_grid.omega_ha.size());
            for (int ifrequency = 0; ifrequency != nfreq; ++ifrequency)
            {
                const SternheimerRPA::Chi0V1Metadata metadata
                    = make_chi0_v1_metadata(ucell,
                                            channels,
                                            ifrequency + 1,
                                            frequency_grid.omega_ha[static_cast<std::size_t>(ifrequency)],
                                            frequency_grid.weights_ha[static_cast<std::size_t>(ifrequency)]);
                const int owner_rank = frequency_assignment(ifrequency).frequency_leader_rank;
                index_entries.push_back({chi0_v1_filename(metadata.iq, metadata.ifrequency, owner_rank), metadata});
            }
            write_chi0_index_file("v1_sternheimer_chi0_index.dat", index_entries);
        }

        const int grid_size = grid_data.grid.nx * grid_data.grid.ny * grid_data.grid.nz;
        if (GlobalV::MY_RANK != 0)
        {
            return;
        }

        out << "status success\n";
        out << "format " << (write_siab ? "siab_v1" : "librpa_v1") << '\n';
        out << "data_files " << (write_siab ? 1 : index_entries.size()) << '\n';
        out << "index_file " << (write_siab ? "none" : "v1_sternheimer_chi0_index.dat") << '\n';
        out << "grid " << grid_data.grid.nx << ' ' << grid_data.grid.ny << ' ' << grid_data.grid.nz << " size "
            << grid_size << " dV " << grid_data.volume_element << '\n';
        out << "nfreq " << nfreq << '\n';
        out << "frequency_grid_source " << frequency_grid_source << '\n';
        if (use_frequency_grid_file)
        {
            out << "frequency_grid_file " << frequency_grid_file << '\n';
        }
        if (transition_window_available)
        {
            out << "transition_window_Ha " << transition_window.emin_ha << ' ' << transition_window.emax_ha << '\n';
        }
        else
        {
            out << "transition_window_Ha unavailable_external_grid\n";
        }
        out << "sternheimer_frequency_mpi " << (use_frequency_mpi ? "yes" : "no") << '\n';
        out << "sternheimer_channel_mpi " << (use_channel_mpi ? "yes" : "no") << '\n';
        out << "sternheimer_mpi_layout " << mpi_layout << '\n';
        const char* equation_owner_formula
            = use_global_equation_mpi ? "occupied_frequency_channel_modulo"
              : use_channel_mpi      ? "frequency_group_then_occupied_channel_modulo"
              : use_frequency_mpi    ? "frequency_round_robin"
                                     : "serial";
        out << "equation_owner_formula " << equation_owner_formula << '\n';
        out << "frequency_group_size " << frequency_group_size << '\n';
        out << "mpi_ranks " << GlobalV::NPROC << '\n';
        out << "frequency_rank_shift " << frequency_rank_shift << '\n';
        out << "rank_local_equations_min " << rank_local_equations_min << '\n';
        out << "rank_local_equations_max " << rank_local_equations_max << '\n';
        out << "rank_local_iterations_min " << rank_local_iterations_min << '\n';
        out << "rank_local_iterations_max " << rank_local_iterations_max << '\n';
        out << "progress_file_pattern "
            << (write_siab ? "STERNHEIMER_SIAB_PROGRESS_rank*.dat"
                           : "STERNHEIMER_CHI0_PROGRESS_rank*.dat")
            << '\n';
        out << "ifrequency omega_Ha weight_Ha omega_Ry data_file\n";
        for (const auto& entry: index_entries)
        {
            const SternheimerRPA::Chi0V1Metadata& metadata = entry.second;
            out << metadata.ifrequency << ' ' << metadata.omega << ' ' << metadata.weight << ' ' << 2.0 * metadata.omega
                << ' ' << entry.first << '\n';
        }
        out << "pca_threshold " << pca_threshold << '\n';
        out << "ccp_rmesh_times " << ccp_rmesh_times << '\n';
        out << "solver_tolerance " << solver_tolerance << '\n';
        out << "sternheimer_zero_order_source " << (use_lcao_zero_order ? "lcao_ks" : "fd_grid") << '\n';
        if (use_lcao_zero_order)
        {
            out << "sternheimer_lcao_virtual_source "
                << sternheimer_lcao_virtual_source_name(lcao_virtual_source) << '\n';
            out << "sternheimer_lcao_unoccupied_bands_per_spin";
            for (const int count: sternheimer_lcao_unoccupied_bands_per_spin(*lcao_occupied_channels))
            {
                out << ' ' << count;
            }
            out << '\n';
        }
        out << "sternheimer_response_spin_channels " << spin_diagnostics.size() << '\n';
        out << "sternheimer_response_spin_indices";
        for (const SpinResponseDiagnostics& diagnostics: spin_diagnostics)
        {
            out << ' ' << diagnostics.spin_index + 1;
        }
        out << '\n';
        if (spin_diagnostics.size() == 1)
        {
            out << "sternheimer_response_spin_channel " << spin_diagnostics.front().spin_index + 1 << '\n';
        }
        int total_occupied_bands = 0;
        out << "occupied_bands_per_spin";
        for (const SpinResponseDiagnostics& diagnostics: spin_diagnostics)
        {
            out << ' ' << diagnostics.occupied_bands;
            total_occupied_bands += diagnostics.occupied_bands;
        }
        out << '\n';
        out << "occupied_bands " << total_occupied_bands << '\n';
        out << "occupied_projector_dimensions_per_spin";
        for (const SpinResponseDiagnostics& diagnostics: spin_diagnostics)
        {
            out << ' ' << diagnostics.occupied_projector_dimension;
        }
        out << '\n';
        if (spin_diagnostics.size() == 1)
        {
            out << "occupied_projector_dimension " << spin_diagnostics.front().occupied_projector_dimension << '\n';
        }
        out << "abfs_channels " << raw_num_channels << '\n';
        out << "response_channels " << num_channels << '\n';
        out << "abfs_source " << abfs_source << '\n';
        if (write_siab)
        {
            out << "auxiliary_basis_sha256 " << auxiliary_basis_sha256 << '\n';
            out << "raw_auxiliary_dimension " << raw_num_channels << '\n';
            out << "whitened_auxiliary_rank " << coulomb_whitening.retained_rank << '\n';
            out << "discarded_auxiliary_rank " << coulomb_whitening.discarded_rank << '\n';
            out << "coulomb_relative_threshold " << coulomb_whitening.relative_threshold << '\n';
            out << "coulomb_max_orthonormality_error " << coulomb_whitening.max_orthonormality_error << '\n';
            out << "coulomb_transform_sha256 " << hash_coulomb_whitening_transform(coulomb_whitening) << '\n';
            out << "coulomb_whitening_diagnostic STERNHEIMER_SIAB_COULOMB_WHITENING.dat\n";
            out << "primitive_representation serial_reciprocal_pw_v1\n";
            out << "primitive_count " << siab_primitives.primitive_count << '\n';
            out << "primitive_reciprocal_count " << siab_primitives.reciprocal_count << '\n';
            out << "estimated_dense_memory_bytes " << siab_memory_estimate.total_bytes << '\n';
            out << "slurm_memory_per_node_bytes " << siab_slurm_memory_bytes << '\n';
            out << "memory_diagnostic STERNHEIMER_SIAB_MEMORY.dat\n";
            out << "target_file sternheimer_matrix.dat\n";
        }
        out << "sternheimer_delta " << (use_delta_sternheimer ? "yes" : "no") << '\n';
        if (use_delta_sternheimer)
        {
            out << "sternheimer_delta_backend reference_value_gradient\n";
            out << "sternheimer_delta_max_states " << PARAM.inp.sternheimer_delta_max_states << '\n';
            out << "sternheimer_delta_norm_tol " << PARAM.inp.sternheimer_delta_norm_tol << '\n';
            out << "sternheimer_delta_virtual_states_per_spin";
            for (const SpinResponseDiagnostics& diagnostics: spin_diagnostics)
            {
                out << ' ' << diagnostics.delta_virtual_states;
            }
            out << '\n';
            out << "sternheimer_delta_accepted_candidates_per_spin";
            for (const SpinResponseDiagnostics& diagnostics: spin_diagnostics)
            {
                out << ' ' << diagnostics.delta_accepted_candidates;
            }
            out << '\n';
            out << "sternheimer_delta_discarded_candidates_per_spin";
            for (const SpinResponseDiagnostics& diagnostics: spin_diagnostics)
            {
                out << ' ' << diagnostics.delta_discarded_candidates;
            }
            out << '\n';
            if (spin_diagnostics.size() == 1)
            {
                out << "sternheimer_delta_virtual_states " << spin_diagnostics.front().delta_virtual_states << '\n';
                out << "sternheimer_delta_accepted_candidates " << spin_diagnostics.front().delta_accepted_candidates
                    << '\n';
                out << "sternheimer_delta_discarded_candidates " << spin_diagnostics.front().delta_discarded_candidates
                    << '\n';
            }
        }
        out << "solved_equations " << solved_equations << '\n';
        out << "all_converged " << (all_converged ? "yes" : "no") << '\n';
        out << "max_solver_relative_residual " << max_solver_relative_residual << '\n';
        out << "max_equation_residual_norm " << max_equation_residual_norm << '\n';
        GlobalV::ofs_running << " Sternheimer chi0 status: " << status_path << std::endl;
    }
    catch (const std::exception& error)
    {
        if (GlobalV::MY_RANK == 0 && out)
        {
            out << "status failed\n";
            out << "reason " << error.what() << '\n';
        }
        GlobalV::ofs_running << ' ' << output_label << " failed: " << error.what() << std::endl;
        GlobalV::ofs_running << ' ' << output_label << " status: " << status_path << std::endl;
#ifdef __MPI
        if (use_distributed_mpi && GlobalV::NPROC > 1)
        {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
#endif
    }
}

void run_sternheimer_abacus_chi0_output(const elecstate::Potential& potential,
                                        const ModulePW::PW_Basis& pw_basis,
                                        const UnitCell& ucell,
                                        const elecstate::ElecState& elec_state,
                                        const std::string& output_dir)
{
    run_sternheimer_abacus_chi0_output_impl(
        potential, pw_basis, ucell, elec_state, output_dir, nullptr, nullptr, nullptr, nullptr, nullptr);
}

void run_sternheimer_abacus_lcao_chi0_output(const elecstate::Potential& potential,
                                             const ModulePW::PW_Basis& pw_basis,
                                             const UnitCell& ucell,
                                             const elecstate::ElecState& elec_state,
                                             const LCAO_Orbitals& orbitals,
                                             const std::vector<SternheimerLCAOOccupiedChannel>& occupied_channels,
                                             const SternheimerLCAOFixedAOMatrices& fixed_ao_matrices,
                                             const ModulePW::PW_Basis_K* pw_wfc,
                                             const Structure_Factor* structure_factor,
                                             const std::string& output_dir)
{
    run_sternheimer_abacus_chi0_output_impl(potential,
                                            pw_basis,
                                            ucell,
                                            elec_state,
                                            output_dir,
                                            &orbitals,
                                            &occupied_channels,
                                            &fixed_ao_matrices,
                                            pw_wfc,
                                            structure_factor);
}

} // namespace ModuleRI
