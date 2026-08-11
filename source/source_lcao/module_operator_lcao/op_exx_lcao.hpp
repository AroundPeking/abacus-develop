#ifndef OPEXXLCAO_HPP
#define OPEXXLCAO_HPP
#ifdef __EXX

#include "op_exx_lcao.h"
#include "source_base/parallel_reduce.h"
#include "source_base/global_variable.h"
#include "source_io/module_parameter/parameter.h"
#include "source_lcao/module_ri/RI_2D_Comm.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_io/module_restart/restart_exx_csr.h"
#include "source_lcao/module_rt/td_info.h"
#include "source_io/module_restart/restart.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace hamilt
{
    using TAC = std::pair<int, std::array<int, 3>>;

    namespace
    {
        inline bool abacus_debug_dump_exx_ao_enabled()
        {
            const char* env = std::getenv("ABACUS_DUMP_EXX_AO");
            if (env == nullptr)
            {
                return false;
            }
            const std::string value(env);
            return !(value.empty() || value == "0" || value == "f" || value == "F"
                     || value == "false" || value == "FALSE");
        }

        inline bool abacus_debug_k_is_offgrid(const ModuleBase::Vector3<double>& kfrac,
                                              const std::array<int, 3>& nmp)
        {
            if (nmp[0] <= 0 || nmp[1] <= 0 || nmp[2] <= 0)
            {
                return true;
            }
            const double tol = 1e-10;
            return std::abs(kfrac.x * nmp[0] - std::round(kfrac.x * nmp[0])) > tol
                   || std::abs(kfrac.y * nmp[1] - std::round(kfrac.y * nmp[1])) > tol
                   || std::abs(kfrac.z * nmp[2] - std::round(kfrac.z * nmp[2])) > tol;
        }

        inline std::string abacus_debug_format_double(const double value)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(6) << value;
            std::string text = oss.str();
            std::replace(text.begin(), text.end(), '-', 'm');
            std::replace(text.begin(), text.end(), '.', 'p');
            return text;
        }

        inline std::complex<double> abacus_debug_to_complex(const double value)
        {
            return std::complex<double>(value, 0.0);
        }

        inline std::complex<double> abacus_debug_to_complex(const std::complex<double>& value)
        {
            return value;
        }

        template <typename T>
        void abacus_debug_dump_exx_ao_delta_if_requested(const K_Vectors& kv,
                                                         const int ik,
                                                         const Parallel_Orbitals* pv,
                                                         const T* hk_before,
                                                         const T* hk_after)
        {
            if (!abacus_debug_dump_exx_ao_enabled())
            {
                return;
            }
            if (GlobalV::NPROC != 1 || GlobalV::MY_RANK != 0 || pv == nullptr)
            {
                return;
            }

            const auto kfrac = kv.kvec_d[ik];
            const std::array<int, 3> nmp{kv.nmp[0], kv.nmp[1], kv.nmp[2]};
            if (!abacus_debug_k_is_offgrid(kfrac, nmp))
            {
                std::cout << "[ABACUS_DUMP_EXX_AO] skip ik=" << ik
                          << " because k is on the mesh: ("
                          << kfrac.x << ", " << kfrac.y << ", " << kfrac.z
                          << "), nmp=(" << nmp[0] << ", " << nmp[1] << ", " << nmp[2]
                          << ")" << std::endl;
                return;
            }

            const int nrow = pv->get_row_size();
            const int ncol = pv->get_col_size();
            if (nrow <= 0 || ncol <= 0 || nrow != ncol)
            {
                std::cout << "[ABACUS_DUMP_EXX_AO] skip ik=" << ik
                          << " because local HK shape is invalid: "
                          << nrow << "x" << ncol << std::endl;
                return;
            }

            std::ostringstream filename;
            filename << PARAM.globalv.global_out_dir << "abacus_hexx_ao_ik" << ik << "_kx_"
                     << abacus_debug_format_double(kfrac.x) << "_ky_"
                     << abacus_debug_format_double(kfrac.y) << "_kz_"
                     << abacus_debug_format_double(kfrac.z) << ".mtx";
            std::ofstream ofs(filename.str());
            if (!ofs.good())
            {
                std::cout << "[ABACUS_DUMP_EXX_AO] failed to open " << filename.str()
                          << std::endl;
                return;
            }

            std::cout << "[ABACUS_DUMP_EXX_AO] writing " << filename.str() << std::endl;

            ofs << "%%MatrixMarket matrix coordinate complex general\n";
            ofs << std::scientific << std::setprecision(15);
            ofs << nrow << " " << ncol << " " << static_cast<long long>(nrow) * ncol << "\n";
            for (int i = 0; i != nrow; ++i)
            {
                for (int j = 0; j != ncol; ++j)
                {
                    const int index = j * nrow + i;
                    const auto delta = abacus_debug_to_complex(hk_after[index])
                                       - abacus_debug_to_complex(hk_before[index]);
                    ofs << (i + 1) << " " << (j + 1) << " " << delta.real() << " "
                        << delta.imag() << "\n";
                }
            }
        }
    } // namespace

    template <typename Tdata>
    std::pair<bool, std::array<int, 3>>
    infer_complete_Rs_period_from_Hexxs(
        const std::vector<std::map<int, std::map<TAC, RI::Tensor<Tdata>>>>& Hexxs)
    {
        std::array<int, 3> Rs_period = {0, 0, 0};
        if (Hexxs.empty())
        {
            return {false, Rs_period};
        }

        std::set<std::array<int, 3>> unique_Rs;
        std::array<int, 3> min_R = {
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max()};
        std::array<int, 3> max_R = {
            std::numeric_limits<int>::min(),
            std::numeric_limits<int>::min(),
            std::numeric_limits<int>::min()};

        for (const auto& Htmp1 : Hexxs[0])
        {
            for (const auto& Htmp2 : Htmp1.second)
            {
                const std::array<int, 3>& cell = Htmp2.first.second;
                if (unique_Rs.insert(cell).second)
                {
                    for (int idim = 0; idim < 3; ++idim)
                    {
                        min_R[idim] = std::min(min_R[idim], cell[idim]);
                        max_R[idim] = std::max(max_R[idim], cell[idim]);
                    }
                }
            }
        }

        if (unique_Rs.empty())
        {
            return {false, Rs_period};
        }

        size_t expected_nR = 1;
        for (int idim = 0; idim < 3; ++idim)
        {
            Rs_period[idim] = max_R[idim] - min_R[idim] + 1;
            if (Rs_period[idim] <= 0)
            {
                return {false, {0, 0, 0}};
            }
            expected_nR *= static_cast<size_t>(Rs_period[idim]);
        }

        if (expected_nR != unique_Rs.size())
        {
            return {false, {0, 0, 0}};
        }

        return {true, Rs_period};
    }

    inline bool can_remap_wigner_seitz_for_nscf(const Add_Hexx_Type add_hexx_type,
                                                const bool period_from_kmesh,
                                                const bool zero_koffset)
    {
        return add_hexx_type == Add_Hexx_Type::R
               && (!period_from_kmesh || zero_koffset);
    }

    struct WignerSeitzRemapStats
    {
        size_t input_blocks = 0;
        size_t output_blocks = 0;
        size_t remapped_blocks = 0;
        size_t split_blocks = 0;
        size_t max_images = 0;
    };

    inline std::vector<std::array<int, 3>> find_nearest_bvk_cells(
        const UnitCell& ucell,
        const int iat0,
        const int iat1,
        const std::array<int, 3>& R,
        const std::array<int, 3>& period)
    {
        // Boundary-equivalent BvK images must all carry the same matrix weight.
        double min_distance_sq = std::numeric_limits<double>::max();
        std::vector<std::array<int, 3>> nearest_cells;

        for (int ix = -1; ix <= 1; ++ix)
        {
            for (int iy = -1; iy <= 1; ++iy)
            {
                for (int iz = -1; iz <= 1; ++iz)
                {
                    const std::array<int, 3> candidate = {
                        R[0] + ix * period[0],
                        R[1] + iy * period[1],
                        R[2] + iz * period[2]};
                    const ModuleBase::Vector3<int> candidate_vector(
                        candidate[0], candidate[1], candidate[2]);
                    const double distance_sq
                        = ucell.cal_dtau(iat0, iat1, candidate_vector).norm2();
                    const double tolerance
                        = 1.0e-10
                          * std::max(1.0,
                                     std::max(std::abs(distance_sq),
                                              std::abs(min_distance_sq)));

                    if (distance_sq + tolerance < min_distance_sq)
                    {
                        min_distance_sq = distance_sq;
                        nearest_cells.clear();
                        nearest_cells.push_back(candidate);
                    }
                    else if (std::abs(distance_sq - min_distance_sq) <= tolerance)
                    {
                        nearest_cells.push_back(candidate);
                    }
                }
            }
        }
        return nearest_cells;
    }

    template <typename Tdata>
    WignerSeitzRemapStats remap_Hexxs_wigner_seitz(
        const UnitCell& ucell,
        const std::array<int, 3>& period,
        std::vector<std::map<int, std::map<TAC, RI::Tensor<Tdata>>>>& Hexxs)
    {
        WignerSeitzRemapStats stats;
        std::vector<std::map<int, std::map<TAC, RI::Tensor<Tdata>>>> remapped(
            Hexxs.size());

        for (size_t is = 0; is < Hexxs.size(); ++is)
        {
            for (const auto& Htmp1 : Hexxs[is])
            {
                const int iat0 = Htmp1.first;
                for (const auto& Htmp2 : Htmp1.second)
                {
                    ++stats.input_blocks;
                    const int iat1 = Htmp2.first.first;
                    const std::array<int, 3>& R = Htmp2.first.second;
                    const auto nearest_cells
                        = find_nearest_bvk_cells(ucell, iat0, iat1, R, period);
                    if (nearest_cells.empty())
                    {
                        throw std::runtime_error("No nearest BvK image found for HexxR block");
                    }

                    stats.max_images = std::max(stats.max_images, nearest_cells.size());
                    if (nearest_cells.size() != 1 || nearest_cells.front() != R)
                    {
                        ++stats.remapped_blocks;
                    }
                    if (nearest_cells.size() > 1)
                    {
                        ++stats.split_blocks;
                    }

                    const Tdata weight
                        = static_cast<Tdata>(1.0 / static_cast<double>(nearest_cells.size()));
                    auto& target = remapped[is][iat0];
                    for (const auto& R_bvk : nearest_cells)
                    {
                        const TAC key = {iat1, R_bvk};
                        auto it = target.find(key);
                        if (it == target.end())
                        {
                            target.emplace(key, weight * Htmp2.second);
                        }
                        else
                        {
                            it->second += weight * Htmp2.second;
                        }
                    }
                }
            }
        }

        Hexxs.swap(remapped);
        for (const auto& Hspin : Hexxs)
        {
            for (const auto& Htmp1 : Hspin)
            {
                stats.output_blocks += Htmp1.second.size();
            }
        }
        return stats;
    }

    // allocate according to the read-in HexxR, used in nscf
    template <typename Tdata, typename TR>
    void reallocate_hcontainer(const std::vector<std::map<int, std::map<TAC, RI::Tensor<Tdata>>>>& Hexxs,
        HContainer<TR>* hR,
        const RI::Cell_Nearest<int, int, 3, double, 3>* const cell_nearest)
    {
        auto* pv = hR->get_paraV();
        bool need_allocate = false;
        for (auto& Htmp1 : Hexxs[0])
        {
            const int& iat0 = Htmp1.first;
            for (auto& Htmp2 : Htmp1.second)
            {
                const int& iat1 = Htmp2.first.first;
                if (pv->get_row_size(iat0) > 0 && pv->get_col_size(iat1) > 0)
                {
                    const Abfs::Vector3_Order<int>& R = RI_Util::array3_to_Vector3(
                        (cell_nearest ?
                            cell_nearest->get_cell_nearest_discrete(iat0, iat1, Htmp2.first.second)
                            : Htmp2.first.second));
                    BaseMatrix<TR>* HlocR = hR->find_matrix(iat0, iat1, R.x, R.y, R.z);
                    if (HlocR == nullptr)
                    { // add R to HContainer
                        need_allocate = true;
                        AtomPair<TR> tmp(iat0, iat1, R.x, R.y, R.z, pv);
                        hR->insert_pair(tmp);
                    }
                }
            }
        }
        if (need_allocate) { hR->allocate(nullptr, true); }
    }

    /// allocate according to BvK cells, used in scf
    template <typename TR>
    void reallocate_hcontainer(const int nat, HContainer<TR>* hR,
        const std::array<int, 3>& Rs_period,
        const RI::Cell_Nearest<int, int, 3, double, 3>* const cell_nearest)
    {
        auto* pv = hR->get_paraV();
        auto Rs = RI_Util::get_Born_von_Karmen_cells(Rs_period);
        bool need_allocate = false;
        for (int iat0 = 0;iat0 < nat;++iat0)
        {
            for (int iat1 = 0;iat1 < nat;++iat1)
            {
                // complete the atom pairs that has orbitals in this processor but not in hR due to the adj_list 
                // but adj_list is not enought for EXX, which is more nonlocal than Nonlocal 
                if(pv->get_row_size(iat0) > 0 && pv->get_col_size(iat1) > 0)
                {
                    for (auto& cell : Rs)
                    {
                        const Abfs::Vector3_Order<int>& R = RI_Util::array3_to_Vector3(
                            (cell_nearest ?
                                cell_nearest->get_cell_nearest_discrete(iat0, iat1, cell)
                                : cell));
                        BaseMatrix<TR>* HlocR = hR->find_matrix(iat0, iat1, R.x, R.y, R.z);

                        if (HlocR == nullptr)
                        { // add R to HContainer
                            need_allocate = true;
                            AtomPair<TR> tmp(iat0, iat1, R.x, R.y, R.z, pv);
                            hR->insert_pair(tmp);
                        }
                    }
                }
            }
        }
        if (need_allocate) { hR->allocate(nullptr, true);}
    }

template <typename TK, typename TR>
OperatorEXX<OperatorLCAO<TK, TR>>::OperatorEXX(HS_Matrix_K<TK>* hsk_in,
    HContainer<TR>*hR_in,
    const UnitCell& ucell_in,
    const K_Vectors& kv_in,
    std::vector<std::map<int, std::map<TAC, RI::Tensor<double>>>>* Hexxd_in,
    std::vector<std::map<int, std::map<TAC, RI::Tensor<std::complex<double>>>>>* Hexxc_in,
    Add_Hexx_Type add_hexx_type_in,
    const int istep,
    int* two_level_step_in,
    const bool restart_in)
    : OperatorLCAO<TK, TR>(hsk_in, kv_in.kvec_d, hR_in),
    ucell(ucell_in),
    kv(kv_in),
    Hexxd(Hexxd_in),
    Hexxc(Hexxc_in),
    add_hexx_type(add_hexx_type_in),
    istep(istep),
    two_level_step(two_level_step_in),
    restart(restart_in)
{
    ModuleBase::TITLE("OperatorEXX", "OperatorEXX");
    this->cal_type = calculation_type::lcao_exx;
    const Parallel_Orbitals* const pv = hR_in->get_paraV();
    const bool zero_koffset =
        (ModuleBase::Vector3<double>(std::fmod(this->kv.get_koffset(0), 1.0),
                                    std::fmod(this->kv.get_koffset(1), 1.0),
                                    std::fmod(this->kv.get_koffset(2), 1.0))
             .norm()
         < 1e-10);

    if (PARAM.inp.calculation == "nscf"
        && (GlobalC::exx_info.info_global.cal_exx
            || abacus_debug_dump_exx_ao_enabled()))
    {    // if nscf, read HexxR first and reallocate hR according to the read-in HexxR
        this->use_cell_nearest = false;
        auto maybe_remap_wigner_seitz = [&](auto& Hexxs_loaded)
        {
            std::array<int, 3> Rs_period = {this->kv.nmp[0], this->kv.nmp[1], this->kv.nmp[2]};
            const bool period_from_kmesh
                = (Rs_period[0] > 0 && Rs_period[1] > 0 && Rs_period[2] > 0);
            if (!can_remap_wigner_seitz_for_nscf(
                    this->add_hexx_type,
                    period_from_kmesh,
                    zero_koffset))
            {
                return;
            }
            if (!period_from_kmesh)
            {
                const auto inferred = infer_complete_Rs_period_from_Hexxs(Hexxs_loaded);
                if (!inferred.first)
                {
                    if (GlobalV::MY_RANK == 0)
                    {
                        ModuleBase::WARNING(
                            "OperatorEXX",
                            "Cannot infer a complete BvK period; HexxR Wigner-Seitz remapping is disabled");
                    }
                    return;
                }
                Rs_period = inferred.second;
            }

            const WignerSeitzRemapStats stats
                = remap_Hexxs_wigner_seitz(ucell, Rs_period, Hexxs_loaded);
            if (GlobalV::MY_RANK == 0)
            {
                std::cout << " NSCF EXX Wigner-Seitz remapping enabled with R period: "
                          << Rs_period[0] << " " << Rs_period[1] << " " << Rs_period[2]
                          << "; input blocks=" << stats.input_blocks
                          << "; output blocks=" << stats.output_blocks
                          << "; remapped blocks=" << stats.remapped_blocks
                          << "; split blocks=" << stats.split_blocks
                          << "; max images=" << stats.max_images
                          << std::endl;
            }
        };
        auto file_name_list_csr = []() -> std::vector<std::string>
        {
            std::vector<std::string> file_name_list;
            for (int irank=0; irank<PARAM.globalv.nproc; ++irank) {
                for (int is=0;is<PARAM.inp.nspin;++is) {
                    file_name_list.push_back( PARAM.globalv.global_readin_dir + "HexxR" + std::to_string(irank) + "_" + std::to_string(is) + ".csr" );
            } }
            return file_name_list;
        };
        auto file_name_list_cereal = []() -> std::vector<std::string>
        {
            std::vector<std::string> file_name_list;
            for (int irank=0; irank<PARAM.globalv.nproc; ++irank)
                { file_name_list.push_back( "HexxR_" + std::to_string(irank) ); }
            return file_name_list;
        };
        auto check_exist = [](const std::vector<std::string> &file_name_list) -> bool
        {
            for (const std::string &file_name : file_name_list)
            {
                std::ifstream ifs(file_name);
                if (!ifs.is_open())
                    { return false; }
            }
            return true;
        };

        std::cout<<" Attention: The number of MPI processes must be strictly identical between SCF and NSCF when computing exact-exchange."<<std::endl;
        if (check_exist(file_name_list_csr()))
        {
            const std::string file_name_exx_csr = PARAM.globalv.global_readin_dir + "HexxR" + std::to_string(PARAM.globalv.myrank);
            // Read HexxR in CSR format
            if (GlobalC::exx_info.info_ri.real_number)
            {
                ModuleIO::read_Hexxs_csr(file_name_exx_csr, ucell, PARAM.inp.nspin, PARAM.globalv.nlocal, *Hexxd);
                if (this->add_hexx_type == Add_Hexx_Type::R)
                {
                    maybe_remap_wigner_seitz(*Hexxd);
                    reallocate_hcontainer(*Hexxd, this->hR);
                }
            }
            else
            {
                ModuleIO::read_Hexxs_csr(file_name_exx_csr, ucell, PARAM.inp.nspin, PARAM.globalv.nlocal, *Hexxc);
                if (this->add_hexx_type == Add_Hexx_Type::R)
                {
                    maybe_remap_wigner_seitz(*Hexxc);
                    reallocate_hcontainer(*Hexxc, this->hR);
                }
            }
        }
        else if (check_exist(file_name_list_cereal()))
        {
            // Read HexxR in binary format (old version)
            const std::string file_name_exx_cereal = PARAM.globalv.global_readin_dir + "HexxR_" + std::to_string(PARAM.globalv.myrank);
            std::ifstream ifs(file_name_exx_cereal, std::ios::binary);
            if (!ifs)
                { ModuleBase::WARNING_QUIT("OperatorEXX", "Can't open EXX file < " + file_name_exx_cereal + " >."); }
            if (GlobalC::exx_info.info_ri.real_number)
            {
                ModuleIO::read_Hexxs_cereal(file_name_exx_cereal, *Hexxd);
                if (this->add_hexx_type == Add_Hexx_Type::R)
                {
                    maybe_remap_wigner_seitz(*Hexxd);
                    reallocate_hcontainer(*Hexxd, this->hR);
                }
            }
            else
            {   
                ModuleIO::read_Hexxs_cereal(file_name_exx_cereal, *Hexxc);
                if (this->add_hexx_type == Add_Hexx_Type::R)
                {
                    maybe_remap_wigner_seitz(*Hexxc);
                    reallocate_hcontainer(*Hexxc, this->hR);
                }
            }
        }
        else
        {
            ModuleBase::WARNING_QUIT("OperatorEXX", "Can't open EXX file in " + PARAM.globalv.global_readin_dir);
        }
    }
    else
    {   // if scf and Add_Hexx_Type::R, init cell_nearest and reallocate hR according to BvK cells
        if (this->add_hexx_type == Add_Hexx_Type::R)
        {
            // if k points has no shift, use cell_nearest to reduce the memory cost
            this->use_cell_nearest = zero_koffset;

            const std::array<int, 3> Rs_period = { this->kv.nmp[0], this->kv.nmp[1], this->kv.nmp[2] };
            if (this->use_cell_nearest)
            {
                this->cell_nearest = init_cell_nearest(ucell, Rs_period);
                reallocate_hcontainer(ucell.nat, this->hR, Rs_period, &this->cell_nearest);
            }
            else { reallocate_hcontainer(ucell.nat, this->hR, Rs_period); }
        }

        if (this->restart)
        {///  Now only Hexx depends on DM, so we can directly read Hexx to reduce the computational cost.
        /// If other operators depends on DM, we can also read DM and then calculate the operators to save the memory to store operator terms.
            assert(this->two_level_step != nullptr);

            if (this->add_hexx_type == Add_Hexx_Type::k)
            {
                /// read in Hexx(k)
                if (std::is_same<TK, double>::value)
                {
                    this->Hexxd_k_load.resize(this->kv.get_nks());
                    for (int ik = 0; ik < this->kv.get_nks(); ik++)
                    {
                        this->Hexxd_k_load[ik].resize(pv->get_local_size(), 0.0);
                        this->restart = GlobalC::restart.load_disk(
                            "Hexx", ik,
                            pv->get_local_size(), this->Hexxd_k_load[ik].data(), false);
                        if (!this->restart) { break; }
                    }
                }
                else
                {
                    this->Hexxc_k_load.resize(this->kv.get_nks());
                    for (int ik = 0; ik < this->kv.get_nks(); ik++)
                    {
                        this->Hexxc_k_load[ik].resize(pv->get_local_size(), 0.0);
                        this->restart = GlobalC::restart.load_disk(
                            "Hexx", ik,
                            pv->get_local_size(), this->Hexxc_k_load[ik].data(), false);
                        if (!this->restart) { break; }
                    }
                }
            }
            else if (this->add_hexx_type == Add_Hexx_Type::R)
            {
                // read in Hexx(R)
                const std::string restart_HR_path = GlobalC::restart.folder + "HexxR" + std::to_string(PARAM.globalv.myrank);
                int all_exist = 1;
                for (int is = 0; is < PARAM.inp.nspin; ++is)
                {
                    std::ifstream ifs(restart_HR_path + "_" + std::to_string(is) + ".csr");
                    if (!ifs) { all_exist = 0; break; }
                }
// Add MPI communication to synchronize all_exist across processes
                #ifdef __MPI
                Parallel_Reduce::reduce_min(all_exist);
                #endif
                if (all_exist)
                {
                    // Read HexxR in CSR format
                    if (GlobalC::exx_info.info_ri.real_number) {
                        ModuleIO::read_Hexxs_csr(restart_HR_path, ucell, PARAM.inp.nspin, PARAM.globalv.nlocal, *Hexxd);
                    }
                    else {
                        ModuleIO::read_Hexxs_csr(restart_HR_path, ucell, PARAM.inp.nspin, PARAM.globalv.nlocal, *Hexxc);
                    }
                }
                else
                {
                    // Read HexxR in binary format (old version)
                    const std::string restart_HR_path_cereal = GlobalC::restart.folder + "HexxR_" + std::to_string(PARAM.globalv.myrank);
                    std::ifstream ifs(restart_HR_path_cereal, std::ios::binary);
                    int all_exist_cereal = ifs ? 1 : 0;
                    #ifdef __MPI
                    Parallel_Reduce::reduce_min(all_exist_cereal);
                    #endif
                    if (!all_exist_cereal)
                    {
                        //no HexxR file in CSR or binary format
                        this->restart = false;
                    }
                    else
                    {
                        if (GlobalC::exx_info.info_ri.real_number) {
                            ModuleIO::read_Hexxs_cereal(restart_HR_path_cereal, *Hexxd);
                        }
                        else {
                            ModuleIO::read_Hexxs_cereal(restart_HR_path_cereal, *Hexxc);
                        }
                    }
                }
            }

            if (!this->restart) {
                std::cout << "WARNING: Hexx not found, restart from the non-exx loop." << std::endl
                    << "If the loaded charge density is EXX-solved, this may lead to poor convergence." << std::endl;
            }
            GlobalC::restart.info_load.load_H_finish = this->restart;
        }
    }
}

template<typename TK, typename TR>
void OperatorEXX<OperatorLCAO<TK, TR>>::contributeHR()
{
    ModuleBase::TITLE("OperatorEXX", "contributeHR");
    // Peize Lin add 2016-12-03
    if (this->istep == 0
        && PARAM.inp.calculation != "nscf"
        && this->two_level_step != nullptr && *this->two_level_step == 0
        && PARAM.inp.init_wfc != "file"
        && !this->restart)
    {
        return;
    }  //in the non-exx loop, do nothing 
    if (this->add_hexx_type == Add_Hexx_Type::k) { return; }

    if (XC_Functional::get_func_type() == 4 || XC_Functional::get_func_type() == 5)
    {
        // add H(R) normally
        if (GlobalC::exx_info.info_ri.real_number)
        {
            RI_2D_Comm::add_HexxR(
                this->current_spin,
                GlobalC::exx_info.info_global.hybrid_alpha,
                *this->Hexxd,
                *this->hR->get_paraV(),
                PARAM.globalv.npol,
                *this->hR,
                this->use_cell_nearest ? &this->cell_nearest : nullptr);
        }
        else
        {
            RI_2D_Comm::add_HexxR(
                this->current_spin,
                GlobalC::exx_info.info_global.hybrid_alpha,
                *this->Hexxc,
                *this->hR->get_paraV(),
                PARAM.globalv.npol,
                *this->hR,
                this->use_cell_nearest ? &this->cell_nearest : nullptr);
        }
    }
    if (PARAM.inp.nspin == 2) { this->current_spin = 1 - this->current_spin; }
}

template<typename TK, typename TR>
void OperatorEXX<OperatorLCAO<TK, TR>>::contributeHk(int ik)
{
    ModuleBase::TITLE("OperatorEXX", "constributeHk");
    // Peize Lin add 2016-12-03
    if (PARAM.inp.calculation != "nscf" && this->two_level_step != nullptr && *this->two_level_step == 0 && !this->restart) { return; }  //in the non-exx loop, do nothing 

    if (this->add_hexx_type == Add_Hexx_Type::R) { throw std::invalid_argument("Set Add_Hexx_Type::k sto call OperatorEXX::contributeHk()."); }

    const bool enable_debug_loaded_hexx =
        (PARAM.inp.calculation == "nscf" && abacus_debug_dump_exx_ao_enabled());
    if (XC_Functional::get_func_type() == 4 || XC_Functional::get_func_type() == 5
        || enable_debug_loaded_hexx)
    {
        if (abacus_debug_dump_exx_ao_enabled() && GlobalV::MY_RANK == 0)
        {
            std::cout << "[ABACUS_DUMP_EXX_AO] OperatorEXX::contributeHk entered for ik="
                      << ik << ", restart=" << this->restart
                      << ", two_level_step="
                      << ((this->two_level_step != nullptr) ? *this->two_level_step : -1)
                      << ", func_type=" << XC_Functional::get_func_type()
                      << ", debug_loaded_hexx=" << enable_debug_loaded_hexx << std::endl;
        }
        if (this->restart && this->two_level_step != nullptr)
        {
            if (*this->two_level_step == 0)
            {
                this->add_loaded_Hexx(ik);
                return;
            }
            else // clear loaded Hexx and release memory
            {
                if (this->Hexxd_k_load.size() > 0)
                {
                    this->Hexxd_k_load.clear();
                    this->Hexxd_k_load.shrink_to_fit();
                }
                else if (this->Hexxc_k_load.size() > 0)
                {
                    this->Hexxc_k_load.clear();
                    this->Hexxc_k_load.shrink_to_fit();
                }
            }
        }
        std::vector<TK> hk_before;
        const Parallel_Orbitals* pv = this->hR->get_paraV();
        if (abacus_debug_dump_exx_ao_enabled() && GlobalV::NPROC == 1 && pv != nullptr)
        {
            const int nrow = pv->get_row_size();
            const int ncol = pv->get_col_size();
            if (nrow > 0 && ncol > 0)
            {
                hk_before.assign(this->hsk->get_hk(), this->hsk->get_hk() + nrow * ncol);
            }
        }
        // cal H(k) from H(R) normally
        if(PARAM.inp.esolver_type == "tddft" && PARAM.inp.td_stype == 2)
        {
            RI_2D_Comm::add_Hexx_td(
                ucell,
                this->kv,
                ik,
                GlobalC::exx_info.info_global.hybrid_alpha,
                *this->Hexxc,
                *this->hR->get_paraV(),
                TD_info::td_vel_op->cart_At,
                this->hsk->get_hk());
        }
        else
        {
            if (GlobalC::exx_info.info_ri.real_number) {
                RI_2D_Comm::add_Hexx(
                    ucell,
                    this->kv,
                    ik,
                    GlobalC::exx_info.info_global.hybrid_alpha,
                    *this->Hexxd,
                    *this->hR->get_paraV(),
                    this->hsk->get_hk());
            } else {
                RI_2D_Comm::add_Hexx(
                    ucell,
                    this->kv,
                    ik,
                    GlobalC::exx_info.info_global.hybrid_alpha,
                    *this->Hexxc,
                    *this->hR->get_paraV(),
                    this->hsk->get_hk());
            }
        }
        if (!hk_before.empty())
        {
            abacus_debug_dump_exx_ao_delta_if_requested(
                this->kv, ik, pv, hk_before.data(), this->hsk->get_hk());
        }
    }
}

} // namespace hamilt
#endif // __EXX
#endif // OPEXXLCAO_HPP
