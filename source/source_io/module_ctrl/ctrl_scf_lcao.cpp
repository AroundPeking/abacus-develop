#include "ctrl_scf_lcao.h" // use ctrl_scf_lcao()

#include "source_estate/elecstate_lcao.h" // use elecstate::ElecState
#include "source_hamilt/hamilt.h"         // use Hamilt<T>
#include "source_lcao/hamilt_lcao.h"      // use hamilt::HamiltLCAO<TK, TR>

#include <cmath>
#include <complex>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

// functions
#include "../module_unk/berryphase.h"                          // use berryphase
#include "../module_hs/cal_pLpR.h"                            // use AngularMomentumCalculator()
#include "source_io/module_hs/output_mat_sparse.h"                   // use ModuleIO::output_mat_sparse()
#include "../module_mulliken/output_mulliken.h"                     // use cal_mag()
#include "../module_wannier/to_wannier90_lcao.h"                   // use toWannier90_LCAO
#include "../module_wannier/to_wannier90_lcao_in_pw.h"             // use toWannier90_LCAO_IN_PW
#include "../module_hs/write_HS.h"                            // use ModuleIO::write_hsk()
#include "../module_dm/write_dmk.h"                           // use ModuleIO::write_dmk()
#include "../module_dm/write_dmr.h"                           // use ModuleIO::write_dmr()
#include "../module_dos/write_dos_lcao.h"                      // use ModuleIO::write_dos_lcao()
#include "../module_wf/write_wfc_nao.h"                       // use ModuleIO::write_wfc_nao()
#include "source_lcao/module_deltaspin/spin_constrain.h"   // use spinconstrain::SpinConstrain<TK>
#include "source_lcao/module_operator_lcao/ekinetic.h" // use hamilt::EKinetic
#ifdef __MLALGO
#include "source_lcao/module_deepks/LCAO_deepks.h"
#include "source_lcao/module_deepks/LCAO_deepks_interface.h"
#endif
#ifdef __EXX
#include "source_lcao/module_ri/Exx_LRI_interface.h" // use EXX codes
#include "source_lcao/module_ri/RPA_LRI.h"           // use RPA code
#include "source_lcao/module_ri/RI_Util.h"
#include "source_lcao/module_ri/module_exx_symmetry/symmetry_rotation.h"
#include "source_lcao/module_ri/sternheimer_abacus_st_smoke.h"
#endif
#include "../module_qo/to_qo.h"                // use toQO
#include "source_lcao/module_rdmft/rdmft.h" // use RDMFT codes
#include "source_lcao/rho_tau_lcao.h"       // mohan add 2025-10-24
#include "source_lcao/module_operator_lcao/overlap.h" // use hamilt::Overlap for NAMD

#ifdef __EXX
namespace
{

template <typename TK>
std::vector<ModuleRI::SternheimerLCAOOccupiedKPoint> gather_sternheimer_lcao_occupied_kpoints(
    const elecstate::ElecState& elec_state,
    const K_Vectors& kv,
    const UnitCell& ucell,
    const Parallel_Orbitals& parallel_orbitals,
    const psi::Psi<TK>& psi)
{
    if (kv.get_nks() != kv.get_nkstot())
    {
        throw std::runtime_error(
            "Sternheimer solid LCAO coefficient gathering currently requires KPAR=1.");
    }
    if (elec_state.wg.nr != kv.get_nks()
        || elec_state.ekb.nr != kv.get_nks()
        || psi.get_nk() != kv.get_nks()
        || kv.ik2iktot.size() != static_cast<std::size_t>(kv.get_nks())
        || kv.kvec_d.size() != static_cast<std::size_t>(kv.get_nks())
        || kv.wk.size() != static_cast<std::size_t>(kv.get_nks())
        || kv.isk.size() != static_cast<std::size_t>(kv.get_nks()))
    {
        throw std::runtime_error("Sternheimer solid LCAO k-point metadata are incomplete.");
    }

    std::vector<ModuleRI::SternheimerLCAOOccupiedKPoint> records;
    records.reserve(static_cast<std::size_t>(kv.get_nks()));
    for (int local_k_index = 0; local_k_index != kv.get_nks(); ++local_k_index)
    {
        if (!std::isfinite(kv.wk[local_k_index]) || kv.wk[local_k_index] <= 0.0)
        {
            throw std::runtime_error("Sternheimer solid LCAO requires positive k-point weights.");
        }
        int occupied_count = 0;
        for (int ib = 0; ib != elec_state.wg.nc; ++ib)
        {
            if (elec_state.wg(local_k_index, ib) / kv.wk[local_k_index] > 1.0e-8)
            {
                occupied_count = ib + 1;
            }
        }
        if (occupied_count == 0)
        {
            continue;
        }

        ModuleRI::SternheimerLCAOOccupiedKPoint record;
        record.local_k_index = local_k_index;
        record.global_k_index = kv.ik2iktot[static_cast<std::size_t>(local_k_index)];
        record.zero_order_k_index = local_k_index;
        record.spin_index = kv.isk[static_cast<std::size_t>(local_k_index)];
        record.kpoint = {kv.kvec_d[static_cast<std::size_t>(local_k_index)].x,
                         kv.kvec_d[static_cast<std::size_t>(local_k_index)].y,
                         kv.kvec_d[static_cast<std::size_t>(local_k_index)].z};
        // ABACUS normalizes sum(wk) to the spin degeneracy, so wg/wk is the band filling, not another spin factor.
        record.kweight = kv.wk[static_cast<std::size_t>(local_k_index)];
        record.eigenvalues.reserve(static_cast<std::size_t>(occupied_count));
        record.occupations.reserve(static_cast<std::size_t>(occupied_count));
        record.coefficients.assign(
            static_cast<std::size_t>(occupied_count),
            std::vector<std::complex<double>>(static_cast<std::size_t>(PARAM.globalv.nlocal),
                                              std::complex<double>(0.0, 0.0)));
        for (int ib = 0; ib != occupied_count; ++ib)
        {
            record.eigenvalues.push_back(elec_state.ekb(local_k_index, ib));
            record.occupations.push_back(elec_state.wg(local_k_index, ib) / record.kweight);
            const int local_band = parallel_orbitals.global2local_col(ib);
            if (local_band >= 0)
            {
                for (int local_basis = 0; local_basis != psi.get_nbasis(); ++local_basis)
                {
                    const int global_basis = parallel_orbitals.local2global_row(local_basis);
                    if (global_basis < 0 || global_basis >= PARAM.globalv.nlocal)
                    {
                        throw std::runtime_error("Sternheimer solid LCAO global basis index is out of range.");
                    }
                    record.coefficients[static_cast<std::size_t>(ib)][static_cast<std::size_t>(global_basis)]
                        = std::complex<double>(psi(local_k_index, local_band, local_basis));
                }
            }
#ifdef __MPI
            MPI_Allreduce(MPI_IN_PLACE,
                          record.coefficients[static_cast<std::size_t>(ib)].data(),
                          PARAM.globalv.nlocal,
                          MPI_DOUBLE_COMPLEX,
                          MPI_SUM,
                          MPI_COMM_WORLD);
#endif
        }
        if (ModuleRI::sternheimer_lcao_virtual_state_gathering_enabled())
        {
            const int unoccupied_count = elec_state.ekb.nc - occupied_count;
            record.unoccupied_eigenvalues.reserve(static_cast<std::size_t>(unoccupied_count));
            record.unoccupied_coefficients.assign(
                static_cast<std::size_t>(unoccupied_count),
                std::vector<std::complex<double>>(static_cast<std::size_t>(PARAM.globalv.nlocal),
                                                  std::complex<double>(0.0, 0.0)));
            for (int ib = occupied_count; ib != elec_state.ekb.nc; ++ib)
            {
                const std::size_t virtual_index = static_cast<std::size_t>(ib - occupied_count);
                record.unoccupied_eigenvalues.push_back(elec_state.ekb(local_k_index, ib));
                const int local_band = parallel_orbitals.global2local_col(ib);
                if (local_band >= 0)
                {
                    for (int local_basis = 0; local_basis != psi.get_nbasis(); ++local_basis)
                    {
                        const int global_basis = parallel_orbitals.local2global_row(local_basis);
                        if (global_basis < 0 || global_basis >= PARAM.globalv.nlocal)
                        {
                            throw std::runtime_error(
                                "Sternheimer solid LCAO global unoccupied basis index is out of range.");
                        }
                        record.unoccupied_coefficients[virtual_index][static_cast<std::size_t>(global_basis)]
                            = std::complex<double>(psi(local_k_index, local_band, local_basis));
                    }
                }
#ifdef __MPI
                MPI_Allreduce(MPI_IN_PLACE,
                              record.unoccupied_coefficients[virtual_index].data(),
                              PARAM.globalv.nlocal,
                              MPI_DOUBLE_COMPLEX,
                              MPI_SUM,
                              MPI_COMM_WORLD);
#endif
            }
        }
        records.push_back(std::move(record));
    }
    ModuleRI::validate_sternheimer_lcao_occupied_kpoints(
        records, kv.get_nks(), kv.get_nkstot(), kv.get_nspin(), PARAM.globalv.nlocal);

    const int ibz_kpoint_count = kv.get_nkstot();
    const int full_kpoint_count = kv.get_nkstot_full();
    if (!ModuleRI::sternheimer_full_k_reconstruction_required(
            ibz_kpoint_count,
            full_kpoint_count,
            PARAM.inp.nspin,
            ModuleSymmetry::Symmetry::symm_flag))
    {
        return records;
    }
    if (full_kpoint_count <= 0
        || kv.kstars.size() != static_cast<std::size_t>(ibz_kpoint_count)
        || kv.ibz_index.size() != static_cast<std::size_t>(full_kpoint_count)
        || kv.kvec_c_full.size() != static_cast<std::size_t>(full_kpoint_count))
    {
        throw std::runtime_error("Sternheimer full-k symmetry metadata are incomplete.");
    }

    std::vector<int> record_by_ibz(static_cast<std::size_t>(ibz_kpoint_count), -1);
    for (std::size_t record_index = 0; record_index != records.size(); ++record_index)
    {
        const int ibz_index = records[record_index].global_k_index;
        if (ibz_index < 0 || ibz_index >= ibz_kpoint_count
            || record_by_ibz[static_cast<std::size_t>(ibz_index)] >= 0)
        {
            throw std::runtime_error("Sternheimer IBZ record map is inconsistent.");
        }
        record_by_ibz[static_cast<std::size_t>(ibz_index)] = static_cast<int>(record_index);
    }

    ModuleSymmetry::Symmetry_rotation symmetry_rotation;
    const std::array<int, 3>& period = RI_Util::get_Born_vonKarmen_period(kv);
    symmetry_rotation.find_irreducible_sector(ucell.symm,
                                               ucell.atoms,
                                               ucell.st,
                                               RI_Util::get_Born_von_Karmen_cells(period),
                                               period,
                                               ucell.lat);
    symmetry_rotation.cal_Ms(kv, ucell, parallel_orbitals);

    auto restrict_kpoint = [&ucell](ModuleBase::Vector3<double> kpoint) {
        const double epsilon = ucell.symm.epsilon;
        kpoint.x = std::fmod(kpoint.x + 100.5 - 0.5 * epsilon, 1.0) - 0.5 + 0.5 * epsilon;
        kpoint.y = std::fmod(kpoint.y + 100.5 - 0.5 * epsilon, 1.0) - 0.5 + 0.5 * epsilon;
        kpoint.z = std::fmod(kpoint.z + 100.5 - 0.5 * epsilon, 1.0) - 0.5 + 0.5 * epsilon;
        if (std::abs(kpoint.x) < epsilon)
        {
            kpoint.x = 0.0;
        }
        if (std::abs(kpoint.y) < epsilon)
        {
            kpoint.y = 0.0;
        }
        if (std::abs(kpoint.z) < epsilon)
        {
            kpoint.z = 0.0;
        }
        return kpoint;
    };
    auto same_kpoint = [&ucell](const ModuleBase::Vector3<double>& lhs,
                                const ModuleBase::Vector3<double>& rhs) {
        return std::abs(lhs.x - rhs.x) < ucell.symm.epsilon
               && std::abs(lhs.y - rhs.y) < ucell.symm.epsilon
               && std::abs(lhs.z - rhs.z) < ucell.symm.epsilon;
    };
    auto conjugate_coefficients = [](std::vector<std::vector<std::complex<double>>> coefficients) {
        for (auto& band: coefficients)
        {
            for (auto& coefficient: band)
            {
                coefficient = std::conj(coefficient);
            }
        }
        return coefficients;
    };

    std::vector<ModuleRI::SternheimerLCAOOccupiedKPoint> full_records;
    full_records.reserve(static_cast<std::size_t>(full_kpoint_count));
    const double full_kweight = 2.0 / static_cast<double>(full_kpoint_count);
    const ModuleBase::Matrix3 reciprocal_inverse = ucell.G.Inverse();
    for (int full_k_index = 0; full_k_index != full_kpoint_count; ++full_k_index)
    {
        const int ibz_index = kv.ibz_index[static_cast<std::size_t>(full_k_index)];
        if (ibz_index < 0 || ibz_index >= ibz_kpoint_count
            || record_by_ibz[static_cast<std::size_t>(ibz_index)] < 0)
        {
            throw std::runtime_error("Sternheimer full-k point has no IBZ source record.");
        }
        const ModuleBase::Vector3<double> full_kpoint
            = restrict_kpoint(kv.kvec_c_full[static_cast<std::size_t>(full_k_index)]
                              * reciprocal_inverse);
        int combined_isym = -1;
        for (const auto& star_member: kv.kstars[static_cast<std::size_t>(ibz_index)])
        {
            if (same_kpoint(restrict_kpoint(star_member.second), full_kpoint))
            {
                combined_isym = star_member.first;
                break;
            }
        }
        if (combined_isym < 0)
        {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "Sternheimer full-k point is absent from its IBZ k-star: full_index="
                    << full_k_index << " ibz_index=" << ibz_index << " full_direct=("
                    << full_kpoint.x << "," << full_kpoint.y << "," << full_kpoint.z
                    << ") full_cartesian=("
                    << kv.kvec_c_full[static_cast<std::size_t>(full_k_index)].x << ","
                    << kv.kvec_c_full[static_cast<std::size_t>(full_k_index)].y << ","
                    << kv.kvec_c_full[static_cast<std::size_t>(full_k_index)].z << ") star=";
            for (const auto& star_member: kv.kstars[static_cast<std::size_t>(ibz_index)])
            {
                const auto member = restrict_kpoint(star_member.second);
                message << " [isym=" << star_member.first << " k=(" << member.x << ","
                        << member.y << "," << member.z << ")]";
            }
            throw std::runtime_error(message.str());
        }

        const auto& ibz_record
            = records[static_cast<std::size_t>(record_by_ibz[static_cast<std::size_t>(ibz_index)])];
        const int spatial_isym = combined_isym % ucell.symm.nrotk;
        const bool time_reversal = combined_isym >= ucell.symm.nrotk;
        auto rotate_coefficients = [&](const std::vector<std::vector<std::complex<double>>>& coefficients) {
            if (coefficients.empty())
            {
                return coefficients;
            }
            if (spatial_isym == 0)
            {
                return time_reversal ? conjugate_coefficients(coefficients) : coefficients;
            }
            return symmetry_rotation.rotate_ao_coefficients(
                coefficients, ibz_index, spatial_isym, parallel_orbitals, time_reversal);
        };
        auto full_record = ModuleRI::make_sternheimer_full_kpoint_record(
            ibz_record,
            full_k_index,
            {full_kpoint.x, full_kpoint.y, full_kpoint.z},
            full_kweight,
            rotate_coefficients(ibz_record.coefficients),
            rotate_coefficients(ibz_record.unoccupied_coefficients));
        full_record.symmetry_spatial_isym = spatial_isym;
        full_record.symmetry_time_reversal = time_reversal;
        full_records.push_back(std::move(full_record));
    }
    ModuleRI::validate_sternheimer_lcao_occupied_kpoints(full_records,
                                                         full_kpoint_count,
                                                         full_kpoint_count,
                                                         1,
                                                         PARAM.globalv.nlocal,
                                                         ibz_kpoint_count);
    return full_records;
}

} // namespace
#endif

template <typename TK, typename TR>
void ModuleIO::ctrl_scf_lcao(UnitCell& ucell,
                             const Input_para& inp,
                             K_Vectors& kv,
                             elecstate::ElecState* pelec,
                             elecstate::DensityMatrix<TK, double>* dm, // mohan add 2025-11-04
                             Parallel_Orbitals& pv,
                             Grid_Driver& gd,
                             psi::Psi<TK>* psi,
                             hamilt::HamiltLCAO<TK, TR>* p_hamilt,
                             Plus_U& dftu, // mohan add 2025-11-07
                             TwoCenterBundle& two_center_bundle,
                             LCAO_Orbitals& orb,
                             const ModulePW::PW_Basis_K* pw_wfc,   // for berryphase
                             const ModulePW::PW_Basis* pw_rho,     // for berryphase
                             const ModulePW::PW_Basis_Big* pw_big, // for Wannier90
                             const Structure_Factor& sf,           // for Wannier90
                             rdmft::RDMFT<TK, TR>& rdmft_solver,   // for RDMFT
                             Setup_DeePKS<TK>& deepks,
                             Exx_NAO<TK>& exx_nao,
                             const bool conv_esolver,
                             const bool scf_nmax_flag,
                             const int istep)
{
    ModuleBase::TITLE("ModuleIO", "ctrl_scf_lcao");
    ModuleBase::timer::tick("ModuleIO", "ctrl_scf_lcao");

    //*****
    // if istep_in = -1, istep will not appear in file name
    // if iter_in = -1, iter will not appear in file name
    int istep_in = -1;
    int iter_in = -1;
    bool out_flag = false;
    if (PARAM.inp.esolver_type != "tddft" && inp.out_freq_ion > 0) // default value of out_freq_ion is 0
    {
        if (istep % inp.out_freq_ion == 0)
        {
            istep_in = istep;
            out_flag = true;
        }
    }
    else if (PARAM.inp.esolver_type == "tddft" && inp.out_freq_td > 0) // default value of out_freq_td is 0
    {
        if (istep % inp.out_freq_td == 0)
        {
            istep_in = istep;
            out_flag = true;
        }
    }
    else if (conv_esolver || scf_nmax_flag) // mohan add scf_nmax_flag on 20250921
    {
        out_flag = true;
    }

    if (!out_flag)
    {
        return;
    }

    //*****

    const bool out_app_flag = inp.out_app_flag;
    const bool gamma_only = PARAM.globalv.gamma_only_local;
    const int nspin = inp.nspin;
    const std::string global_out_dir = PARAM.globalv.global_out_dir;

    //------------------------------------------------------------------
    //! 1) print out density of states (DOS)
    //------------------------------------------------------------------
    if (inp.out_dos)
    {
        ModuleIO::write_dos_lcao(psi,
                                 p_hamilt,
                                 pv,
                                 ucell,
                                 kv,
                                 inp.nbands,
                                 pelec->eferm,
                                 pelec->ekb,
                                 pelec->wg,
                                 inp.dos_edelta_ev,
                                 inp.dos_scale,
                                 inp.dos_sigma,
                                 out_app_flag,
                                 istep,
                                 GlobalV::ofs_running);
    }

    //------------------------------------------------------------------
    //! 2) Output density matrix DM(R)
    //------------------------------------------------------------------
    if (inp.out_dmr[0])
    {
        const int precision = inp.out_dmr[1];

        ModuleIO::write_dmr(dm->get_DMR_vector(), &ucell, precision, pv, out_app_flag, 
			ucell.get_iat2iwt(), ucell.nat, istep);
    }

    //------------------------------------------------------------------
    //! 3) Output density matrix DM(k)
    //------------------------------------------------------------------
    if (inp.out_dmk[0])
    {
        std::vector<double> efermis(nspin == 2 ? 2 : 1);
        for (int ispin = 0; ispin < efermis.size(); ispin++)
        {
            efermis[ispin] = pelec->eferm.get_efval(ispin);
        }
        const int precision = inp.out_dmk[1];

        ModuleIO::write_dmk(dm->get_DMK_vector(), kv, precision, efermis, &(ucell), pv, istep);
    }

    //------------------------------------------------------------------
    // 4) Output H(k) and S(k) matrices for each k-point
    //------------------------------------------------------------------
    if (inp.out_mat_hs[0])
    {
        ModuleIO::write_hsk(global_out_dir,
                            nspin,
                            kv.get_nks(),
                            kv.get_nkstot(),
                            kv.ik2iktot,
                            kv.isk,
                            p_hamilt,
                            pv,
                            gamma_only,
                            out_app_flag,
                            istep,
                            GlobalV::ofs_running);
    }

    //------------------------------------------------------------------
    //! 5) Output electronic wavefunctions Psi(k)
    //------------------------------------------------------------------
    if (elecstate::ElecStateLCAO<TK>::out_wfc_lcao)
    {
        ModuleIO::write_wfc_nao(elecstate::ElecStateLCAO<TK>::out_wfc_lcao,
                                out_app_flag,
                                psi[0],
                                pelec->ekb,
                                pelec->wg,
                                kv.kvec_c,
                                kv.ik2iktot,
                                kv.get_nkstot(),
                                pv,
                                nspin,
                                istep);
    }

    //------------------------------------------------------------------
    //! 6) Output DeePKS information
    //------------------------------------------------------------------
#ifdef __MLALGO
    // need control parameter
    hamilt::HamiltLCAO<TK, TR>* p_ham_deepks = p_hamilt;
    std::shared_ptr<LCAO_Deepks<TK>> ld_shared_ptr(&deepks.ld, [](LCAO_Deepks<TK>*) {});
    LCAO_Deepks_Interface<TK, TR> deepks_interface(ld_shared_ptr);

    deepks_interface.out_deepks_labels(pelec->f_en.etot,
                                       kv.get_nks(),
                                       ucell.nat,
                                       PARAM.globalv.nlocal,
                                       pelec->ekb,
                                       kv.kvec_d,
                                       ucell,
                                       orb,
                                       gd,
                                       &pv,
                                       *psi,
                                       dm,
                                       p_ham_deepks,
                                       -1,   // -1 when called in after scf
                                       true, // no used when after scf
                                       GlobalV::MY_RANK,
                                       GlobalV::ofs_running);
#endif

    //------------------------------------------------------------------
    //! 7) Output <phi_i|O|phi_j> matrices, where O can be chosen as
    //!    H, S, dH, dS, T, r. The format is CSR format.
    //------------------------------------------------------------------
    hamilt::Hamilt<TK>* p_ham_tk = static_cast<hamilt::Hamilt<TK>*>(p_hamilt);

    ModuleIO::output_mat_sparse(inp.out_mat_hs2,
                                inp.out_mat_dh,
                                inp.out_mat_ds,
                                inp.out_mat_t,
                                inp.out_mat_r,
                                istep,
                                pelec->pot->get_eff_v(),
                                pv,
                                two_center_bundle,
                                orb,
                                ucell,
                                gd,
                                kv,
                                p_ham_tk,
                                &dftu);

    //------------------------------------------------------------------
    //! 8) Output kinetic matrix
    //------------------------------------------------------------------
    if (inp.out_mat_tk[0])
    {
        hamilt::HS_Matrix_K<TK> hsk(&pv, true);
        hamilt::HContainer<TR> hR(&pv);
        hamilt::Operator<TK>* ekinetic
            = new hamilt::EKinetic<hamilt::OperatorLCAO<TK, TR>>(&hsk,
                                                                    kv.kvec_d,
                                                                    &hR,
                                                                    &ucell,
                                                                    orb.cutoffs(),
                                                                    &gd,
                                                                    two_center_bundle.kinetic_orb.get());

        const int nspin_k = (nspin == 2 ? 2 : 1);
        for (int ik = 0; ik < kv.get_nks() / nspin_k; ++ik)
        {
            ekinetic->init(ik);

            const int out_label = 1; // 1: .txt, 2: .dat

            std::string t_fn = ModuleIO::filename_output(global_out_dir,
                                                         "tk",
                                                         "nao",
                                                         ik,
                                                         kv.ik2iktot,
                                                         inp.nspin,
                                                         kv.get_nkstot(),
                                                         out_label,
                                                         out_app_flag,
                                                         gamma_only,
                                                         istep);

            ModuleIO::save_mat(istep,
                               hsk.get_hk(),
                               PARAM.globalv.nlocal,
                               false, // bit
                               inp.out_mat_tk[1],
                               1, // true for upper triangle matrix
                               inp.out_app_flag,
                               t_fn,
                               pv,
                               GlobalV::DRANK);
        }

        delete ekinetic;
    }

    //------------------------------------------------------------------
    //! 9) Output expectation of angular momentum operator
    //------------------------------------------------------------------
    if (inp.out_mat_l[0])
    {
        ModuleIO::AngularMomentumCalculator mylcalculator(inp.orbital_dir,
                                                          ucell,
                                                          orb.get_rcutmax_Phi(),
                                                          inp.test_deconstructor,
                                                          inp.test_grid,
                                                          inp.test_atom_input,
                                                          PARAM.globalv.search_pbc,
                                                          &GlobalV::ofs_running,
                                                          GlobalV::MY_RANK);
        mylcalculator.calculate(inp.suffix, global_out_dir, ucell, inp.out_mat_l[1], GlobalV::MY_RANK);
    }

    //------------------------------------------------------------------
    //! 10) Output Mulliken charge
    //------------------------------------------------------------------
    if (inp.out_mul)
    {
        ModuleIO::cal_mag(&pv,
                          p_hamilt,
                          kv,
                          dm, // mohan add 2025-11-04
                          two_center_bundle,
                          orb,
                          ucell,
                          gd,
                          istep,
                          true);
    }

    //------------------------------------------------------------------
    //! 11) Output atomic magnetization by using 'spin_constraint'
    //------------------------------------------------------------------
    if (inp.sc_mag_switch)
    {
        spinconstrain::SpinConstrain<TK>& sc = spinconstrain::SpinConstrain<TK>::getScInstance();
        sc.cal_mi_lcao(istep);
        sc.print_Mi(GlobalV::ofs_running);
        sc.print_Mag_Force(GlobalV::ofs_running);
    }

    //------------------------------------------------------------------
    //! 12) Output Berry phase
    //------------------------------------------------------------------
    if (inp.calculation == "nscf" && berryphase::berry_phase_flag && ModuleSymmetry::Symmetry::symm_flag != 1)
    {
        std::cout << FmtCore::format("\n * * * * * *\n << Start %s.\n", "Berry phase calculation");
        berryphase bp(&pv);
        bp.lcao_init(ucell, gd, kv, orb);
        // additional step before calling macroscopic_polarization
        bp.Macroscopic_polarization(ucell, pw_wfc->npwk_max, psi, pw_rho, pw_wfc, kv);
        std::cout << FmtCore::format(" >> Finish %s.\n * * * * * *\n", "Berry phase calculation");
    }

    //------------------------------------------------------------------
    //! 13) Wannier90 interface in LCAO basis
    // added by jingan in 2018.11.7
    //------------------------------------------------------------------
    if (inp.calculation == "nscf" && inp.towannier90)
    {
        std::cout << FmtCore::format("\n * * * * * *\n << Start %s.\n", "Wave function to Wannier90");
        if (inp.wannier_method == 1)
        {
            toWannier90_LCAO_IN_PW wan(inp.out_wannier_mmn,
                                       inp.out_wannier_amn,
                                       inp.out_wannier_unk,
                                       inp.out_wannier_eig,
                                       inp.out_wannier_wvfn_formatted,
                                       inp.nnkpfile,
                                       inp.wannier_spin);
            wan.set_tpiba_omega(ucell.tpiba, ucell.omega);
            wan.calculate(ucell, pelec->ekb, pw_wfc, pw_big, sf, kv, psi, &pv);
        }
        else if (inp.wannier_method == 2)
        {
            toWannier90_LCAO wan(inp.out_wannier_mmn,
                                 inp.out_wannier_amn,
                                 inp.out_wannier_unk,
                                 inp.out_wannier_eig,
                                 inp.out_wannier_wvfn_formatted,
                                 inp.nnkpfile,
                                 inp.wannier_spin,
                                 orb);

            wan.calculate(ucell, gd, pelec->ekb, kv, *psi, &pv);
        }
        std::cout << FmtCore::format(" >> Finish %s.\n * * * * * *\n", "Wave function to Wannier90");
    }

    // 14) calculate the kinetic energy density tau
    // mohan add 2025-10-24
    //    if (inp.out_elf[0] > 0)
    //	{
    //		LCAO_domain::dm2tau(pelec->DM->get_DMR_vector(), inp.nspin, pelec->charge);
    //	}

#ifdef __EXX
    //------------------------------------------------------------------
    //! 15) Output Hexx matrix in LCAO basis
    // (see `out_chg` in docs/advanced/input_files/input-main.md)
    //------------------------------------------------------------------
    if (inp.out_chg[0])
    {
        if (GlobalC::exx_info.info_global.cal_exx && inp.calculation != "nscf") // Peize Lin add if 2022.11.14
        {
            const std::string file_name_exx = global_out_dir + "HexxR" + std::to_string(GlobalV::MY_RANK);
            if (GlobalC::exx_info.info_ri.real_number)
            {
                ModuleIO::write_Hexxs_csr(file_name_exx, ucell, exx_nao.exd->get_Hexxs());
            }
            else
            {
                ModuleIO::write_Hexxs_csr(file_name_exx, ucell, exx_nao.exc->get_Hexxs());
            }
        }
    }

    //------------------------------------------------------------------
    //! 16) Write RPA information in LCAO basis
    //------------------------------------------------------------------
    if (inp.rpa)
    {
        RPA_LRI<TK, double> rpa_lri_double(GlobalC::exx_info.info_ri);
        rpa_lri_double.postSCF(ucell, MPI_COMM_WORLD, *dm, pelec, kv, orb, pv, *psi);
    }

    if (inp.out_sternheimer_librpa || inp.out_sternheimer_siab)
    {
        if (pelec == nullptr || pelec->pot == nullptr || pw_rho == nullptr || psi == nullptr)
        {
            ModuleBase::WARNING_QUIT("ctrl_scf_lcao", "Sternheimer LCAO output requires potential, grid, and KS states.");
        }
        const auto occupied_kpoints
            = gather_sternheimer_lcao_occupied_kpoints(*pelec, kv, ucell, pv, *psi);
        ModuleRI::run_sternheimer_abacus_lcao_chi0_output(*(pelec->pot),
                                                          *pw_rho,
                                                          ucell,
                                                          *pelec,
                                                          orb,
                                                          occupied_kpoints,
                                                          {kv.nmp[0], kv.nmp[1], kv.nmp[2]},
                                                          pw_wfc,
                                                          &sf,
                                                          global_out_dir);
    }
#endif

    //------------------------------------------------------------------
    //! 17) Perform RDMFT calculations, added by jghan, 2024-10-17
    //------------------------------------------------------------------
    if (inp.rdmft == true)
    {
        ModuleBase::matrix occ_num(pelec->wg);
        for (int ik = 0; ik < occ_num.nr; ++ik)
        {
            for (int inb = 0; inb < occ_num.nc; ++inb)
            {
                occ_num(ik, inb) /= kv.wk[ik];
            }
        }
        rdmft_solver.update_elec(ucell, occ_num, *psi);

        //! initialize the gradients of Etotal with respect to occupation numbers and wfc,
        //! and set all elements to 0.
        //! dedocc = d E/d Occ_Num
        ModuleBase::matrix dedocc(pelec->wg.nr, pelec->wg.nc, true);

        //! dedwfc = d E/d wfc
        psi::Psi<TK> dedwfc(psi->get_nk(), psi->get_nbands(), psi->get_nbasis(), kv.ngk, true);
        dedwfc.zero_out();

        double etot_rdmft = rdmft_solver.run(dedocc, dedwfc);
    }

    //------------------------------------------------------------------
    //! 17) Output quasi orbitals
    //------------------------------------------------------------------
    if (inp.qo_switch)
    {
        toQO tqo(inp.qo_basis, inp.qo_strategy, inp.qo_thr, inp.qo_screening_coeff);
        tqo.initialize(global_out_dir,
                       inp.pseudo_dir,
                       inp.orbital_dir,
                       &ucell,
                       kv.kvec_d,
                       GlobalV::ofs_running,
                       GlobalV::MY_RANK,
                       GlobalV::NPROC);
        tqo.calculate();
    }

    //------------------------------------------------------------------
    //! 18) Calculate and output asynchronous overlap matrix for Hefei-NAMD
    //------------------------------------------------------------------
    if (inp.cal_syns && (istep > 0 || inp.init_vel))
    {
        ModuleBase::TITLE("ModuleIO", "output_namd_async_overlap");
        ModuleBase::timer::tick("ModuleIO", "output_namd_async_overlap");

        // Create a new Overlap instance specifically for SR_async calculation
        // This allows SR_async to be initialized with velocity-shifted dtau
        hamilt::Overlap<hamilt::OperatorLCAO<TK, TR>>* overlap_async =
            new hamilt::Overlap<hamilt::OperatorLCAO<TK, TR>>(
                nullptr,  // hsk_in: not needed for SR_async calculation
                kv.kvec_d,
                nullptr,  // hR_in: not needed for SR_async calculation
                nullptr,  // SR_in: not needed for SR_async calculation
                &ucell,
                orb.cutoffs(),
                &gd,
                two_center_bundle.overlap_orb.get());

        // Use same precision as DMR output (default 8 if not specified)
        const int precision = inp.out_dmr[0] > 0 ? inp.out_dmr[1] : 8;
        const Parallel_Orbitals* paraV = p_hamilt->getSR()->get_paraV();
        hamilt::HContainer<TR>* SR_async = overlap_async->calculate_SR_async(ucell, PARAM.mdp.md_dt, paraV);
        overlap_async->output_SR_async_csr(istep, SR_async, precision);

        // Clean up
        delete SR_async;
        delete overlap_async;

        ModuleBase::timer::tick("ModuleIO", "output_namd_async_overlap");
    }

    ModuleBase::timer::tick("ModuleIO", "ctrl_scf_lcao");
}

// For gamma only
template void ModuleIO::ctrl_scf_lcao<double, double>(
    UnitCell& ucell,
    const Input_para& inp,
    K_Vectors& kv,
    elecstate::ElecState* pelec,
    elecstate::DensityMatrix<double, double>* dm, // mohan add 2025-11-04
    Parallel_Orbitals& pv,
    Grid_Driver& gd,
    psi::Psi<double>* psi,
    hamilt::HamiltLCAO<double, double>* p_hamilt,
    Plus_U& dftu, // mohan add 2025-11-07
    TwoCenterBundle& two_center_bundle,
    LCAO_Orbitals& orb,
    const ModulePW::PW_Basis_K* pw_wfc,         // for berryphase
    const ModulePW::PW_Basis* pw_rho,           // for berryphase
    const ModulePW::PW_Basis_Big* pw_big,       // for Wannier90
    const Structure_Factor& sf,                 // for Wannier90
    rdmft::RDMFT<double, double>& rdmft_solver, // for RDMFT
    Setup_DeePKS<double>& deepks,
    Exx_NAO<double>& exx_nao,
    const bool conv_esolver,
    const bool scf_nmax_flag,
    const int istep);

// For multiple k-points
template void ModuleIO::ctrl_scf_lcao<std::complex<double>, double>(
    UnitCell& ucell,
    const Input_para& inp,
    K_Vectors& kv,
    elecstate::ElecState* pelec,
    elecstate::DensityMatrix<std::complex<double>, double>* dm, // mohan add 2025-11-04
    Parallel_Orbitals& pv,
    Grid_Driver& gd,
    psi::Psi<std::complex<double>>* psi,
    hamilt::HamiltLCAO<std::complex<double>, double>* p_hamilt,
    Plus_U& dftu, // mohan add 2025-11-07
    TwoCenterBundle& two_center_bundle,
    LCAO_Orbitals& orb,
    const ModulePW::PW_Basis_K* pw_wfc,                       // for berryphase
    const ModulePW::PW_Basis* pw_rho,                         // for berryphase
    const ModulePW::PW_Basis_Big* pw_big,                     // for Wannier90
    const Structure_Factor& sf,                               // for Wannier90
    rdmft::RDMFT<std::complex<double>, double>& rdmft_solver, // for RDMFT
    Setup_DeePKS<std::complex<double>>& deepks,
    Exx_NAO<std::complex<double>>& exx_nao,
    const bool conv_esolver,
    const bool scf_nmax_flag,
    const int istep);

template void ModuleIO::ctrl_scf_lcao<std::complex<double>, std::complex<double>>(
    UnitCell& ucell,
    const Input_para& inp,
    K_Vectors& kv,
    elecstate::ElecState* pelec,
    elecstate::DensityMatrix<std::complex<double>, double>* dm, // mohan add 2025-11-04
    Parallel_Orbitals& pv,
    Grid_Driver& gd,
    psi::Psi<std::complex<double>>* psi,
    hamilt::HamiltLCAO<std::complex<double>, std::complex<double>>* p_hamilt,
    Plus_U& dftu, // mohan add 2025-11-07
    TwoCenterBundle& two_center_bundle,
    LCAO_Orbitals& orb,
    const ModulePW::PW_Basis_K* pw_wfc,                                     // for berryphase
    const ModulePW::PW_Basis* pw_rho,                                       // for berryphase
    const ModulePW::PW_Basis_Big* pw_big,                                   // for Wannier90
    const Structure_Factor& sf,                                             // for Wannier90
    rdmft::RDMFT<std::complex<double>, std::complex<double>>& rdmft_solver, // for RDMFT
    Setup_DeePKS<std::complex<double>>& deepks,
    Exx_NAO<std::complex<double>>& exx_nao,
    const bool conv_esolver,
    const bool scf_nmax_flag,
    const int istep);
