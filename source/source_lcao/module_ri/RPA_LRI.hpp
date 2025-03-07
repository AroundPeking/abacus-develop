//=======================
// AUTHOR : Rong Shi
// DATE :   2022-12-09
//=======================

#ifndef RPA_LRI_HPP
#define RPA_LRI_HPP
#include "RPA_LRI.h"
#include "module_parameter/parameter.h"
#include "module_ri/module_exx_symmetry/symmetry_rotation.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

using TA = int;
using Tcell = int;
static constexpr std::size_t Ndim = 3;
using TC = std::array<Tcell, Ndim>;
using Tq = std::array<double, Ndim>;
using TAC = std::pair<TA, TC>;
using TAq = std::pair<TA, Tq>;

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::init(const MPI_Comm& mpi_comm_in, const K_Vectors& kv_in, const std::vector<double>& orb_cutoff)
{
    ModuleBase::TITLE("RPA_LRI", "init");
    ModuleBase::timer::tick("RPA_LRI", "init");
    this->mpi_comm = mpi_comm_in;
    this->orb_cutoff_ = orb_cutoff;
    this->lcaos = exx_lri_rpa->lcaos;
    this->abfs = exx_lri_rpa->abfs;
    this->abfs_ccp = exx_lri_rpa->abfs_ccp;
    this->p_kv = &kv_in;

    //	this->cv = std::move(exx_lri_rpa.cv);
    //    exx_lri_rpa.cv = exx_lri_rpa.cv;
}

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::cal_rpa_cv(const UnitCell& ucell)
{
    std::vector<TA> atoms(ucell.nat);
    for (int iat = 0; iat < ucell.nat; ++iat)
    {
        atoms[iat] = iat;
    }
    const std::array<Tcell, Ndim> period = {p_kv->nmp[0], p_kv->nmp[1], p_kv->nmp[2]};

    const std::array<Tcell, Ndim> period_Vs
        = LRI_CV_Tools::cal_latvec_range<Tcell>(1 + this->info.ccp_rmesh_times, orb_cutoff_);
    const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA, std::array<Tcell, Ndim>>>>> list_As_Vs
        = RI::Distribute_Equally::distribute_atoms(this->mpi_comm, atoms, period_Vs, 2, false);

    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> Vs
        = exx_lri_rpa->cv.cal_Vs(list_As_Vs.first, list_As_Vs.second[0], {{"writable_Vws", true}});
    std::cout << "Use rpa_ccp_rmesh_times=" << this->info.ccp_rmesh_times << " to calculate cut Coulomb" << std::endl;
    this->Vs_period = RI::RI_Tools::cal_period(Vs, period);

    const std::array<Tcell, Ndim> period_Cs = LRI_CV_Tools::cal_latvec_range<Tcell>(2, lcaos_rmax);
    const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA, std::array<Tcell, Ndim>>>>> list_As_Cs
        = RI::Distribute_Equally::distribute_atoms_periods(this->mpi_comm, atoms, period_Cs, 2, false);

    std::pair<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>,
              std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, 3>>>>
        Cs_dCs = exx_lri_rpa->cv.cal_Cs_dCs(list_As_Cs.first,
                                            list_As_Cs.second[0],
                                            {{"cal_dC", false},
                                             {"writable_Cws", true},
                                             {"writable_dCws", true},
                                             {"writable_Vws", false},
                                             {"writable_dVws", false}});
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Cs = std::get<0>(Cs_dCs);
    this->Cs_period = RI::RI_Tools::cal_period(Cs, period);
}

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::cal_postSCF_exx(const int istep,
                                        const elecstate::DensityMatrix<T, Tdata>& dm,
                                        const MPI_Comm& mpi_comm_in,
                                        const K_Vectors& kv,
                                        const LCAO_Orbitals& orb)
{
    Mix_DMk_2D mix_DMk_2D;
    bool exx_spacegroup_symmetry = (PARAM.inp.nspin < 4 && ModuleSymmetry::Symmetry::symm_flag == 1);
    if (exx_spacegroup_symmetry)
    {
        mix_DMk_2D.set_nks(kv.get_nkstot_full() * (PARAM.inp.nspin == 2 ? 2 : 1), PARAM.globalv.gamma_only_local);
    }
    else
    {
        mix_DMk_2D.set_nks(kv.get_nks(), PARAM.globalv.gamma_only_local);
    }

    mix_DMk_2D.set_mixing(nullptr);
    ModuleSymmetry::Symmetry_rotation symrot;
    if (exx_spacegroup_symmetry)
    {
        const std::array<Tcell, Ndim> period = RI_Util::get_Born_vonKarmen_period(kv);
        symrot.find_irreducible_sector(GlobalC::ucell.symm,
                                       GlobalC::ucell.atoms,
                                       GlobalC::ucell.st,
                                       RI_Util::get_Born_von_Karmen_cells(period),
                                       period,
                                       GlobalC::ucell.lat);
        symrot.cal_Ms(kv, GlobalC::ucell, *dm.get_paraV_pointer());
        mix_DMk_2D.mix(symrot.restore_dm(kv, dm.get_DMK_vector(), *dm.get_paraV_pointer()), true);
    }
    else
    {
        mix_DMk_2D.mix(dm.get_DMK_vector(), true);
    }

    const std::vector<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>> Ds
        = PARAM.globalv.gamma_only_local ? RI_2D_Comm::split_m2D_ktoR<Tdata>(kv,
                                                                             mix_DMk_2D.get_DMk_gamma_out(),
                                                                             *dm.get_paraV_pointer(),
                                                                             PARAM.inp.nspin)
                                         : RI_2D_Comm::split_m2D_ktoR<Tdata>(kv,
                                                                             mix_DMk_2D.get_DMk_k_out(),
                                                                             *dm.get_paraV_pointer(),
                                                                             PARAM.inp.nspin,
                                                                             exx_spacegroup_symmetry);

    // set parameters for bare Coulomb potential
    GlobalC::exx_info.info_global.ccp_type = Conv_Coulomb_Pot_K::Ccp_Type::Hf;
    // reserve exx_ccp_rmesh_times to calculate full Coulomb
    this->exx_ccp_rmesh_times = GlobalC::exx_info.info_ri.ccp_rmesh_times;
    // rpa=1 set
    // GlobalC::exx_info.info_ri.ccp_rmesh_times=rpa_ccp_rmesh_times
    // Using this->info.ccp_rmesh_times to calculate cut Coulomb this->Vs_period
    GlobalC::exx_info.info_ri.ccp_rmesh_times = PARAM.inp.rpa_ccp_rmesh_times;
    GlobalC::exx_info.info_global.hybrid_alpha = 1;
    if (!exx_lri_rpa)
        exx_lri_rpa = new Exx_LRI<double>(GlobalC::exx_info.info_ri, GlobalC::exx_info.info_ewald);

    exx_lri_rpa->init(mpi_comm_in, kv, orb);
    exx_lri_rpa->cal_exx_ions(0, PARAM.inp.out_ri_cv);

    if (exx_spacegroup_symmetry && PARAM.inp.exx_symmetry_realspace)
    {
        exx_lri_rpa->cal_exx_elec(Ds, *dm.get_paraV_pointer(), &symrot);
    }
    else
    {
        exx_lri_rpa->cal_exx_elec(Ds, *dm.get_paraV_pointer());
    }
    // cout<<"postSCF_Eexx: "<<exx_lri_rpa.Eexx<<endl;
}

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::out_for_RPA(const UnitCell& ucell,
                                    const Parallel_Orbitals& parav,
                                    const psi::Psi<T>& psi,
                                    const elecstate::ElecState* pelec,
                                    const K_Vectors& kv,
                                    const LCAO_Orbitals& orb)
{
    ModuleBase::TITLE("DFT_RPA_interface", "out_for_RPA");
    this->out_bands(pelec);
    this->out_eigen_vector(parav, psi);
    this->out_struc(ucell.latvec, ucell.G);

    this->cal_rpa_cv(ucell);
    std::cout << "rpa_pca_threshold: " << this->info.pca_threshold << std::endl;
    std::cout << "rpa_ccp_rmesh_times: " << this->info.ccp_rmesh_times << std::endl;
    std::cout << "rpa_lcao_exx(Ha): " << std::fixed << std::setprecision(15) << exx_lri_rpa->Eexx / 2.0 << std::endl;
    this->out_Cs();
    this->out_coulomb_k(this->Vs_period, "coulomb_cut_", exx_lri_rpa);

    std::cout << "etxc(Ha): " << std::fixed << std::setprecision(15) << pelec->f_en.etxc / 2.0 << std::endl;
    std::cout << "etot(Ha): " << std::fixed << std::setprecision(15) << pelec->f_en.etot / 2.0 << std::endl;
    std::cout << "Etot_without_rpa(Ha): " << std::fixed << std::setprecision(15)
              << (pelec->f_en.etot - pelec->f_en.etxc + exx_lri_rpa->Eexx) / 2.0 << std::endl;

    if (GlobalC::exx_info.info_ri.shrink_abfs_pca_thr > 0.0)
    {
        cal_abfs_overlap(orb, kv);
    }
    if (this->info_ewald.use_ewald)
    {
        delete exx_lri_rpa;
        exx_lri_rpa = nullptr;
        Vs_period.clear();
        Cs_period.clear();
        {
            std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> tmp;
            Vs_period.swap(tmp);
            Cs_period.swap(tmp);
        }
        GlobalC::exx_info.info_ewald.fq_type = Singular_Value::Fq_type(PARAM.inp.exx_fq_type);
        GlobalC::exx_info.info_global.ccp_type = Conv_Coulomb_Pot_K::Ccp_Type::Ccp;
        // Using exx_ccp_rmesh_times to calculate full Coulomb
        GlobalC::exx_info.info_ri.ccp_rmesh_times = this->exx_ccp_rmesh_times;
        if (!exx_full_coulomb)
            exx_full_coulomb = new Exx_LRI<double>(GlobalC::exx_info.info_ri, GlobalC::exx_info.info_ewald);

        exx_full_coulomb->init(mpi_comm, kv, orb);
        exx_full_coulomb->cal_exx_ions(0, PARAM.inp.out_ri_cv);
        auto& Vs_full = exx_full_coulomb->Vs;
        this->out_coulomb_k(Vs_full, "coulomb_mat_", exx_full_coulomb);
        delete exx_full_coulomb;
        exx_full_coulomb = nullptr;
    }

    return;
}

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::cal_abfs_overlap(const LCAO_Orbitals& orb, const K_Vectors& kv)
{
    if (GlobalV::MY_RANK != 0)
    {
        return;
    }
    ModuleBase::TITLE("DFT_RPA_interface", "out_abfs_overlap");

    // shrinked abfs
    const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> abfs_s
        = Exx_Abfs::Construct_Orbs::abfs_same_atom(orb,
                                                   this->lcaos,
                                                   this->info.kmesh_times,
                                                   this->info.shrink_abfs_pca_thr);
    Exx_Abfs::Construct_Orbs::print_orbs_size(abfs_s, GlobalV::ofs_running);
    int Lmax;
    ORB_gaunt_table MGT;
    // const double lcaos_rmax = Exx_Abfs::Construct_Orbs::get_Rmax(this->lcaos);
    // const double abfs_rmax = Exx_Abfs::Construct_Orbs::get_Rmax(this->abfs);
    this->m_abfs_abf.init(2, orb, this->info.kmesh_times, orb.get_Rmax(), Lmax);
    for (size_t T = 0; T != this->abfs.size(); ++T)
    {
        Lmax = std::max(Lmax, static_cast<int>(this->abfs[T].size()) - 1);
    }
    MGT.init_Gaunt_CH(Lmax);
    MGT.init_Gaunt(Lmax);
    this->m_abfs_abf.init_radial(abfs_s, this->abfs, MGT);
    this->m_abfs_abf.init_radial_table();

    // const double abfs_s_rmax = Exx_Abfs::Construct_Orbs::get_Rmax(abfs_s);
    this->m_abfs_abfs.init(2, orb, this->info.kmesh_times, orb.get_Rmax(), Lmax);
    this->m_abfs_abfs.init_radial(abfs_s, abfs_s, MGT);
    this->m_abfs_abfs.init_radial_table();
    // get Rlist
    const std::array<Tcell, Ndim> period = RI_Util::get_Born_vonKarmen_period(kv);
    const auto R_period = RI_Util::get_Born_von_Karmen_cells(period);
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> overlap_abfs_abfs;
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> overlap_abfs_abf;

    // index of smaller abfs
    const ModuleBase::Element_Basis_Index::Range range_abfs_s = Exx_Abfs::Abfs_Index::construct_range(abfs_s);
    const ModuleBase::Element_Basis_Index::IndexLNM index_abfs_s
        = ModuleBase::Element_Basis_Index::construct_index(range_abfs_s);
    // index of larger abfs
    const ModuleBase::Element_Basis_Index::Range range_abfs = Exx_Abfs::Abfs_Index::construct_range(this->abfs);
    const ModuleBase::Element_Basis_Index::IndexLNM index_abfs
        = ModuleBase::Element_Basis_Index::construct_index(range_abfs);

    for (int I = 0; I != GlobalC::ucell.nat; I++)
    {
        const size_t TA = GlobalC::ucell.iat2it[I];
        const size_t IA = GlobalC::ucell.iat2ia[I];
        const ModuleBase::Vector3<double>& tauA(GlobalC::ucell.atoms[TA].tau[IA]);

        for (int J = 0; J != GlobalC::ucell.nat; J++)
        {
            const size_t TB = GlobalC::ucell.iat2it[J];
            const size_t IB = GlobalC::ucell.iat2ia[J];
            const ModuleBase::Vector3<double>& tauB(GlobalC::ucell.atoms[TB].tau[IB]);

            for (auto& iR: R_period)
            {
                const ModuleBase::Vector3<double> tauB_shift
                    = tauB + (RI_Util::array3_to_Vector3(iR) * GlobalC::ucell.latvec);
                overlap_abfs_abfs[I][{J, iR}]
                    = this->m_abfs_abfs.cal_overlap_matrix<Tdata>(TA,
                                                                  TB,
                                                                  tauA,
                                                                  tauB_shift,
                                                                  index_abfs_s,
                                                                  index_abfs_s,
                                                                  Matrix_Orbs11::Matrix_Order::AB);
                overlap_abfs_abf[I][{J, iR}]
                    = this->m_abfs_abf.cal_overlap_matrix<Tdata>(TA,
                                                                 TB,
                                                                 tauA,
                                                                 tauB_shift,
                                                                 index_abfs_s,
                                                                 index_abfs,
                                                                 Matrix_Orbs11::Matrix_Order::AB);
            }
        }
    }
    // debug
    //  for (auto& iR: R_period)
    //  {
    //      std::cout << iR[0] << iR[1] << iR[2] << std::endl;
    //      std::cout << "(0,59): " << overlap_abfs_abfs[0][{1, iR}](0, 0) << std::endl;
    //      std::cout << "(59,0): " << overlap_abfs_abfs[1][{0, iR}](0, 0) << std::endl;
    //  }
    out_abfs_overlap(overlap_abfs_abfs, overlap_abfs_abf, "shrink_sinvS_", index_abfs_s, index_abfs);
}

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::inverse_olp(std::map<TA, std::map<TAq, RI::Tensor<std::complex<double>>>>& overlap_abfs_abfs,
                                    const std::vector<ModuleBase::Vector3<double>>& q_period,
                                    const ModuleBase::Element_Basis_Index::IndexLNM& index_abfs_s)
{
    int all_mu_s = 0;
    vector<int> mu_s_shift(GlobalC::ucell.nat);
    for (int I = 0; I != GlobalC::ucell.nat; I++)
    {
        mu_s_shift[I] = all_mu_s;
        all_mu_s += index_abfs_s[GlobalC::ucell.iat2it[I]].count_size;
    }
    RI::Tensor<std::complex<double>> olp_all = RI::Tensor<std::complex<double>>({all_mu_s, all_mu_s});
    for (int iq = 0; iq < q_period.size(); iq++)
    {
        for (auto& Ip: overlap_abfs_abfs)
        {
            auto I = Ip.first;
            size_t mu_s_I = index_abfs_s[GlobalC::ucell.iat2it[I]].count_size;
            for (auto& JPp: Ip.second)
            {
                auto q = JPp.first.second;
                if (RI_Util::array3_to_Vector3(q) != q_period[iq])
                    continue;
                auto J = JPp.first.first;
                auto mu_s_J = index_abfs_s[GlobalC::ucell.iat2it[J]].count_size;
                for (int ir = 0; ir < mu_s_I; ir++)
                {
                    for (int ic = 0; ic < mu_s_J; ic++)
                        olp_all(mu_s_shift[I] + ir, mu_s_shift[J] + ic) = JPp.second(ir, ic);
                }
            }
        }
        // debug
        //  std::cout << "(0,59): " << olp_all(0, 59) << std::endl;
        //  std::cout << "(59,0): " << olp_all(59, 0) << std::endl;
        //  std::cout << "(0,59): " << overlap_abfs_abfs[0][{1, RI_Util::Vector3_to_array3(q_period[iq])}](0, 0)
        //            << std::endl;
        //  std::cout << "(59,0): " << overlap_abfs_abfs[1][{0, RI_Util::Vector3_to_array3(q_period[iq])}](0, 0)
        //            << std::endl;
        auto olp_inv = LRI_CV_Tools::cal_I(olp_all);
        for (auto& Ip: overlap_abfs_abfs)
        {
            auto I = Ip.first;
            size_t mu_s_I = index_abfs_s[GlobalC::ucell.iat2it[I]].count_size;
            for (auto& JPp: Ip.second)
            {
                auto q = JPp.first.second;
                if (RI_Util::array3_to_Vector3(q) != q_period[iq])
                    continue;
                auto J = JPp.first.first;
                auto mu_s_J = index_abfs_s[GlobalC::ucell.iat2it[J]].count_size;

                for (int ir = 0; ir < mu_s_I; ir++)
                {
                    for (int ic = 0; ic < mu_s_J; ic++)
                        JPp.second(ir, ic) = olp_inv(mu_s_shift[I] + ir, mu_s_shift[J] + ic);
                }
            }
        }
    }
}

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::out_abfs_overlap(std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& overlap_abfs_abfs,
                                         std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& overlap_abfs_abf,
                                         std::string filename,
                                         const ModuleBase::Element_Basis_Index::IndexLNM& index_abfs_s,
                                         const ModuleBase::Element_Basis_Index::IndexLNM& index_abfs)
{
    int all_mu_s = 0;
    int all_mu = 0;
    vector<int> mu_s_shift(GlobalC::ucell.nat);
    vector<int> mu_shift(GlobalC::ucell.nat);
    for (int I = 0; I != GlobalC::ucell.nat; I++)
    {
        mu_s_shift[I] = all_mu_s;
        mu_shift[I] = all_mu;
        all_mu_s += index_abfs_s[GlobalC::ucell.iat2it[I]].count_size;
        all_mu += index_abfs[GlobalC::ucell.iat2it[I]].count_size;
    }
    const int nks_tot = PARAM.inp.nspin == 2 ? (int)p_kv->get_nks() / 2 : p_kv->get_nks();
    std::stringstream ss;
    ss << filename << GlobalV::MY_RANK << ".txt";

    std::ofstream ofs;
    ofs.open(ss.str().c_str(), std::ios::out);

    ofs << nks_tot << std::endl;
    // Fourier of ss(R->k), s(R->k)
    std::map<TA, std::map<TAq, RI::Tensor<std::complex<double>>>> olp_q_ss;
    std::map<TA, std::map<TAq, RI::Tensor<std::complex<double>>>> olp_q_s;
    for (int ik = 0; ik != nks_tot; ik++)
    {
        for (auto& Ip: overlap_abfs_abfs)
        {
            auto I = Ip.first;
            for (auto& JPp: Ip.second)
            {
                auto J = JPp.first.first;
                auto R = JPp.first.second;
                auto q = RI_Util::Vector3_to_array3(p_kv->kvec_c[ik]);
                RI::Tensor<std::complex<double>> tmp_olp_ss
                    = RI::Global_Func::convert<std::complex<double>>(JPp.second);
                RI::Tensor<std::complex<double>> tmp_olp_s
                    = RI::Global_Func::convert<std::complex<double>>(overlap_abfs_abf[I][{J, R}]);
                if (olp_q_ss[I][{J, q}].empty())
                {
                    olp_q_ss[I][{J, q}] = RI::Tensor<std::complex<double>>({tmp_olp_ss.shape[0], tmp_olp_ss.shape[1]});
                    olp_q_s[I][{J, q}] = RI::Tensor<std::complex<double>>({tmp_olp_s.shape[0], tmp_olp_s.shape[1]});
                }
                const double arg = 1 * (p_kv->kvec_c[ik] * (RI_Util::array3_to_Vector3(R) * GlobalC::ucell.latvec))
                                   * ModuleBase::TWO_PI; // latvec
                const std::complex<double> kphase = std::complex<double>(cos(arg), sin(arg));

                olp_q_ss[I][{J, q}] = olp_q_ss[I][{J, q}] + tmp_olp_ss * kphase;
                olp_q_s[I][{J, q}] = olp_q_s[I][{J, q}] + tmp_olp_s * kphase;
            }
        }
    }
    inverse_olp(olp_q_ss, p_kv->kvec_c, index_abfs_s);
    for (auto& Ip: overlap_abfs_abf)
    {
        auto I = Ip.first;
        size_t mu_num_s = index_abfs_s[GlobalC::ucell.iat2it[I]].count_size;
        size_t mu_num = index_abfs[GlobalC::ucell.iat2it[I]].count_size;

        for (int ik = 0; ik != nks_tot; ik++)
        {
            std::map<size_t, RI::Tensor<std::complex<double>>> sinvS;
            for (auto& JPp: Ip.second)
            {
                auto J = JPp.first.first;
                auto R = JPp.first.second;
                if (sinvS[J].empty())
                {
                    sinvS[J] = RI::Tensor<std::complex<double>>(
                        {overlap_abfs_abf[I][{J, R}].shape[0], overlap_abfs_abf[I][{J, R}].shape[1]});
                }
            }
            for (const auto& pair: sinvS)
            {
                auto J = pair.first;
                auto q = RI_Util::Vector3_to_array3(p_kv->kvec_c[ik]);
                for (int K = 0; K != GlobalC::ucell.nat; K++)
                {
                    sinvS[J] += olp_q_ss[I][{K, q}] * olp_q_s[K][{J, q}];
                }
            }
            for (auto& iJU: sinvS)
            {
                auto iJ = iJU.first;
                auto& vq_J = iJU.second;
                size_t nu_num = index_abfs[GlobalC::ucell.iat2it[iJ]].count_size;
                ofs << all_mu_s << "   " << all_mu << "   " << mu_s_shift[I] + 1 << "   " << mu_s_shift[I] + mu_num_s
                    << "  " << mu_shift[iJ] + 1 << "   " << mu_shift[iJ] + nu_num << std::endl;
                ofs << ik + 1 << "  " << p_kv->wk[ik] / 2.0 * PARAM.inp.nspin << std::endl;
                for (int i = 0; i != vq_J.data->size(); i++)
                {
                    ofs << std::setw(25) << std::fixed << std::setprecision(15) << (*vq_J.data)[i].real()
                        << std::setw(25) << std::fixed << std::setprecision(15) << (*vq_J.data)[i].imag() << std::endl;
                }
            }
        }
    }
    ofs.close();
}

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::out_eigen_vector(const Parallel_Orbitals& parav, const psi::Psi<T>& psi)
{

    ModuleBase::TITLE("DFT_RPA_interface", "out_eigen_vector");

    const int nks_tot = PARAM.inp.nspin == 2 ? p_kv->get_nks() / 2 : p_kv->get_nks();
    const int npsin_tmp = PARAM.inp.nspin == 2 ? 2 : 1;
    const std::complex<double> zero(0.0, 0.0);

    for (int ik = 0; ik < nks_tot; ik++)
    {
        std::stringstream ss;
        ss << "KS_eigenvector_" << ik << ".dat";

        std::ofstream ofs;
        if (GlobalV::MY_RANK == 0)
        {
            ofs.open(ss.str().c_str(), std::ios::out);
        }
        std::vector<ModuleBase::ComplexMatrix> is_wfc_ib_iw(npsin_tmp);
        for (int is = 0; is < npsin_tmp; is++)
        {
            is_wfc_ib_iw[is].create(PARAM.inp.nbands, PARAM.globalv.nlocal);
            for (int ib_global = 0; ib_global < PARAM.inp.nbands; ++ib_global)
            {
                std::vector<std::complex<double>> wfc_iks(PARAM.globalv.nlocal, zero);

                const int ib_local = parav.global2local_col(ib_global);

                if (ib_local >= 0)
                {
                    for (int ir = 0; ir < psi.get_nbasis(); ir++)
                    {
                        wfc_iks[parav.local2global_row(ir)] = psi(ik + nks_tot * is, ib_local, ir);
                    }
                }

                std::vector<std::complex<double>> tmp = wfc_iks;
#ifdef __MPI
                MPI_Allreduce(&tmp[0], &wfc_iks[0], PARAM.globalv.nlocal, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
#endif
                for (int iw = 0; iw < PARAM.globalv.nlocal; iw++)
                {
                    is_wfc_ib_iw[is](ib_global, iw) = wfc_iks[iw];
                }
            } // ib
        } // is
        ofs << ik + 1 << std::endl;
        for (int iw = 0; iw < PARAM.globalv.nlocal; iw++)
        {
            for (int ib = 0; ib < PARAM.inp.nbands; ib++)
            {
                for (int is = 0; is < npsin_tmp; is++)
                {
                    ofs << std::setw(30) << std::fixed << std::setprecision(20) << is_wfc_ib_iw[is](ib, iw).real()
                        << std::setw(30) << std::fixed << std::setprecision(20) << is_wfc_ib_iw[is](ib, iw).imag()
                        << std::endl;
                }
            }
        }
        ofs.close();
    } // ik
    return;
}

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::out_struc(const ModuleBase::Matrix3& latvec, const ModuleBase::Matrix3& G)
{
    if (GlobalV::MY_RANK != 0)
    {
        return;
    }
    ModuleBase::TITLE("DFT_RPA_interface", "out_struc");
    double TWOPI_Bohr2A = ModuleBase::TWO_PI * ModuleBase::BOHR_TO_A;
    const int nks_tot = PARAM.inp.nspin == 2 ? (int)p_kv->get_nks() / 2 : p_kv->get_nks();
    ModuleBase::Matrix3 lat = latvec / ModuleBase::BOHR_TO_A;
    ModuleBase::Matrix3 G_RPA = G * TWOPI_Bohr2A;
    std::stringstream ss;
    ss << "stru_out";
    std::ofstream ofs;
    ofs.open(ss.str().c_str(), std::ios::out);
    ofs << lat.e11 << std::setw(15) << lat.e12 << std::setw(15) << lat.e13 << std::endl;
    ofs << lat.e21 << std::setw(15) << lat.e22 << std::setw(15) << lat.e23 << std::endl;
    ofs << lat.e31 << std::setw(15) << lat.e32 << std::setw(15) << lat.e33 << std::endl;

    ofs << G_RPA.e11 << std::setw(15) << G_RPA.e12 << std::setw(15) << G_RPA.e13 << std::endl;
    ofs << G_RPA.e21 << std::setw(15) << G_RPA.e22 << std::setw(15) << G_RPA.e23 << std::endl;
    ofs << G_RPA.e31 << std::setw(15) << G_RPA.e32 << std::setw(15) << G_RPA.e33 << std::endl;

    ofs << p_kv->nmp[0] << std::setw(6) << p_kv->nmp[1] << std::setw(6) << p_kv->nmp[2] << std::setw(6) << std::endl;

    for (int ik = 0; ik != nks_tot; ik++)
    {
        ofs << std::setw(15) << std::fixed << std::setprecision(9) << p_kv->kvec_c[ik].x * TWOPI_Bohr2A << std::setw(15)
            << std::fixed << std::setprecision(9) << p_kv->kvec_c[ik].y * TWOPI_Bohr2A << std::setw(15) << std::fixed
            << std::setprecision(9) << p_kv->kvec_c[ik].z * TWOPI_Bohr2A << std::endl;
    }
    ofs.close();
    return;
}

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::out_bands(const elecstate::ElecState* pelec)
{
    ModuleBase::TITLE("DFT_RPA_interface", "out_bands");
    if (GlobalV::MY_RANK != 0)
    {
        return;
    }
    const int nks_tot = PARAM.inp.nspin == 2 ? (int)p_kv->get_nks() / 2 : p_kv->get_nks();
    const int nspin_tmp = PARAM.inp.nspin == 2 ? 2 : 1;
    std::stringstream ss;
    ss << "band_out";
    std::ofstream ofs;
    ofs.open(ss.str().c_str(), std::ios::out);
    ofs << nks_tot << std::endl;
    ofs << PARAM.inp.nspin << std::endl;
    ofs << PARAM.inp.nbands << std::endl;
    ofs << PARAM.globalv.nlocal << std::endl;
    ofs << (pelec->eferm.ef / 2.0) << std::endl;

    for (int ik = 0; ik != nks_tot; ik++)
    {
        for (int is = 0; is != nspin_tmp; is++)
        {
            ofs << std::setw(6) << ik + 1 << std::setw(6) << is + 1 << std::endl;
            for (int ib = 0; ib != PARAM.inp.nbands; ib++)
            {
                ofs << std::setw(5) << ib + 1 << "   " << std::setw(8) << pelec->wg(ik + is * nks_tot, ib) * nks_tot
                    << std::setw(25) << std::fixed << std::setprecision(20) << pelec->ekb(ik + is * nks_tot, ib) / 2.0
                    << std::setw(18) << std::fixed << std::setprecision(8)
                    << pelec->ekb(ik + is * nks_tot, ib) * ModuleBase::Ry_to_eV << std::endl;
            }
        }
    }
    ofs.close();
    return;
}

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::out_Cs(const UnitCell& ucell)
{
    std::stringstream ss;
    ss << "Cs_data_" << GlobalV::MY_RANK << ".txt";
    std::ofstream ofs;
    ofs.open(ss.str().c_str(), std::ios::out);
    ofs << ucell.nat << "    " << 0 << std::endl;
    for (auto& Ip: this->Cs_period)
    {
        size_t I = Ip.first;
        size_t i_num = ucell.atoms[ucell.iat2it[I]].nw;
        for (auto& JPp: Ip.second)
        {
            size_t J = JPp.first.first;
            auto R = JPp.first.second;
            auto& tmp_Cs = JPp.second;
            size_t j_num = ucell.atoms[ucell.iat2it[J]].nw;

            ofs << I + 1 << "   " << J + 1 << "   " << R[0] << "   " << R[1] << "   " << R[2] << "   " << i_num
                << std::endl;
            ofs << j_num << "   " << tmp_Cs.shape[0] << std::endl;
            for (int i = 0; i != i_num; i++)
            {
                for (int j = 0; j != j_num; j++)
                {
                    for (int mu = 0; mu != tmp_Cs.shape[0]; mu++)
                    {
                        ofs << std::setw(30) << std::fixed << std::setprecision(20) << tmp_Cs(mu, i, j) << std::endl;
                    }
                }
            }
        }
    }
    ofs.close();
    return;
}

template <typename T, typename Tdata>
void RPA_LRI<T, Tdata>::out_coulomb_k(std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs,
                                      std::string filename,
                                      Exx_LRI<double>* exx_lri)
{
    int all_mu = 0;
    vector<int> mu_shift(ucell.nat);
    for (int I = 0; I != ucell.nat; I++)
    {
        mu_shift[I] = all_mu;
        all_mu += exx_lri->cv.get_index_abfs_size(GlobalC::ucell.iat2it[I]);
    }
    const int nks_tot = PARAM.inp.nspin == 2 ? (int)p_kv->get_nks() / 2 : p_kv->get_nks();
    std::stringstream ss;
    ss << filename << GlobalV::MY_RANK << ".txt";

    std::ofstream ofs;
    ofs.open(ss.str().c_str(), std::ios::out);

    ofs << nks_tot << std::endl;
    for (auto& Ip: Vs)
    {
        auto I = Ip.first;
        size_t mu_num = exx_lri->cv.get_index_abfs_size(GlobalC::ucell.iat2it[I]);

        for (int ik = 0; ik != nks_tot; ik++)
        {
            std::map<size_t, RI::Tensor<std::complex<double>>> Vq_k_IJ;
            for (auto& JPp: Ip.second)
            {
                auto J = JPp.first.first;

                auto R = JPp.first.second;
                if (J < I)
                {
                    continue;
                }
                RI::Tensor<std::complex<double>> tmp_VR = RI::Global_Func::convert<std::complex<double>>(JPp.second);

                const double arg = 1 * (p_kv->kvec_c[ik] * (RI_Util::array3_to_Vector3(R) * ucell.latvec))
                                   * ModuleBase::TWO_PI; // latvec
                const std::complex<double> kphase = std::complex<double>(cos(arg), sin(arg));
                if (Vq_k_IJ[J].empty())
                {
                    Vq_k_IJ[J] = RI::Tensor<std::complex<double>>({tmp_VR.shape[0], tmp_VR.shape[1]});
                }
                Vq_k_IJ[J] = Vq_k_IJ[J] + tmp_VR * kphase;
            }
            for (auto& vq_Jp: Vq_k_IJ)
            {
                auto iJ = vq_Jp.first;
                auto& vq_J = vq_Jp.second;
                size_t nu_num = exx_lri->cv.get_index_abfs_size(GlobalC::ucell.iat2it[iJ]);
                ofs << all_mu << "   " << mu_shift[I] + 1 << "   " << mu_shift[I] + mu_num << "  " << mu_shift[iJ] + 1
                    << "   " << mu_shift[iJ] + nu_num << std::endl;
                ofs << ik + 1 << "  " << p_kv->wk[ik] / 2.0 * PARAM.inp.nspin << std::endl;
                for (int i = 0; i != vq_J.data->size(); i++)
                {
                    ofs << std::setw(25) << std::fixed << std::setprecision(15) << (*vq_J.data)[i].real()
                        << std::setw(25) << std::fixed << std::setprecision(15) << (*vq_J.data)[i].imag() << std::endl;
                }
            }
        }
    }
    ofs.close();
}

// template<typename Tdata>
// void RPA_LRI<T, Tdata>::init(const MPI_Comm &mpi_comm_in)
// {
// 	if(this->info == this->exx.info)
// 	{
// 		this->lcaos = this->exx.lcaos;
// 		this->abfs = this->exx.abfs;
// 		this->abfs_ccp = this->exx.abfs_ccp;

// 		exx_lri_rpa.cv = std::move(this->exx.cv);
// 	}
// 	else
// 	{
// 		this->lcaos = ...
// 		this->abfs = ...
// 		this->abfs_ccp = ...

// 		exx_lri_rpa.cv.set_orbitals(
// 			this->lcaos, this->abfs, this->abfs_ccp,
// 			this->info.kmesh_times, this->info.ccp_rmesh_times );
// 	}

// //	for( size_t T=0; T!=this->abfs.size(); ++T )
// //		GlobalC::exx_info.info_ri.abfs_Lmax = std::max(
// GlobalC::exx_info.info_ri.abfs_Lmax, static_cast<int>(this->abfs[T].size())-1
// );

// }

// template<typename Tdata>
// void RPA_LRI<T, Tdata>::cal_rpa_ions()
// {
// 	// this->rpa_lri.set_parallel(this->mpi_comm, atoms_pos, latvec, period);

// 	if(this->info == this->exx.info)
// 		exx_lri_rpa.cv.Vws = std::move(this->exx.cv.Vws);

// 	const std::array<Tcell,Ndim> period_Vs =
// LRI_CV_Tools::cal_latvec_range<Tcell>(1+this->info.ccp_rmesh_times); const
// std::pair<std::vector<TA>,
// std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>> 		list_As_Vs
// = RI::Distribute_Equally::distribute_atoms(this->mpi_comm, atoms, period_Vs,
// 2, false);

// 	std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>
// 		Vs = exx_lri_rpa.cv.cal_Vs(
// 			list_As_Vs.first, list_As_Vs.second[0],
// 			{{"writable_Vws",true}});

// 	// Vs[iat0][{iat1,cell1}]	按 (iat0,iat1) 分进程，每个进程有所有 cell1
// 	Vqs = FFT(Vs);
// 	out_Vs(Vqs);

// 	if(this->info == this->exx.info)
// 		exx_lri_rpa.cv.Cws = std::move(this->exx.cv.Cws);

// 	const std::array<Tcell,Ndim> period_Cs =
// LRI_CV_Tools::cal_latvec_range<Tcell>(2); 	const std::pair<std::vector<TA>,
// std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>> 		list_As_Cs
// = RI::Distribute_Equally::distribute_atoms_periods(this->mpi_comm, atoms,
// period_Cs, 2, false);

// 	std::pair<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>,
// std::array<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>,3>> 		Cs_dCs =
// exx_lri_rpa.cv.cal_Cs_dCs( 			list_As_Cs.first, list_As_Cs.second[0],
// 			{{"cal_dC",false},
// 			 {"writable_Cws",true}, {"writable_dCws",true},
// {"writable_Vws",false},
// {"writable_dVws",false}}); 	std::map<TA,std::map<TAC,RI::Tensor<Tdata>>> &Cs
// = std::get<0>(Cs_dCs);

// 	out_Cs(Cs);

// 	// rpa_lri.set_Cs(Cs);
// }

#endif
