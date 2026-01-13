#ifndef EXX_ROTATE_ABFS_HPP
#define EXX_ROTATE_ABFS_HPP
#include "../module_base/constants.h"
#include "../module_base/math_integral.h"
#include "exx_rotate_abfs.h"

#include <vector>

template <typename Tdata>
void Moment_abfs<Tdata>::cal_multipole(const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in)
{
    ModuleBase::TITLE("Rotate_abfs", "cal_multipole");
    ModuleBase::timer::tick("Rotate_abfs", "cal_multipole");

    this->multipole.resize(orb_in.size());
    for (size_t T = 0; T != orb_in.size(); ++T)
    {
        this->multipole[T].resize(orb_in[T].size());
        for (size_t L = 0; L != orb_in[T].size(); ++L)
        {
            this->multipole[T][L].resize(orb_in[T][L].size());
            for (size_t N = 0; N != orb_in[T][L].size(); ++N)
            {
                const Numerical_Orbital_Lm& orb_lm = orb_in[T][L][N];
                const int nr = orb_lm.getNr();
                double* integrated_func = new double[nr];
                for (size_t ir = 0; ir != nr; ++ir)
                    integrated_func[ir] = orb_lm.getPsi(ir) * std::pow(orb_lm.getRadial(ir), 2 + L);

                ModuleBase::Integral::Simpson_Integral(nr, integrated_func, orb_lm.getRab(), this->multipole[T][L][N]);
            }
        }
    }

    ModuleBase::timer::tick("Rotate_abfs", "cal_multipole");
}

template <typename Tdata>
void Moment_abfs<Tdata>::rotate_abfs(std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in)
{
    ModuleBase::TITLE("Rotate_abfs", "rotate_abfs");
    ModuleBase::timer::tick("Rotate_abfs", "rotate_abfs");

    // construct tranformation matrix A
    for (int T = 0; T != orb_in.size(); ++T)
    {
        for (int L = 0; L != orb_in[T].size(); ++L)
        {
            for (int N = 0; N != orb_in[T][L].size(); ++N)
            {
                Numerical_Orbital_Lm& orb_lm_old = orb_in[T][L][N];
                Numerical_Orbital_Lm orb_lm_mod = orb_lm_old * 0.0;
                double square = 0.0;
                for (int N2 = 0; N2 != orb_in[T][L].size(); ++N2)
                {
                    square += this->multipole[T][L][N2] * this->multipole[T][L][N2];
                }
                double norm = std::sqrt(square);
                // change abfs
                if (N == 0)
                {
                    for (int N2 = 0; N2 != orb_in[T][L].size(); ++N2)
                        orb_lm_mod += orb_in[T][L][N2] * this->multipole[T][L][N2] / norm;
                }
                else
                {
                    for (int N2 = 0; N2 != orb_in[T][L].size(); ++N2)
                    {
                        if (N2 == N)
                        {
                            orb_lm_mod += (1.0 - this->multipole[T][L][N] * this->multipole[T][L][N2] / square)
                                          * orb_in[T][L][N2];
                        }
                        else
                        {
                            orb_lm_mod
                                += (-this->multipole[T][L][N] * this->multipole[T][L][N2] / square) * orb_in[T][L][N2];
                        }
                    }
                }
                orb_lm_old = orb_lm_mod;
                // change moment
                if (N == 0)
                {
                    this->multipole[T][L][N] = norm;
                    std::cout << "Atom type " << T << ", L " << L << ", N " << N
                              << ", multipole after rotation: " << this->multipole[T][L][N] << std::endl;
                }
                else
                {
                    this->multipole[T][L][N] = 0.0;
                }
            }
        }
    }

    ModuleBase::timer::tick("Rotate_abfs", "rotate_abfs");
}

template <typename Tdata>
int Moment_abfs<Tdata>::dfact(const int& l) const
{
    int result = 1;
    for (int i = l; i > 1; i -= 2)
    {
        result *= i;
    }
    return result;
}

template <typename Tdata>
int Moment_abfs<Tdata>::factorial(const int& n) const
{
    if (n == 0)
    {
        return 1;
    }
    else if (n > 0)
    {
        return n * this->factorial(n - 1);
    }
    else
    {
        ModuleBase::WARNING_QUIT("Moment_abfs::factorial", "n is out of range");
        return 0;
    }
}

template <typename Tdata>
double Moment_abfs<Tdata>::cal_clmlm(int l2, int m2, int l, int m, const ORB_gaunt_table& MGT) const
{
    double result = 0.0;
    // real m and index_m mix due to MGT.get_lm_index
    result = std::pow(-1, m2) * std::pow(ModuleBase::FOUR_PI, 2) * dfact(2 * (l2 + l) - 1) / dfact(2 * l2 + 1)
             / dfact(2 * l + 1)
             * MGT.Gaunt_Coefficients(MGT.get_lm_index(l2, m2 + l2),
                                      MGT.get_lm_index(l, m + l),
                                      MGT.get_lm_index(l2 + l, m - m2 + l2 + l));

    // only real m
    // result = std::pow(ModuleBase::FOUR_PI, 1.5)
    //          * std::sqrt((this->factorial(l2 + l + m - m2) * this->factorial(l2 + l - m + m2))
    //                      / (this->factorial(l2 + m2) * this->factorial(l2 - m2) * this->factorial(l + m)
    //                         * this->factorial(l - m)))
    //          / std::sqrt((2 * l2 + 1) * (2 * l + 1) * (2 * l2 + 2 * l + 1));
    // result = std::pow(ModuleBase::FOUR_PI, 1.5)
    //          * std::sqrt((this->factorial(l2 + l + m + m2) * this->factorial(l2 + l - m - m2))
    //                      / (this->factorial(l2 + m2) * this->factorial(l2 - m2) * this->factorial(l + m)
    //                         * this->factorial(l - m)))
    //          / std::sqrt((2 * l2 + 1) * (2 * l + 1) * (2 * l2 + 2 * l + 1));

    return result;
}

template <typename Tdata>
void Moment_abfs<Tdata>::cal_VR(
    const UnitCell& ucell,
    const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in,
    const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA, std::array<Tcell, Ndim>>>>>& list_r,
    const std::vector<double>& orb_cutoff,
    const double Rc,
    LRI_CV<Tdata>& cv,
    std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_cut)
{
    ModuleBase::TITLE("Rotate_abfs", "cal_VR");
    ModuleBase::timer::tick("Rotate_abfs", "cal_VR");
    // copy in source\module_hamilt_lcao\hamilt_lcaodft\center2_orb-orb11.cpp
    const double tiny1 = 1e-12;
    const double tiny2 = 1e-10;
    const ModuleBase::Element_Basis_Index::Range range = Exx_Abfs::Abfs_Index::construct_range(orb_in);
    // index: T: type, L: angular momentum, N: radial index, M: magnetic moment
    const ModuleBase::Element_Basis_Index::IndexLNM index = ModuleBase::Element_Basis_Index::construct_index(range);

    ORB_gaunt_table MGT0;
    int Lmax = 0;
    for (size_t T = 0; T != orb_in.size(); ++T)
    {
        Lmax = std::max(Lmax, static_cast<int>(orb_in[T].size()) - 1);
    }

    ModuleBase::TITLE("cal_Gaunt_before");
    MGT0.init_Gaunt_CH(Lmax);
    MGT0.init_Gaunt(Lmax);
    ModuleBase::TITLE("cal_Gaunt_after");

    const auto list_A0 = list_r.first;
    const auto list_A1 = list_r.second[0];

    auto& Vws = cv.Vws;
    for (size_t i0 = 0; i0 < list_A0.size(); ++i0)
    {
        const TA iat0 = list_A0[i0];
        const int T1 = ucell.iat2it[iat0];
        const size_t I1 = ucell.iat2ia[iat0];
        const auto& tauA = ucell.atoms[T1].tau[I1];
        const size_t sizeA = index[T1].count_size;
        for (size_t i1 = 0; i1 < list_A1.size(); ++i1)
        {
            const TA iat1 = list_A1[i1].first;
            const int T2 = ucell.iat2it[iat1];
            const size_t I2 = ucell.iat2ia[iat1];
            const auto& tauB = ucell.atoms[T2].tau[I2];
            const size_t sizeB = index[T2].count_size;
            const auto R = list_A1[i1].second;
            // delta_R: Angstrom
            const auto delta_R = tauB - tauA + (RI_Util::array3_to_Vector3(R) * ucell.latvec);
            // bohr
            const double distance_true = (delta_R).norm() * ucell.lat0;
            // bohr
            const double distance = (distance_true >= tiny1) ? distance_true : distance_true + tiny1;
            // Rcut_lcao: bohr, ucell.lat0 = 1.8897 transform angstrom to bohr
            double Rcut_lcao = orb_cutoff[T1] + orb_cutoff[T2];
            // Rcut_coul: bohr
            double Rcut_coul = std::min(cv.cal_V_Rcut(T1, T2), cv.cal_V_Rcut(T2, T2));
            if (distance < Rcut_lcao || distance >= Rcut_coul)
                continue;
            const auto JR = std::make_pair(iat1, R);
            auto tmp_tensor = RI::Tensor<Tdata>({sizeA, sizeB});
            for (int L1 = 0; L1 != orb_in[T1].size(); ++L1)
            {
                for (int L2 = 0; L2 != orb_in[T2].size(); ++L2)
                {
                    std::vector<double> rly;
                    // keep bohr
                    ModuleBase::Ylm::rl_sph_harm(L1 + L2,
                                                 (delta_R * ucell.lat0).x,
                                                 (delta_R * ucell.lat0).y,
                                                 (delta_R * ucell.lat0).z,
                                                 rly);
                    const double prefactor1 = std::pow(distance, L1 + L2 + 1);
                    for (int M1 = -L1; M1 <= L1; ++M1)
                    {
                        const int index_M1 = M1 + L1;
                        for (int M2 = -L2; M2 <= L2; ++M2)
                        {
                            const int index_M2 = M2 + L2;
                            // (M1 - M2 + L1 + L2) changes to
                            // (M2 - M1 + L1 + L2) (-)^{M1-M2} (-)^{L2+M2}
                            // = (-)^{L2+M1}
                            const double prefactor = std::pow(-1, L2 + M1) / prefactor1;
                            const double clmlm = this->cal_clmlm(L2, M2, L1, M1, MGT0);
                            // (M1 - M2 + L1 + L2) changes to
                            // (M2 - M1 + L1 + L2) (-)^{M1-M2} (-)^{L2+M2}
                            // = (-)^{L2+M1}
                            const double ylm_solid = rly.at(MGT0.get_lm_index(L1 + L2, (M2 - M1 + L1 + L2)));
                            const double ylm_real = (distance > tiny2) ? ylm_solid / pow(distance, L1 + L2) : ylm_solid;
                            const double mom1 = this->multipole[T1][L1][0];
                            const double mom2 = this->multipole[T2][L2][0];
                            // every L has only one moment!=0 after rotation (N=0)
                            const size_t iA = index[T1][L1][0][index_M1];
                            const size_t iB = index[T2][L2][0][index_M2];

                            const double value = prefactor * mom1 * mom2 * clmlm * ylm_real;

                            tmp_tensor(iA, iB) = value;

                            if (iat0 == 0 && iat0 == 0 && R[0] == 2 && R[1] == 2 && R[2] == 1 && iat0 == 0 && iat1 == 0)
                            {
                                std::cout << "iA: " << iA << ", iB: " << iB << ", L2: " << L2 << ", M2: " << M2
                                          << ", V= " << value << ", prefactor:" << prefactor << ",mom1: " << mom1
                                          << ", mom2: " << mom2 << ", clmlm: " << clmlm << ", ylm_real: " << ylm_real
                                          << std::endl;
                            }

                            if (R[0] == 2 && R[1] == 2 && R[2] == 1 && iat0 == 0 && iat1 == 0)
                            {
                                out_pure_ri_tensor("Vs_tensor.txt", tmp_tensor, 0.0);
                            }
                            // Rc: bohr
                            // if (distance >= Rc)
                            //    tmp_tensor(iA, iB) = 0.0;
                            // debug
                            // std::cout << "T1: " << T1 << ", L1: " << L1 << ", T2: " << T2 << ", L2: " << L2
                            //           << ", delta_R: " << delta_R[0] << ", " << delta_R[1] << ", " << delta_R[2]
                            //           << ", distance: " << distance << ", outside V= " << value
                            //           << ", prefactor:" << prefactor << ",mom1: " << mom1 << ", mom2: " << mom2
                            //           << ", clmlm: " << clmlm << ", ylm_real: " << ylm_real << std::endl;
                        }
                    }
                }
            }
            Vws[T1][T2][delta_R] = tmp_tensor;
            // I must contain all atoms in unit cell
            auto& target_inner = Vs_cut.at(iat0);
            if (target_inner.find(JR) == target_inner.end())
            {
                target_inner.emplace(JR, tmp_tensor);
            }
            // otherwise, warning
            else
            {
                target_inner[JR] = tmp_tensor;
                const auto J = JR.first;
                const auto R = JR.second;
                // std::cout << "J=" << J << ", R= " << R[0] << ", " << R[1] << ", " << R[2] << std::endl;
                ModuleBase::WARNING_QUIT("merge_VR", "JR already exists in Vs_cut[I], which is unexpected.");
            }
        }
    }
    ModuleBase::timer::tick("Rotate_abfs", "cal_VR");
}

template <typename Tdata>
void Moment_abfs<Tdata>::out_pure_ri_tensor(const std::string fn, RI::Tensor<double>& olp, const double threshold)
{
    std::ofstream fs;
    auto format = std::scientific;
    int prec = 15;
    fs.open(fn);
    int nr = olp.shape[0];
    int nc = olp.shape[1];
    size_t nnz = nr * nc;
    fs << "%%MatrixMarket matrix coordinate complex general" << std::endl;
    fs << "%" << std::endl;

    fs << nr << " " << nc << " " << nnz << std::endl;

    for (int i = 0; i < nr; i++)
    {
        for (int j = 0; j < nc; j++)
        {
            auto v = olp(i, j);
            if (fabs(v) > threshold)
                fs << i + 1 << " " << j + 1 << " " << std::showpoint << format << std::setprecision(prec) << v << "\n";
        }
    }

    fs.close();
}

template <typename Tdata>
void Moment_abfs<Tdata>::out_pure_ri_tensor(const std::string fn,
                                            RI::Tensor<std::complex<double>>& olp,
                                            const double threshold)
{
    std::ofstream fs;
    auto format = std::scientific;
    int prec = 15;
    fs.open(fn);
    int nr = olp.shape[0];
    int nc = olp.shape[1];
    size_t nnz = nr * nc;
    fs << "%%MatrixMarket matrix coordinate complex general" << std::endl;
    fs << "%" << std::endl;

    fs << nr << " " << nc << " " << nnz << std::endl;

    for (int j = 0; j < nc; j++)
    {
        for (int i = 0; i < nr; i++)
        {
            auto v = olp(i, j);
            if (fabs(v.real()) > threshold || fabs(v.imag()) > threshold)
                fs << i + 1 << " " << j + 1 << " " << std::showpoint << format << std::setprecision(prec) << v.real()
                   << " " << std::showpoint << format << std::setprecision(prec) << v.imag() << "\n";
        }
    }

    fs.close();
}
// template <typename Tdata>
// void Moment_abfs<Tdata>::diverge_list(
//     const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA, std::array<Tcell, Ndim>>>>>&
//     list_As_Vs, std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA, std::array<Tcell, Ndim>>>>>&
//     list_k, std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA, std::array<Tcell, Ndim>>>>>& list_r,
//     const UnitCell& ucell,
//     const std::vector<double>& orb_cutoff)
// {
//     ModuleBase::TITLE("Rotate_abfs", "diverge_list");
//     ModuleBase::timer::tick("Rotate_abfs", "diverge_list");

//     double Rcut_max = 0;
//     for (int T = 0; T < ucell.ntype; ++T)
//         Rcut_max = std::max(Rcut_max, orb_cutoff[T]);

//     list_k.first.clear();
//     list_k.second.clear();
//     list_r.first.clear();
//     list_r.second.clear();
//     list_k.second.resize(1);
//     list_r.second.resize(1);
//     int flag_k = 0;
//     int flag_r = 0;
//     for (size_t iA = 0; iA < list_As_Vs.first.size(); ++iA)
//     {
//         const auto& A = list_As_Vs.first[iA];
//         const size_t TA = ucell.iat2it[A];
//         const size_t IA = ucell.iat2ia[A];
//         const auto& tauA = ucell.atoms[TA].tau[IA];
//         for (const auto& BR: list_As_Vs.second[0])
//         {
//             const auto& B = BR.first;
//             const size_t TB = ucell.iat2it[B];
//             const size_t IB = ucell.iat2ia[B];
//             const auto& tauB = ucell.atoms[TB].tau[IB];
//             const auto& R = BR.second;

//             const ModuleBase::Vector3<double> tauB_shift = tauB + (RI_Util::array3_to_Vector3(R) * ucell.latvec);
//             const ModuleBase::Vector3<double> tau_delta = (tauB_shift - tauA) * ucell.lat0;
//             const double distance = tau_delta.norm();
//             if (distance <= Rcut_max)
//             {
//                 if (std::find(list_k.first.begin(), list_k.first.end(), A) == list_k.first.end())
//                 {
//                     list_k.first.emplace_back(A);
//                 }
//                 if (std::find(list_k.second[0].begin(), list_k.second[0].end(), BR) == list_k.second[0].end())
//                 {
//                     list_k.second[0].emplace_back(BR);
//                 }
//             }
//             else
//             {
//                 if (std::find(list_r.first.begin(), list_r.first.end(), A) == list_r.first.end())
//                 {
//                     list_r.first.emplace_back(A);
//                 }
//                 if (std::find(list_r.second[0].begin(), list_r.second[0].end(), BR) == list_r.second[0].end())
//                 {
//                     list_r.second[0].emplace_back(BR);
//                 }
//             }
//         }
//     }
//     for (const auto& I: list_r.first)
//     {
//         for (const auto& JR: list_r.second[0])
//             flag_r += 1;
//     }
//     for (const auto& I: list_k.first)
//     {
//         for (const auto& JR: list_k.second[0])
//             flag_k += 1;
//     }
//     std::cout << "All No.(atom pairs)=" << flag_k + flag_r << ", " << flag_k << " inside Rcut, " << flag_r
//               << " outside Rcut" << std::endl;
//     ModuleBase::timer::tick("Rotate_abfs", "diverge_list");
// }

// template <typename Tdata>
// void Moment_abfs<Tdata>::merge_list(std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>& Vs_cut)
// {
//     ModuleBase::TITLE("Rotate_abfs", "merge_list");
//     ModuleBase::timer::tick("Rotate_abfs", "merge_list");

//     for (const auto& IJRc: this->VR)
//     {
//         const TA& I = IJRc.first;
//         const auto& JRc = IJRc.second;

//         // if Vs_cut[I] does not exist, then create an empty inner map
//         auto& target_inner = Vs_cut[I];

//         for (const auto& JRc_tensor: JRc)
//         {
//             const TAC& JR = JRc_tensor.first;
//             const RI::Tensor<Tdata>& tensor = JRc_tensor.second;

//             // if JR does not exist in Vs_cut[I], then insert it
//             if (target_inner.find(JR) == target_inner.end())
//             {
//                 target_inner.emplace(JR, tensor);
//             }
//             // otherwise, warning
//             else
//             {
//                 target_inner[JR] = tensor;
//                 const auto J = JR.first;
//                 const auto R = JR.second;
//                 std::cout << "J=" << J << ", R= " << R[0] << ", " << R[1] << ", " << R[2] << std::endl;
//                 // ModuleBase::WARNING_QUIT("merge_VR", "JR already exists in Vs_cut[I], which is unexpected.");
//             }
//         }
//     }
//     ModuleBase::timer::tick("Rotate_abfs", "merge_list");
// }

#endif