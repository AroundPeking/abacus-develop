//=======================
// AUTHOR : jiyy
// DATE :   2024-03-08
//=======================

#ifndef EWALD_VQ_HPP
#define EWALD_VQ_HPP

template <typename Tdata>
void Ewald_Vq<Tdata>::init(std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& lcaos,
                           std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs,
                           const double& gauss_gamma)
{
    ModuleBase::TITLE("Ewald_Vq", "init");
    ModuleBase::timer::tick("Ewald_Vq", "init");

    this->gamma = gauss_gamma;
    this->g_lcaos = this->init_gauss(lcaos);
    this->g_abfs = this->init_gauss(abfs);

    ModuleBase::timer::tick("Ewald_Vq", "init");
}

template <typename Tdata>
std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> Ewald_Vq<Tdata>::init_gauss(
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in)
{
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> gauss;
    gauss.resize(orb_in.size());
    for (size_t T = 0; T != orb_in.size(); ++T)
    {
        gauss[T].resize(orb_in[T].size());
        for (size_t L = 0; L != orb_in[T].size(); ++L)
        {
            gauss[T][L].resize(orb_in[T][L].size());
            for (size_t N = 0; N != orb_in[T][L].size(); ++N)
            {
                gauss[T][L][N] = this->gaussian_abfs.Gauss(orb_in[T][L][N], this->gamma);
            }
        }
    }

    return gauss;
}

#endif
