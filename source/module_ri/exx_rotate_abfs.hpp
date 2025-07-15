#ifndef EXX_ROTATE_ABFS_HPP
#define EXX_ROTATE_ABFS_HPP

#include <vector>

std::vector<std::vector<std::vector<double>>> Moment_abfs::get_multipole(
    const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_in)
{
    std::vector<std::vector<std::vector<double>>> multipole;
    multipole.resize(orb_in.size());
    for (size_t T = 0; T != orb_in.size(); ++T)
    {
        multipole[T].resize(orb_in[T].size());
        for (size_t L = 0; L != orb_in[T].size(); ++L)
        {
            multipole[T][L].resize(orb_in[T][L].size());
            for (size_t N = 0; N != orb_in[T][L].size(); ++N)
            {
                const Numerical_Orbital_Lm& orb_lm = orb_in[T][L][N];
                const int nr = orb_lm.getNr();
                double* integrated_func = new double[nr];
                for (size_t ir = 0; ir != nr; ++ir)
                    integrated_func[ir] = orb_lm.getPsi(ir) * std::pow(orb_lm.getRadial(ir), 2 + L) / (2 * L + 1);

                ModuleBase::Integral::Simpson_Integral(nr, integrated_func, orb_lm.getRab(), multipole[T][L][N]);
            }
        }
    }

    return multipole;
}