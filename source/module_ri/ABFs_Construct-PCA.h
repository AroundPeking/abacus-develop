#ifndef ABFS_CONSTRUCT_PCA_H
#define ABFS_CONSTRUCT_PCA_H

#include "../module_basis/module_ao/ORB_atomic_lm.h"

#include <RI/global/Tensor.h>
#include <vector>

//	training data: lcaos[i] * lcaos[j]
//	old basis:     abfs
//	new basis:     to be constructed
//	( all lcaos and abfs on same atom )

namespace ABFs_Construct
{
namespace PCA
{
extern std::vector<std::vector<std::pair<std::vector<double>, RI::Tensor<double>>>> cal_PCA(
    const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& lcaos,
    const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& abfs, // abfs must be orthonormal
    const double kmesh_times);
extern void tensor_dsyev(const char jobz, const char uplo, RI::Tensor<double>& a, double* const w, int& info);
} // namespace PCA
} // namespace ABFs_Construct

#endif // ABFS_CONSTRUCT_PCA_H