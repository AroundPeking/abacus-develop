#ifndef MODULE_DELTA_ST_STERNHEIMER_LAPACK_GESV_H
#define MODULE_DELTA_ST_STERNHEIMER_LAPACK_GESV_H

// The Delta-Sternheimer solver needs the complex general linear solver
// ZGESV, which master_ghj declares in
// source_base/module_external/lapack_connector.h.  The base branch of this
// worktree does not declare it there yet, so the declaration is kept local
// to module_delta_st to avoid depending on an unrelated core header change.
//
// When lapack_connector.h gains the upstream ZGESV declaration, this header
// (and the two includes of it) can be dropped without changing behaviour:
// the declaration below matches the upstream one exactly.

#include <complex>

extern "C"
{
void zgesv_(const int* n,
            const int* nrhs,
            std::complex<double>* a,
            const int* lda,
            int* ipiv,
            std::complex<double>* b,
            const int* ldb,
            int* info);
}

#endif
