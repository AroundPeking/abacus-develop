#include "source_lcao/module_ri/sternheimer_response_openmp.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ModuleRI
{
namespace
{

constexpr const char* kResponseOpenMPThreadsEnv = "ABACUS_STERNHEIMER_RESPONSE_OMP_THREADS";

int response_openmp_threads_from_env(const int default_value)
{
    const char* raw = std::getenv(kResponseOpenMPThreadsEnv);
    if (raw == nullptr)
    {
        return default_value;
    }

    try
    {
        std::size_t parsed = 0;
        const int value = std::stoi(raw, &parsed);
        if (raw[parsed] != '\0' || value <= 0)
        {
            throw std::invalid_argument("not a positive integer");
        }
        return value;
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument(std::string("Invalid positive integer in ")
                                    + kResponseOpenMPThreadsEnv + ".");
    }
}

} // namespace

ScopedSternheimerResponseOpenMPThreads::ScopedSternheimerResponseOpenMPThreads()
{
#ifdef _OPENMP
    previous_threads_ = omp_get_max_threads();
    previous_dynamic_ = omp_get_dynamic();
    const int requested_threads = response_openmp_threads_from_env(previous_threads_);
    omp_set_dynamic(0);
    omp_set_num_threads(requested_threads);
    active_threads_ = omp_get_max_threads();
#endif
}

ScopedSternheimerResponseOpenMPThreads::~ScopedSternheimerResponseOpenMPThreads() noexcept
{
#ifdef _OPENMP
    omp_set_num_threads(previous_threads_);
    omp_set_dynamic(previous_dynamic_);
#endif
}

int ScopedSternheimerResponseOpenMPThreads::previous_threads() const noexcept
{
    return previous_threads_;
}

int ScopedSternheimerResponseOpenMPThreads::active_threads() const noexcept
{
    return active_threads_;
}

} // namespace ModuleRI
