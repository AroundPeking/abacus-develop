#ifndef STERNHEIMER_RESPONSE_OPENMP_H
#define STERNHEIMER_RESPONSE_OPENMP_H

namespace ModuleRI
{

class ScopedSternheimerResponseOpenMPThreads
{
  public:
    ScopedSternheimerResponseOpenMPThreads();
    ~ScopedSternheimerResponseOpenMPThreads() noexcept;

    ScopedSternheimerResponseOpenMPThreads(const ScopedSternheimerResponseOpenMPThreads&) = delete;
    ScopedSternheimerResponseOpenMPThreads& operator=(const ScopedSternheimerResponseOpenMPThreads&) = delete;

    int previous_threads() const noexcept;
    int active_threads() const noexcept;

  private:
    int previous_threads_ = 1;
    int active_threads_ = 1;
    int previous_dynamic_ = 0;
};

} // namespace ModuleRI

#endif
