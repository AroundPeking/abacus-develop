#ifndef STERNHEIMER_RUNTIME_OPTIONS_H
#define STERNHEIMER_RUNTIME_OPTIONS_H

#include <vector>

namespace ModuleRI
{

struct SternheimerFrequencyRecyclingRuntimeOptions
{
    bool enabled = false;
    int group_size = 3;
    int max_basis_dimension = 48;
};

struct SternheimerFrequencyRecyclingLayout
{
    bool enabled = false;
    std::vector<std::vector<int>> groups;
};

bool sternheimer_environment_flag(const char* name, bool default_value);
int sternheimer_channel_batch_width();
SternheimerFrequencyRecyclingRuntimeOptions sternheimer_frequency_recycling_runtime_options();
SternheimerFrequencyRecyclingLayout make_sternheimer_frequency_recycling_layout(
    const SternheimerFrequencyRecyclingRuntimeOptions& options,
    int frequency_count,
    bool use_delta_sternheimer,
    bool use_frequency_mpi,
    bool use_channel_mpi,
    bool use_global_equation_mpi,
    int channel_batch_width,
    int mpi_ranks,
    int kpoint_groups);

} // namespace ModuleRI

#endif
