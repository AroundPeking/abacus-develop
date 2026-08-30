#ifndef STERNHEIMER_RUNTIME_OPTIONS_H
#define STERNHEIMER_RUNTIME_OPTIONS_H

namespace ModuleRI
{

struct SternheimerFrequencyRecyclingRuntimeOptions
{
    bool enabled = false;
    int group_size = 3;
    int max_basis_dimension = 48;
};

bool sternheimer_environment_flag(const char* name, bool default_value);
int sternheimer_channel_batch_width();
SternheimerFrequencyRecyclingRuntimeOptions sternheimer_frequency_recycling_runtime_options();

} // namespace ModuleRI

#endif
