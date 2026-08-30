#include "source_lcao/module_ri/sternheimer_runtime_options.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ModuleRI
{
namespace
{

int positive_environment_integer(const char* name,
                                 const int default_value,
                                 const int maximum_value,
                                 const char* description)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr)
    {
        return default_value;
    }

    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (errno != 0 || end == raw || end == nullptr || *end != '\0' || parsed <= 0
        || parsed > maximum_value || parsed > std::numeric_limits<int>::max())
    {
        throw std::invalid_argument("Invalid " + std::string(description) + " in " + name + ": " + raw);
    }
    return static_cast<int>(parsed);
}

} // namespace

bool sternheimer_environment_flag(const char* name, const bool default_value)
{
    if (name == nullptr || name[0] == '\0')
    {
        throw std::invalid_argument("Sternheimer environment flag name must not be empty.");
    }

    const char* raw = std::getenv(name);
    if (raw == nullptr)
    {
        return default_value;
    }

    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    if (value == "1" || value == "true" || value == "on" || value == "yes")
    {
        return true;
    }
    if (value == "0" || value == "false" || value == "off" || value == "no")
    {
        return false;
    }

    throw std::invalid_argument("Invalid boolean value for " + std::string(name) + ": " + value);
}

int sternheimer_channel_batch_width()
{
    constexpr const char* name = "ABACUS_STERNHEIMER_CHANNEL_BATCH_WIDTH";
    constexpr int default_width = 2;
    constexpr int maximum_width = 64;
    return positive_environment_integer(name, default_width, maximum_width, "channel batch width");
}

SternheimerFrequencyRecyclingRuntimeOptions sternheimer_frequency_recycling_runtime_options()
{
    SternheimerFrequencyRecyclingRuntimeOptions options;
    options.enabled
        = sternheimer_environment_flag("ABACUS_STERNHEIMER_FREQUENCY_RECYCLING", false);
    options.group_size = positive_environment_integer(
        "ABACUS_STERNHEIMER_FREQUENCY_RECYCLING_GROUP_SIZE", 3, 64, "frequency recycling group size");
    options.max_basis_dimension = positive_environment_integer(
        "ABACUS_STERNHEIMER_FREQUENCY_RECYCLING_MAX_BASIS_DIMENSION",
        48,
        512,
        "frequency recycling maximum basis dimension");
    return options;
}

SternheimerFrequencyRecyclingLayout make_sternheimer_frequency_recycling_layout(
    const SternheimerFrequencyRecyclingRuntimeOptions& options,
    const int frequency_count,
    const bool use_delta_sternheimer,
    const bool use_frequency_mpi,
    const bool use_channel_mpi,
    const bool use_global_equation_mpi,
    const int channel_batch_width,
    const int mpi_ranks,
    const int kpoint_groups)
{
    if (frequency_count <= 0)
    {
        throw std::invalid_argument("Sternheimer frequency count must be positive.");
    }

    SternheimerFrequencyRecyclingLayout layout;
    if (!options.enabled)
    {
        layout.groups.reserve(static_cast<std::size_t>(frequency_count));
        for (int frequency = 0; frequency != frequency_count; ++frequency)
        {
            layout.groups.push_back({frequency});
        }
        return layout;
    }

    const bool compatible_layout
        = use_delta_sternheimer && !use_frequency_mpi && !use_channel_mpi
          && !use_global_equation_mpi && channel_batch_width == 1 && mpi_ranks > 0
          && kpoint_groups == mpi_ranks && options.group_size >= 2
          && frequency_count >= options.group_size
          && frequency_count % options.group_size == 0;
    if (!compatible_layout)
    {
        throw std::invalid_argument(
            "Sternheimer frequency recycling requires Delta-ST, frequency/channel/global-equation "
            "MPI disabled, channel batch width 1, one k-point group per MPI rank, and a frequency "
            "count divisible by the recycling group size.");
    }

    layout.enabled = true;
    layout.groups.reserve(static_cast<std::size_t>(frequency_count / options.group_size));
    for (int begin = 0; begin != frequency_count; begin += options.group_size)
    {
        std::vector<int> group;
        group.reserve(static_cast<std::size_t>(options.group_size));
        for (int offset = 0; offset != options.group_size; ++offset)
        {
            group.push_back(begin + offset);
        }
        layout.groups.push_back(std::move(group));
    }
    return layout;
}

} // namespace ModuleRI
