#include "source_lcao/module_ri/sternheimer_runtime_options.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

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

} // namespace ModuleRI
