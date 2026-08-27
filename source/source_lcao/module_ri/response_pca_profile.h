#ifndef RESPONSE_PCA_PROFILE_H
#define RESPONSE_PCA_PROFILE_H

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ResponsePCA
{

using FixedNuProfiles = std::vector<std::vector<std::size_t>>;

inline std::size_t parse_nonnegative_count(const std::string& token)
{
    if (token.empty()
        || !std::all_of(token.begin(), token.end(), [](const unsigned char c) { return std::isdigit(c) != 0; }))
    {
        throw std::invalid_argument("rpa_pca_fixed_nu contains a non-integer radial count.");
    }

    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(token, &consumed);
    if (consumed != token.size() || value > std::numeric_limits<std::size_t>::max())
    {
        throw std::invalid_argument("rpa_pca_fixed_nu radial count is out of range.");
    }
    return static_cast<std::size_t>(value);
}

inline FixedNuProfiles parse_fixed_nu_profiles(const std::string& specification)
{
    if (specification.empty())
    {
        return {};
    }

    FixedNuProfiles profiles;
    std::size_t profile_begin = 0;
    while (profile_begin <= specification.size())
    {
        const std::size_t profile_end = specification.find(';', profile_begin);
        const std::string profile_text = specification.substr(
            profile_begin,
            profile_end == std::string::npos ? std::string::npos : profile_end - profile_begin);
        if (profile_text.empty())
        {
            throw std::invalid_argument("rpa_pca_fixed_nu contains an empty atom-type profile.");
        }

        std::vector<std::size_t> profile;
        std::size_t count_begin = 0;
        while (count_begin <= profile_text.size())
        {
            const std::size_t count_end = profile_text.find(',', count_begin);
            profile.push_back(parse_nonnegative_count(profile_text.substr(
                count_begin,
                count_end == std::string::npos ? std::string::npos : count_end - count_begin)));
            if (count_end == std::string::npos)
            {
                break;
            }
            count_begin = count_end + 1;
        }
        profiles.push_back(std::move(profile));

        if (profile_end == std::string::npos)
        {
            break;
        }
        profile_begin = profile_end + 1;
    }
    return profiles;
}

inline void validate_fixed_nu_profiles(const FixedNuProfiles& profiles,
                                       const std::vector<std::vector<std::size_t>>& available_nu)
{
    if (profiles.empty())
    {
        return;
    }
    if (profiles.size() != available_nu.size())
    {
        throw std::invalid_argument("rpa_pca_fixed_nu must provide exactly one semicolon-separated profile "
                                    "for each atom type.");
    }

    for (std::size_t type = 0; type != profiles.size(); ++type)
    {
        const std::vector<std::size_t>& profile = profiles[type];
        if (profile.size() > available_nu[type].size())
        {
            throw std::invalid_argument("rpa_pca_fixed_nu has more angular-momentum channels than the AO basis.");
        }
        if (std::none_of(profile.begin(), profile.end(), [](const std::size_t count) { return count > 0; }))
        {
            throw std::invalid_argument("Each rpa_pca_fixed_nu atom-type profile must retain at least one radial "
                                        "function.");
        }
        for (std::size_t angular_momentum = 0; angular_momentum != profile.size(); ++angular_momentum)
        {
            if (profile[angular_momentum] > available_nu[type][angular_momentum])
            {
                throw std::invalid_argument("rpa_pca_fixed_nu exceeds the available AO radial functions.");
            }
        }
    }
}

inline bool radial_is_fixed(const std::vector<std::size_t>& fixed_nu,
                            const std::size_t angular_momentum,
                            const std::size_t radial_index)
{
    return angular_momentum < fixed_nu.size() && radial_index < fixed_nu[angular_momentum];
}

inline bool keep_radial_product(const std::vector<std::size_t>& fixed_nu,
                                const std::size_t l1,
                                const std::size_t n1,
                                const std::size_t l2,
                                const std::size_t n2)
{
    return fixed_nu.empty() || radial_is_fixed(fixed_nu, l1, n1) || radial_is_fixed(fixed_nu, l2, n2);
}

inline std::vector<bool> make_fixed_ao_mask(const std::vector<std::size_t>& fixed_nu,
                                            const std::vector<std::size_t>& available_nu)
{
    std::vector<bool> mask;
    for (std::size_t angular_momentum = 0; angular_momentum != available_nu.size(); ++angular_momentum)
    {
        const std::size_t magnetic_components = 2 * angular_momentum + 1;
        for (std::size_t radial_index = 0; radial_index != available_nu[angular_momentum]; ++radial_index)
        {
            const bool fixed = radial_is_fixed(fixed_nu, angular_momentum, radial_index);
            mask.insert(mask.end(), magnetic_components, fixed);
        }
    }
    return mask;
}

inline std::size_t count_kept_ordered_pairs(const std::vector<bool>& fixed_ao_mask)
{
    const std::size_t fixed_count = std::count(fixed_ao_mask.begin(), fixed_ao_mask.end(), true);
    const std::size_t total_count = fixed_ao_mask.size();
    return total_count * total_count - (total_count - fixed_count) * (total_count - fixed_count);
}

} // namespace ResponsePCA

#endif
