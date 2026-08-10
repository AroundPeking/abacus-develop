#ifndef STERNHEIMER_RESPONSE_ORBITAL_LAYOUT_H
#define STERNHEIMER_RESPONSE_ORBITAL_LAYOUT_H

#include "source_lcao/module_ri/sternheimer_siab_data.h"

#include <cstddef>
#include <string>
#include <vector>

namespace ModuleRI
{

struct SternheimerResponseOrbitalAtomSpec
{
    std::string element;
    int atom_index = -1;
    std::vector<int> radial_counts;
};

struct SternheimerResponseOrbitalLayout
{
    std::vector<std::size_t> sampled_indices;
    std::vector<module_ri::sternheimer_siab::PrimitiveBlock> blocks;
};

SternheimerResponseOrbitalLayout build_sternheimer_response_orbital_layout(
    const std::vector<SternheimerResponseOrbitalAtomSpec>& atoms);

} // namespace ModuleRI

#endif
