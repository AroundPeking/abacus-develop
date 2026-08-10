#include "source_lcao/module_ri/sternheimer_response_orbital_layout.h"

#include <limits>
#include <stdexcept>

namespace ModuleRI
{

SternheimerResponseOrbitalLayout build_sternheimer_response_orbital_layout(
    const std::vector<SternheimerResponseOrbitalAtomSpec>& atoms)
{
    if (atoms.empty())
    {
        throw std::invalid_argument("Sternheimer response-orbital layout requires atoms.");
    }

    SternheimerResponseOrbitalLayout layout;
    std::size_t sampled_atom_offset = 0;
    for (const SternheimerResponseOrbitalAtomSpec& atom: atoms)
    {
        if (atom.element.empty() || atom.atom_index < 0 || atom.radial_counts.empty())
        {
            throw std::invalid_argument("Sternheimer response-orbital atom specification is incomplete.");
        }

        std::size_t sampled_l_offset = sampled_atom_offset;
        for (std::size_t l_index = 0; l_index != atom.radial_counts.size(); ++l_index)
        {
            const int l = static_cast<int>(l_index);
            const int radial_count = atom.radial_counts[l_index];
            if (radial_count < 0)
            {
                throw std::invalid_argument("Sternheimer response-orbital radial counts must be non-negative.");
            }
            if (radial_count == 0)
            {
                continue;
            }

            const std::size_t magnetic_count = static_cast<std::size_t>(2 * l + 1);
            for (std::size_t magnetic_index = 0; magnetic_index != magnetic_count; ++magnetic_index)
            {
                if (layout.sampled_indices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                {
                    throw std::overflow_error("Sternheimer response-orbital block offset exceeds integer range.");
                }
                layout.blocks.push_back(module_ri::sternheimer_siab::PrimitiveBlock{
                    atom.element,
                    atom.atom_index,
                    l,
                    static_cast<int>(magnetic_index) - l,
                    radial_count,
                    static_cast<int>(layout.sampled_indices.size())});
                for (int radial_index = 0; radial_index != radial_count; ++radial_index)
                {
                    layout.sampled_indices.push_back(sampled_l_offset
                                                     + static_cast<std::size_t>(radial_index) * magnetic_count
                                                     + magnetic_index);
                }
            }
            sampled_l_offset += static_cast<std::size_t>(radial_count) * magnetic_count;
        }
        sampled_atom_offset = sampled_l_offset;
    }
    if (layout.sampled_indices.empty())
    {
        throw std::invalid_argument("Sternheimer response-orbital layout contains no orbitals.");
    }
    return layout;
}

} // namespace ModuleRI
