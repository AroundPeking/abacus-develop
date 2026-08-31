#ifndef LRI_HERMITIAN_PAIR_H
#define LRI_HERMITIAN_PAIR_H

namespace LRI_CV_Tools
{
template <typename Vector>
struct HermitianPair
{
    int atom_i = 0;
    int atom_j = 0;
    Vector displacement{};
    bool transform_result = false;
};

template <typename Vector>
bool hermitian_pair_less(const int atom_i_lhs,
                         const int atom_j_lhs,
                         const Vector& displacement_lhs,
                         const int atom_i_rhs,
                         const int atom_j_rhs,
                         const Vector& displacement_rhs)
{
    if (atom_i_lhs != atom_i_rhs)
    {
        return atom_i_lhs < atom_i_rhs;
    }
    if (atom_j_lhs != atom_j_rhs)
    {
        return atom_j_lhs < atom_j_rhs;
    }
    if (displacement_lhs.x != displacement_rhs.x)
    {
        return displacement_lhs.x < displacement_rhs.x;
    }
    if (displacement_lhs.y != displacement_rhs.y)
    {
        return displacement_lhs.y < displacement_rhs.y;
    }
    return displacement_lhs.z < displacement_rhs.z;
}

template <typename Vector>
HermitianPair<Vector> canonical_hermitian_pair(const int atom_i,
                                               const int atom_j,
                                               const Vector& displacement)
{
    const Vector partner_displacement = -displacement;
    if (hermitian_pair_less(atom_j,
                            atom_i,
                            partner_displacement,
                            atom_i,
                            atom_j,
                            displacement))
    {
        return {atom_j, atom_i, partner_displacement, true};
    }
    return {atom_i, atom_j, displacement, false};
}
} // namespace LRI_CV_Tools

#endif
