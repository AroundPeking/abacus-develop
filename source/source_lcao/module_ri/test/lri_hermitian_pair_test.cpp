#include "../lri_hermitian_pair.h"

#include <cstdlib>
#include <iostream>

namespace
{
struct Vector3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vector3 operator-(const Vector3& value)
{
    return {-value.x, -value.y, -value.z};
}

bool operator==(const Vector3& lhs, const Vector3& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

void require(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_atom_pair_and_partner_share_one_canonical_integral()
{
    const Vector3 displacement{1.25, -0.5, 0.75};
    const auto direct = LRI_CV_Tools::canonical_hermitian_pair(0, 1, displacement);
    const auto partner = LRI_CV_Tools::canonical_hermitian_pair(1, 0, -displacement);

    require(direct.atom_i == partner.atom_i && direct.atom_j == partner.atom_j
                && direct.displacement == partner.displacement,
            "Hermitian partners must select the same integral orientation");
    require(direct.transform_result != partner.transform_result,
            "exactly one Hermitian partner must transform the canonical integral");
}

void test_same_atom_nonzero_partner_uses_one_canonical_integral()
{
    const Vector3 displacement{0.0, 1.5, -0.25};
    const auto direct = LRI_CV_Tools::canonical_hermitian_pair(0, 0, displacement);
    const auto partner = LRI_CV_Tools::canonical_hermitian_pair(0, 0, -displacement);

    require(direct.atom_i == partner.atom_i && direct.atom_j == partner.atom_j
                && direct.displacement == partner.displacement,
            "same-atom R and minus-R requests must select the same integral orientation");
    require(direct.transform_result != partner.transform_result,
            "exactly one same-atom partner must transform the canonical integral");
}

void test_onsite_pair_is_unchanged()
{
    const auto onsite = LRI_CV_Tools::canonical_hermitian_pair(2, 2, Vector3{});
    require(onsite.atom_i == 2 && onsite.atom_j == 2 && onsite.displacement == Vector3{}
                && !onsite.transform_result,
            "an onsite integral must keep its original orientation");
}
} // namespace

int main()
{
    test_atom_pair_and_partner_share_one_canonical_integral();
    test_same_atom_nonzero_partner_uses_one_canonical_integral();
    test_onsite_pair_is_unchanged();
    return EXIT_SUCCESS;
}
