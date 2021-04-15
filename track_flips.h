#include <igl/barycentric_coordinates.h>
#include <vector>

//      C              C
//     / \            /│\
//    /   \          / │ \
//   /     \        /  │  \
//  A───────B ───► A   │   B
//   \     /        \  │  /
//    \   /          \ │ /
//     \ /            \│/
//      D              D
//
// (ABC),(ADB) -> (ADC),(CDB)

struct Pt
{
    // int f_id;
    Eigen::Vector3i face;
    Eigen::Vector3d bc;

    // Pt &operator=(const Pt &a)
    // {
    //     face = a.face;
    //     bc = a.bc;
    //     return *this;
    // }
};

struct FLIP
{
    int A;
    int B;
    int C;
    int D;
};

void UnfoldTwoTriangles(
    Eigen::RowVector3d &A,
    Eigen::RowVector3d &B,
    Eigen::RowVector3d &C,
    Eigen::RowVector3d &D);

// input: triangle mesh V, F(not necessary)
//        sequence of flips (A, B, C, D)
//        set of Pts(face, bc)
// output: set of Pts on final mesh
void track_flips(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const std::vector<Pt> &pts,
    const std::vector<FLIP> &flips,
    std::vector<Pt> &pts_out);