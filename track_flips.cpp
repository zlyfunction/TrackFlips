#include "track_flips.h"
#include <igl/barycentric_coordinates.h>
void UnfoldTwoTriangles(
  Eigen::RowVector3d &A,
  Eigen::RowVector3d &B,
  Eigen::RowVector3d &C,
  Eigen::RowVector3d &D
)
{
  Eigen::RowVector3d A_new, B_new, C_new, D_new;
  
  auto AB = B - A;
  auto AC = C - A;
  auto AD = D - A;
  
  double l_AB = AB.norm();
  double l_AC = AC.norm();
  double l_AD = AD.norm();

  A_new << 0, 0, 0;
  B_new << 0, l_AB, 0;
  
  double cos_BAC = AB.dot(AC) / l_AB / l_AC;
  double sin_BAC = std::sqrt(1 - cos_BAC * cos_BAC);
  C_new << l_AC * cos_BAC, l_AC * sin_BAC, 0;

  double cos_BAD = AB.dot(AD) / l_AB / l_AD;
  double sin_BAD = std::sqrt(1 - cos_BAD * cos_BAD);
  D_new << l_AD * cos_BAD, -l_AD * sin_BAD, 0;

  A = A_new; B = B_new; C = C_new; D = D_new;
}

void track_flip(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    std::vector<Pt> &pts,
    const FLIP &flip
)
{
    Eigen::RowVector3d A = V.row(flip.A);
    Eigen::RowVector3d B = V.row(flip.B);
    Eigen::RowVector3d C = V.row(flip.C);
    Eigen::RowVector3d D = V.row(flip.D);

    UnfoldTwoTriangles(A, B, C, D);

    for (auto &pt : pts)
    {
        bool flag = false;
        Eigen::RowVector3d P;
        for (int j = 0; j < 3; j++)
        {
            if ((pt.face(j) == flip.A) && 
                (pt.face((j + 1) % 3) == flip.B) && 
                (pt.face((j + 2) % 3) == flip.C))
            {
                flag = true;
                P = pt.bc(j) * A + pt.bc((j + 1) % 3) * B + pt.bc((j + 2) % 3) * C;
                break;
            }
            if ((pt.face(j) == flip.A) && 
                (pt.face((j + 1) % 3) == flip.D) && 
                (pt.face((j + 2) % 3) == flip.B))
            {
                flag = true;
                P = pt.bc(j) * A + pt.bc((j + 1) % 3) * D + pt.bc((j + 2) % 3) * B;
                break;
            }
        }

        if (!flag) continue;

        Eigen::MatrixXd new_bc;

        igl::barycentric_coordinates(P, A, D, C, new_bc);
        if (new_bc.row(0).minCoeff() >= 0)
        {
            pt.face(0) = flip.A; pt.face(1) = flip.B; pt.face(2) = flip.C;
            pt.bc = new_bc.row(0);
        }
        else
        {
            igl::barycentric_coordinates(P, C, D, B, new_bc);
            pt.face(0) = flip.C; pt.face(1) = flip.D; pt.face(2) = flip.B;
            pt.bc = new_bc.row(0);
        }
    }


}



void track_flips(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const std::vector<Pt> &pts,
    const std::vector<FLIP> &flips,
    std::vector<Pt> &pts_out
)
{
    pts_out = pts;
    for (auto one_flip : flips)
    {
        track_flip(V, F, pts_out, one_flip);
    }
}