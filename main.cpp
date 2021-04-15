#include <iostream>
#include <vector>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include "track_flips.h"
#include <fstream>

void readflips(std::vector<FLIP> &flips)
{
  std::ifstream ifs;
  ifs.open("flips.txt");
  int n;
  ifs >> n;
  for (int i = 0; i < n; i++)
  {
    FLIP flip;
    ifs >> flip.A >> flip.B >> flip.C >> flip.D;
    flips.push_back(flip);
  }
  ifs.close();
}

void readpts(const Eigen::MatrixXi &F, std::vector<Pt> &pts)
{
  std::ifstream ifs;
  ifs.open("pts.txt");
  int n;
  ifs >> n;
  for (int i = 0; i < n; i++)
  {
    Pt pt;
    int f_id;
    ifs >> f_id >> pt.bc(0) >> pt.bc(1) >> pt.bc(2);
    pt.face = F.row(f_id);
    pts.push_back(pt);
  }
  ifs.close();
}

void printpts(const Eigen::MatrixXd &V, const std::vector<Pt> &pts)
{
  for (int i = 0; i < pts.size(); i++)
  {
    std::cout << "pt " << i << std::endl;
    std::cout << "face (" << pts[i].face(0) << "," << pts[i].face(1) << "," << pts[i].face(2) << ")";
    std::cout << "\nbc (" << pts[i].bc(0) << "," << pts[i].bc(1) << "," << pts[i].bc(2) << ")";
    std::cout << "\ncoord " << pts[i].bc(0) * V.row(pts[i].face(0)) + pts[i].bc(1) * V.row(pts[i].face(1)) + pts[i].bc(2) * V.row(pts[i].face(2)) << std::endl;
  }
}
int main(int argc, char *argv[])
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::readOBJ("test.obj", V, F);
  std::vector<FLIP> flips;
  std::vector<Pt> pts_in, pts_out;
  readflips(flips);
  readpts(F, pts_in);
  std::cout << "pts before flips" << std::endl;
  printpts(V, pts_in);
  track_flips(V, F, pts_in, flips, pts_out);
  std::cout << "pts after flips" << std::endl;
  printpts(V, pts_out);

  return 0;

  // // Plot the mesh
  // igl::opengl::glfw::Viewer viewer;
  // viewer.data().set_mesh(V, F);
  // // viewer.data().set_face_based(true);
  // viewer.launch();
}
