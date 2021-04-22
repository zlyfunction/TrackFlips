#include <iostream>
#include <vector>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include "track_flips.h"
#include <fstream>
#include "ConformalSeamlessSimilarityMapping.hh"

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
    std::cout << std::endl;
  }
}
int main(int argc, char *argv[])
{
  Mesh M;
  std::vector<int> _n{-1, 4, 1, -1, 2, 6, 8, -1, 5, -1};
  std::vector<int> _to{0, 1, 0, 2, 2, 1, 3, 1, 2, 3};
  std::vector<int> _f{-1, 0, 0, -1, 0, 1, 1, -1, 1, -1};
  std::vector<int> _h{1, 5};
  std::vector<int> _out{1, 4, 2, 8};
  M.n = _n; M.to = _to; M.f = _f; M.h = _h; M.out = _out;
  M.pts.resize(2);
  
  Eigen::RowVector3d pt;
  pt << 0.1,0.3,0.6;
  M.pts[0].push_back(pt);
  pt << 0.7,0.2,0.1;
  M.pts[0].push_back(pt);
  pt << 0.3,0.4,0.3;
  M.pts[1].push_back(pt);
  for (int i = 0; i < M.pts.size(); i++)
  {
    std::cout << "face " << i << std::endl;
    std::cout << M.v0(M.h[i]) << " " << M.v0(M.n[M.h[i]]) << " " << M.v1(M.n[M.h[i]]) << std::endl;
    std::cout << "pts:\n";
    for (auto pt : M.pts[i])
    {
      std::cout << pt << std::endl;
    }
    std::cout << std::endl;
  }

  std::cout << "do flip\n";
  M.flip_ccw(4);
  // M.flip_ccw(4);
  for (int i = 0; i < M.pts.size(); i++)
  {
    std::cout << "face " << i << std::endl;
    std::cout << M.v0(M.h[i]) << " " << M.v0(M.n[M.h[i]]) << " " << M.v1(M.n[M.h[i]]) << std::endl;
    std::cout << "pts:\n";
    for (auto pt : M.pts[i])
    {
      std::cout << pt << std::endl;
    }
    std::cout << std::endl;
  }
  
  // for (int v : M.n) std::cout << v << " ";
  std::cout << std::endl;
  

  return 0;

  // // Plot the mesh
  // igl::opengl::glfw::Viewer viewer;
  // viewer.data().set_mesh(V, F);
  // // viewer.data().set_face_based(true);
  // viewer.launch();
}
