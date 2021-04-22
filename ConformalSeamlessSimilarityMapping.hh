// Author: Marcel Campen
//
// v0.1
//
// class ConformalSeamlessSimilarityMapping implements the degeneracy-flip algorithm from the 2017 Seamless Similarity Maps paper.
// class ConformalMetricDelaunay implements the new Delaunay-flip version from the 2021 paper draft.
//
// both only support meshes without boundary.
// ConformalSeamlessSimilarityMapping only supports genus 0 for comparison purposes (for higher genus it requires specifying global turning numbers, which ConformalMetricDelaunay does not support).
// ConformalMetricDelaunay supports arbitrary genus.
//
// both use class Mesh as underlying halfedge data structure
//
// with USE_MPFR one can switch between standard double and MPFR number types - BUT: ConformalSeamlessSimilarityMapping only supports double.
// with MPFR_PREC the precision (in mantissa bits) can be set.
//
// after computing the metric, one may call compute_layout() to compute (u,v)-coordinates. This is generally done with double numbers.
//
#ifndef CONFORMAL_HH
#define CONFORMAL_HH

// #include <unsupported/Eigen/MPRealSupport>
#include <Eigen/Sparse>
#include <set>
#include <queue>
#include <vector>
#include <iostream>
#include <igl/barycentric_coordinates.h>
// #define USE_MPFR
#ifdef USE_MPFR
  #define TODOUBLE .toDouble()
  #define MPFR_PREC 100
#else
  #define TODOUBLE
  using std::min;
  using std::max;
#endif


class Mesh {

#ifdef USE_MPFR
  typedef mpfr::mpreal Scalar;
#else
  typedef double Scalar;
#endif

public:
  std::vector<int> n; // next halfedge of halfedge
  std::vector<int> to; // to vertex of halfedge
  std::vector<int> f; // face of halfedge
  std::vector<int> h; // one halfedge of face
  std::vector<int> out; // one outgoing halfedge of vertex
  std::vector<char> type; // for symmetric double cover
  std::vector<int> R; // reflection map for halfedges

  std::vector<Scalar> l; // discrete metric (length per edge)

  std::vector< std::vector< Eigen::Matrix<Scalar, 1, 3> > > pts; //pts per face 
  
  int n_halfedges() { return n.size(); }
  int n_edges() { return n_halfedges()/2; }
  int n_faces() { return h.size(); }
  int n_vertices() { return out.size(); }

  int e(int h) { return h/2; }
  int opp(int h) { return (h%2 == 0) ? (h+1) : (h-1); }
  int v0(int h) { return to[opp(h)]; }
  int v1(int h) { return to[h]; }
  int h0(int e) { return e*2; }
  int h1(int e) { return e*2+1; }
  double sign(int h) { return (h%2 == 0) ? 1.0 : -1.0; }
  
  Mesh()
  {
    #ifdef USE_MPFR
      mpfr::mpreal::set_default_prec(MPFR_PREC);
    #endif
  }
  
  virtual void init() { };

  virtual void recompute_bc(int _h)
  {
    int ha = _h;
    int hb = opp(_h);
    int f0 = f[ha];
    int f1 = f[hb];

    // unit length
    Eigen::Matrix<Scalar, 1, 3> A, B, C, D;
    A << 0,0,0;
    B << 1,0,0;
    C << 0.5, sqrt(3)/2, 0;
    D << 0.5, -sqrt(3)/2, 0;

    int it0 = 0, it1 = 0;
    int h_tmp = h[f0];
    while (h_tmp != ha)
    {
      h_tmp = n[h_tmp];
      it0++;
    }
    h_tmp = h[f1];
    while (h_tmp != hb)
    {
      h_tmp = n[h_tmp];
      it1++;
    }
    std::vector<Eigen::Matrix<Scalar,1,3>> pts_new_f0, pts_new_f1;
    Eigen::Matrix<Scalar, -1, -1> new_bc;
    for (auto pt : pts[f0])
    {
      auto P = pt(it0) * A + pt((it0 + 1) % 3) * B + pt((it0 + 2) % 3) * C;
      if (pt(it0) > pt((it0 + 1) % 3))
      {
        igl::barycentric_coordinates(P, A, D, C, new_bc);
        pts_new_f0.push_back(new_bc.row(0));
      }
      else
      {
        igl::barycentric_coordinates(P, B, C, D, new_bc);
        pts_new_f1.push_back(new_bc.row(0));
      }
    }

    for (auto pt : pts[f1])
    {
      auto P = pt(it1) * B + pt((it1 + 1) % 3) * A + pt((it1 + 2) % 3) * D;
      if (pt(it1) < pt((it1 + 1) % 3))
      {
        igl::barycentric_coordinates(P, A, D, C, new_bc);
        pts_new_f0.push_back(new_bc.row(0));
      }
      else
      {
        igl::barycentric_coordinates(P, B, C, D, new_bc);
        pts_new_f1.push_back(new_bc.row(0));
      }
    }

    pts[f0] = pts_new_f0;
    pts[f1] = pts_new_f1;
    
  }
  
  virtual bool flip_ccw(int _h)
  {
    int ha = _h;
    int hb = opp(_h);
    int f0 = f[ha];
    int f1 = f[hb];
    if(f0 == f1) return false;
    int h2 = n[ha];
    int h3 = n[h2];
    int h4 = n[hb];
    int h5 = n[h4];
    // recompute the bc
    recompute_bc(_h);
    
    // do flip
    out[to[hb]] = h4;
    out[to[ha]] = h2;
    f[h4] = f0;
    f[h2] = f1;
    h[f0] = h4;
    h[f1] = h2;
    to[ha] = to[h2];
    to[hb] = to[h4];
    n[h5] = h2;
    n[h3] = h4;
    n[h2] = hb;
    n[h4] = ha;
    n[ha] = h3;
    n[hb] = h5;
    return true;
  }
  
  virtual void get_mesh(std::vector<int>& _n, // next halfedge of halfedge
                        std::vector<int>& _to, // to vertex of halfedge
                        std::vector<int>& _f, // face of halfedge
                        std::vector<int>& _h, // one halfedge of face
                        std::vector<int>& _out) // one outgoing halfedge of vertex
  {
    _n = n;
    _to = to;
    _f = f;
    _h = h;
    _out = out;
  }
  
  template<typename T>
  std::vector<T> interpolate(const std::vector<T>& u)
  {
    return u;
  }
  
  bool is_complex()
  {
    int nh = n_halfedges();
    for(int i = 0; i < nh; i++)
    {
      if(to[i] == to[opp(i)]) return true; //contains loop edge
    }
    int nv = n_vertices();
    for(int i = 0; i < nv; i++)
    {
      std::set<int> onering;
      int h = out[i];
      if(h < 0) continue;
      int k = h;
      do {
        int v = to[k];
        if(onering.find(v) != onering.end()) return true; //contains multi-edges
        onering.insert(v);
        k = n[opp(k)];
      } while(k != h);
    }
    return false;
  }
};


class ConformalSeamlessSimilarityMapping {
public:

  Mesh& m;
  
  std::vector<double> Theta_hat; //target cone angles per vertex
  std::vector<double> kappa_hat; //target holonomy angles per gamma loop
  std::vector< std::vector<int> > gamma; //directed dual loops, represented by halfedges (the ones adjacent to the earlier triangles in the dual loop)

  const double cot_infty = 1e10;

  int n_s;
  int n_e;
  int n_h;
  int n_f;
  int n_v;

  std::vector<double> xi;
  std::vector<double> delta_xi;
  std::vector<double> cot_alpha;
  std::vector<double> alpha;

  Eigen::SparseMatrix<double> A;
  Eigen::VectorXd b;

  ConformalSeamlessSimilarityMapping(Mesh& _m, const std::vector<double>& _Theta_hat, const std::vector<double>& _kappa_hat, std::vector< std::vector<int> >& _gamma) : m(_m), Theta_hat(_Theta_hat), kappa_hat(_kappa_hat), gamma(_gamma)
  {
    n_s = gamma.size();
    n_e = m.n_edges();
    n_h = m.n_halfedges();
    n_f = m.n_faces();
    n_v = m.n_vertices();

    xi.resize(n_h, 0.0);
    delta_xi.resize(n_h, 0.0);
    cot_alpha.resize(n_h);
    alpha.resize(n_h);
  }
  
  void log(const char* c)
  {
    std::cout << c << std::endl;
  }
  
  void compute_angles() // compute alpha and cot_alpha from scaled edge lengths
  {
    #pragma omp parallel for
    for(int f = 0; f < n_f; f++)
    {
      int hi = m.h[f];
      int hj = m.n[hi];
      int hk = m.n[hj];
      // (following "On Discrete Conformal Seamless Similarity Maps")
      double li = m.l[m.e(hi)]TODOUBLE * std::exp(1.0/6.0*(xi[hk]-xi[hj]));
      double lj = m.l[m.e(hj)]TODOUBLE * std::exp(1.0/6.0*(xi[hi]-xi[hk]));
      double lk = m.l[m.e(hk)]TODOUBLE * std::exp(1.0/6.0*(xi[hj]-xi[hi]));
      // (following "A Cotangent Laplacian for Images as Surfaces")
      double s = (li+lj+lk)/2.0;
      double Aijk4 = 4.0*std::sqrt(std::max(0.0, s*(s-li)*(s-lj)*(s-lk)));
      double Ijk = (-li*li+lj*lj+lk*lk);
      double iJk = (li*li-lj*lj+lk*lk);
      double ijK = (li*li+lj*lj-lk*lk);
      cot_alpha[hi] = Aijk4 == 0.0 ? copysign(cot_infty,Ijk) : (Ijk/Aijk4);
      cot_alpha[hj] = Aijk4 == 0.0 ? copysign(cot_infty,iJk) : (iJk/Aijk4);
      cot_alpha[hk] = Aijk4 == 0.0 ? copysign(cot_infty,ijK) : (ijK/Aijk4);
      
      alpha[hi] = std::acos(std::min(1.0, std::max(-1.0, Ijk/(2.0*lj*lk))));
      alpha[hj] = std::acos(std::min(1.0, std::max(-1.0, iJk/(2.0*lk*li))));
      alpha[hk] = std::acos(std::min(1.0, std::max(-1.0, ijK/(2.0*li*lj))));
    }
  }

  void setup_b() // system right-hand sid
  {
    b.resize(n_v-1 + n_s + n_f-1);
    b.fill(0.0);
    
    std::vector<double> Theta(n_v, 0.0);
    std::vector<double> kappa(n_s, 0.0);
    
    for(int h = 0; h < n_h; h++)
    {
      Theta[m.to[m.n[h]]] += alpha[h];
    }
    #pragma omp parallel for
    for(int r = 0; r < n_v-1; r++)
    {
      b[r] = Theta_hat[r] - Theta[r];
    }
    #pragma omp parallel for
    for(int s = 0; s < n_s; s++)
    {
      kappa[s] = 0.0;
      int loop_size = gamma[s].size();
      for(int si = 0; si < loop_size; si++)
      {
        int h = gamma[s][si];
        int hn = m.n[h];
        int hnn = m.n[hn];
        if(m.opp(hn) == gamma[s][(si+1)%loop_size])
          kappa[s] -= alpha[hnn];
        else if(m.opp(hnn) == gamma[s][(si+1)%loop_size])
          kappa[s] += alpha[hn];
        else std::cerr << "ERROR: loop is broken" << std::endl;
      }
      b[n_v-1+s] = kappa_hat[s] - kappa[s];
    }
  }
  
  void setup_A() // system matrix
  {
    A.resize(n_v-1 + n_s + n_f-1, n_e);
    int loop_trips = 0;
    for(int i = 0; i < n_s; i++)
      loop_trips += gamma[i].size();
    
    typedef Eigen::Triplet<double> Trip;
    std::vector<Trip> trips;
    trips.clear();
    trips.resize(n_h*2 + loop_trips + (n_f-1)*3);
    #pragma omp parallel for
    for(int h = 0; h < n_h; h++)
    {
      int v0 = m.v0(h);
      int v1 = m.v1(h);
      if(v0 < n_v-1) trips[h*2] = Trip(v0, m.e(h), m.sign(h)*0.5*cot_alpha[h]);
      if(v1 < n_v-1) trips[h*2+1] = Trip(v1, m.e(h), -m.sign(h)*0.5*cot_alpha[h]);
    }
    
    int base = n_h*2;
    for(int s = 0; s < n_s; s++)
    {
      int loop_size = gamma[s].size();
      #pragma omp parallel for
      for(int si = 0; si < loop_size; si++)
      {
        int h = gamma[s][si];
        trips[base+si] = Trip(n_v-1 + s, m.e(h), m.sign(h)*0.5*(cot_alpha[h]+cot_alpha[m.opp(h)]));
      }
      base += loop_size;
    }
    
    #pragma omp parallel for
    for(int f = 0; f < n_f-1; f++)
    {
      int hi = m.h[f];
      int hj = m.n[hi];
      int hk = m.n[hj];
      trips[base+f*3] = Trip(n_v-1 + n_s + f, m.e(hi), m.sign(hi));
      trips[base+f*3+1] = Trip(n_v-1 + n_s + f, m.e(hj), m.sign(hj));
      trips[base+f*3+2] = Trip(n_v-1 + n_s + f, m.e(hk), m.sign(hk));
    }
    
    A.setFromTriplets(trips.begin(), trips.end());
  }
  
  double I(int i, int j, int k, double lambda = 0.0)
  {
    return m.l[m.e(i)]TODOUBLE*std::exp((-xi[j]-delta_xi[j]*lambda)/2) + m.l[m.e(j)]TODOUBLE*std::exp((xi[i]+delta_xi[i]*lambda)/2) - m.l[m.e(k)]TODOUBLE;
  }

  double firstDegeneracy(int& degen, double lambda)
  {
    bool repeat = true;
    while(repeat)
    {
      repeat = false;
      #pragma omp parallel for
      for(int i = 0; i < n_h; i++)
      {
        int j = m.n[i];
        int k = m.n[j];
        double local_lambda = lambda;
        if(I(i,j,k,local_lambda) < 0.0)
        {
          // root finding (from below) by bracketing bisection
          double lo = 0.0;
          double hi = local_lambda;
          for(int r = 0; r < 100; r++)
          {
            double mid = (lo+hi)*0.5;
            if(I(i,j,k,mid) <= 0.0)
              hi = mid;
            else
              lo = mid;
          }
          
          #pragma omp critical
          {
            if(lo < lambda)
            {
              lambda = lo;
              degen = k;
              repeat = true;
            }
          }
        }
      }
    }
    return lambda;
  }
  
  double avg_abs(const Eigen::VectorXd& v)
  {
    double res = 0.0;
    int v_size = v.size();
    for(int i = 0; i < v_size; i++)
      res += std::abs(b[i]);
    return res/v_size;
  }
  
  double max_abs(const Eigen::VectorXd& v)
  {
    double res = 0.0;
    int v_size = v.size();
    for(int i = 0; i < v_size; i++)
      res = std::max(res, std::abs(b[i]));
    return res;
  }
  
  void compute_metric()
  {
    double eps = 1e-12; //if max curvature error below eps: consider converged
    int max_iter = 25; //max full Newton steps
    bool converged = false;
    
    log("computing angles");
    compute_angles();
    log("setup b");
    setup_b();
    
    std::vector< std::pair<double,double> > errors;
    int n_flips = 0;
    
    log("starting Newton");
    int degen = -1;
    while(!converged && max_iter > 0)
    {
      double error = max_abs(b);
      if(degen < 0) errors.push_back( std::pair<double,double>(avg_abs(b), error) );
      if(error <= eps) { converged = true; break; }
      
      double diff = avg_abs(b);
      
      log("setup A");
      setup_A();
      
      log("factorize A");
      Eigen::SparseLU< Eigen::SparseMatrix<double> > chol(A);
      log("solve Ax=b");
      Eigen::VectorXd result = chol.solve(b);
      if(chol.info() != Eigen::Success) { log("factorization failed"); return; }
      log("solved");
      
      #pragma omp parallel for
      for(int i = 0; i < n_e; i++)
      {
        delta_xi[i*2] = result[i];
        delta_xi[i*2+1] = -result[i];
      }
      
      log("line search");
      double lambda = 1.0;
      
      int max_linesearch = 25;
      degen = -1;
      while(true) // line search
      {
        log("  checking for degeneration events");
        double first_degen = firstDegeneracy(degen, lambda);
        if(first_degen < lambda)
        {
          lambda = first_degen;
          std::cout << "    degeneracy at lambda = " << lambda << std::endl;
        }
        
        log("  checking for improvement");
        std::vector<double> xi_old = xi;
        
        #pragma omp parallel for
        for(int i = 0; i < n_h; i++)
          xi[i] = xi_old[i] + lambda * delta_xi[i];
        
        compute_angles();
        setup_b();
        
        if(lambda == 0.0)
        {
          converged = true;
          break;
        }
        
        double new_diff = avg_abs(b);
        if(new_diff < diff)
        {
          std::cout << "    OK. (" << diff << " -> " << new_diff << ")" << std::endl;
          break;
        }
        
        lambda *= 0.5;
        if(max_linesearch-- == 0) lambda = 0.0;
        std::cout << "    reduced to    lambda = " << lambda << std::endl;
      }
      
      if(degen < 0) max_iter--; //no degeneration event
      
      if(!converged) //flip edge(s) of degeneracy/ies
      {
        std::set<int> degens;
        if(degen >= 0) degens.insert(m.e(degen));
        #pragma omp parallel for
        for(int i = 0; i < n_h; i++) //check for additional (simultaneous) degeneracies
        {
          int j = m.n[i];
          int k = m.n[j];
          if(I(i,j,k) <= 0.0)
          {
            #pragma omp critical
            {
              degens.insert(m.e(k));
            }
          }
        }
        int n_d = degens.size();
        if(n_d == 1) std::cout << "handling a degeneracy by edge flip" << std::endl;
        else if(n_d > 1) std::cout << "handling " << degens.size() << " degeneracies by edge flips" << std::endl;
        for(std::set<int>::iterator it = degens.begin(); it != degens.end(); it++)
        {
          int e = *it;
          int h = m.h0(e);
          int hl = m.n[h];
          int hr = m.n[m.n[m.opp(h)]];
          
          int hlu = m.n[hl];
          int hru = m.n[m.n[hr]];
          int ho = m.opp(h);
          int hu = h;
          
          double angle = alpha[m.n[hl]]+alpha[m.n[m.opp(h)]];
          double a = m.l[m.e(hl)]TODOUBLE * std::exp(xi[hl]/2);
          double b = m.l[m.e(hr)]TODOUBLE * std::exp(-xi[hr]/2);
          m.l[e] = std::sqrt(a*a + b*b - 2.0*a*b*std::cos(angle)) / std::exp((xi[hl]-xi[hr])/2); //intrinsic flip (law of cosines)
          
          xi[h] = xi[hl]+xi[hr];
          xi[m.opp(h)] = -xi[h];
          
          if(!m.flip_ccw(h)) { std::cerr << "ERROR: edge could not be flipped." << std::endl; converged = true; break; };
          n_flips++;
          
          if(m.l[e] <= 0.0)
          {
            m.l[e] = 1e-20;
            std::cerr << "WARNING: numerical issue: flipped edge had zero length.";
          }
          
          // adjust gamma loops that contain the flipped edge e
          #pragma omp parallel for
          for(int i = 0; i < n_s; ++i)
          {
            std::vector<int>& li = gamma[i];
            int n = li.size();
            for(int j = 0; j < n; ++j)
            {
              int hij = li[j];
              int hij1 = li[(j+1)%n];
              int hij2 = li[(j+2)%n];
              
              bool change = true;
              
              if(hij == hru && hij1 == m.opp(hr)) li.insert(li.begin()+j+1,ho);
              else if(hij == hru && hij1 == hu && hij2 == m.opp(hl)) li[(j+1)%n] = ho;
              else if(hij == hru && hij1 == hu && hij2 == m.opp(hlu)) li.erase(li.begin()+((j+1)%n));
              
              else if(hij == hr && hij1 == m.opp(hru)) li.insert(li.begin()+j+1,hu);
              else if(hij == hr && hij1 == hu && hij2 == m.opp(hlu)) li[(j+1)%n] = hu;
              else if(hij == hr && hij1 == hu && hij2 == m.opp(hl)) li.erase(li.begin()+((j+1)%n));
              
              else if(hij == hl && hij1 == m.opp(hlu)) li.insert(li.begin()+j+1,hu);
              else if(hij == hl && hij1 == ho && hij2 == m.opp(hru)) li[(j+1)%n] = hu;
              else if(hij == hl && hij1 == ho && hij2 == m.opp(hr)) li.erase(li.begin()+((j+1)%n));
              
              else if(hij == hlu && hij1 == m.opp(hl)) li.insert(li.begin()+j+1,ho);
              else if(hij == hlu && hij1 == ho && hij2 == m.opp(hr)) li[(j+1)%n] = ho;
              else if(hij == hlu && hij1 == ho && hij2 == m.opp(hru)) li.erase(li.begin()+((j+1)%n));
              
              else change = false;
              
              if(change) // cleanup "cusps" in loop
              {
                n = li.size();
                int j0 = j;
                int j1 = (j+1)%n;
                int j2 = (j+2)%n;
                if(li[j0] == m.opp(li[j1]))
                {
                  if(j1 < j0) std::swap(j0,j1);
                  li.erase(li.begin()+j1);
                  li.erase(li.begin()+j0);
                }
                else if(li[j1] == m.opp(li[j2]))
                {
                  if(j2 < j1) std::swap(j1,j2);
                  li.erase(li.begin()+j2);
                  li.erase(li.begin()+j1);
                }
              }
            }
          }
        }
        if(n_d > 0) //recompute angles after flipping
        {
          compute_angles();
          setup_b(); 
          
          //sanity check
          for(int i = 0; i < n_h; i++)
          {
            int j = m.n[i];
            int k = m.n[j];
            double indicator = I(i,j,k);
            if(indicator <= 0.0)
            {
              #pragma omp critical
              {
                if(indicator == 0.0) std::cerr << "WARNING: numerical issue: triangle("<<i<<", "<<j<<", "<<k<<") is degenerate after Newton step." << std::endl;
                if(indicator < 0.0) std::cerr << "ERROR: numerical issue: triangle("<<i<<", "<<j<<", "<<k<<") is violating after Newton step." << std::endl;
                degens.insert(m.e(k));
              }
            }
          }
        }
      }
    }
    
    double error = max_abs(b);
    
    if(error > eps) std::cerr << "WARNING: the final max error is larger than desired ("<<error<<")." << std::endl;
    
    std::cout << "\nSTATISTICS:\n";
    std::cout << "Flips: " << n_flips << std::endl;
    std::cout << "Error Decay: (iter, avg, max)" << std::endl;
    for(size_t i = 0; i < errors.size(); i++)
    {
      std::cout << i << ": " << errors[i].first << "  " << errors[i].second << std::endl;
    }
    std::cout << std::endl;
    if(n_flips > 0)
      std::cout << "HINT: The given mesh m has been modified by edge flips. Get the modified mesh by m.get_mesh(...)" << std::endl;
    if(n_flips > 0 && m.is_complex())
      std::cout << "HINT: The modified mesh is non-simple (e.g. contains a loop edge or multiple edges between a pair of vertices). Beware that many mesh data structures and libraries do not support this appropriately." << std::endl;
    std::cout << std::endl;
  }


  void compute_layout(std::vector<double>& u, std::vector<double>& v) //metric -> parametrization
  {
    std::vector<double> phi(n_h);
    
    u.resize(n_h);
    v.resize(n_h);
    
    //set starting point
    int h = 0;
    phi[h] = 0.0;
    u[h] = 0.0;
    v[h] = 0.0;
    h = m.n[h];
    phi[h] = xi[h];
    u[h] = m.l[m.e(h)]TODOUBLE*std::exp(phi[h]/2);
    v[h] = 0.0;
    
    // layout the rest of the mesh by BFS
    std::vector<bool> visited(n_f, false);
    std::queue<int> q;
    q.push(h);
    visited[m.f[h]] = true;
    while(!q.empty())
    {
      h = q.front();
      q.pop();
      
      int hn = m.n[h];
      int hp = m.n[hn];
      
      phi[hn] = phi[h] + xi[hn];
      
      double len = m.l[m.e(hn)]TODOUBLE * std::exp((phi[h]+phi[hn])/2);
      
      double ud = u[hp]-u[h];
      double vd = v[hp]-v[h];
      double d = std::sqrt(ud*ud + vd*vd);
      double co = std::cos(alpha[hp]);
      double si = std::sin(alpha[hp]);
      
      u[hn] = u[h] + (co*ud + si*vd)*len/d;
      v[hn] = v[h] + (co*vd - si*ud)*len/d;
      
      int hno = m.opp(hn);
      int hpo = m.opp(hp);
      if(!visited[m.f[hno]])
      {
        visited[m.f[hno]] = true;
        phi[hno] = phi[h];
        phi[m.n[m.n[hno]]] = phi[hn];
        u[hno] = u[h];
        v[hno] = v[h];
        u[m.n[m.n[hno]]] = u[hn];
        v[m.n[m.n[hno]]] = v[hn];
        q.push(hno);
      }
      if(!visited[m.f[hpo]])
      {
        visited[m.f[hpo]] = true;
        phi[hpo] = phi[hn];
        phi[m.n[m.n[hpo]]] = phi[hp];
        u[hpo] = u[hn];
        v[hpo] = v[hn];
        u[m.n[m.n[hpo]]] = u[hp];
        v[m.n[m.n[hpo]]] = v[hp];
        q.push(hpo);
      }
    }
  }

  void compute(std::vector<double>& u, std::vector<double>& v) //main method
  {
    compute_metric();
    compute_layout(u, v);
  }

};





class ConformalMetricDelaunay {
public:

#ifdef USE_MPFR
  typedef mpfr::mpreal Scalar;
#else
  typedef double Scalar;
#endif

  typedef Eigen::Matrix<Scalar,Eigen::Dynamic,1> VectorX;

  Mesh& m;
  
  VectorX Theta_hat; //target cone angles per vertex
  
  const Scalar cot_infty = Scalar(1e10);

  int n_e;
  int n_h;
  int n_f;
  int n_v;

  VectorX u;
  VectorX cot_alpha;
  VectorX alpha;
  
  Eigen::SparseMatrix<Scalar> H_;
  
  int n_flips, n_flips_s, n_flips_t, n_flips_q, n_flips_12;
  int n_solves, n_g, n_checks;

  ConformalMetricDelaunay(Mesh& _m, const std::vector<double>& _Theta_hat) : m(_m)
  {
    #ifdef USE_MPFR
      mpfr::mpreal::set_default_prec(MPFR_PREC);
    #endif
  
    n_e = m.n_edges();
    n_h = m.n_halfedges();
    n_f = m.n_faces();
    n_v = m.n_vertices();
    n_flips = 0;
    n_flips_s = 0;
    n_flips_t = 0;
    n_flips_q = 0;
    n_flips_12 = 0;
    assert(_Theta_hat.size() == n_v);
    
    Theta_hat.resize(n_v);
    for(int i = 0; i < n_v; ++i)
      Theta_hat[i] = _Theta_hat[i];
      
    //make sure Gauss-Bonnet is respected
    int genus = 1 - (m.n_vertices() - m.n_edges() + m.n_faces())/2;
#ifdef USE_MPFR
    Scalar pi = mpfr::const_pi();
#else
    Scalar pi = M_PI;
#endif
    Scalar targetsum = pi * (2 * m.n_vertices() - 4 * (1-genus));
    std::cerr << "Gauss-Bonnet violation before correction: " << Theta_hat.sum() - targetsum << std::endl;
    Theta_hat[0] -= (Theta_hat.sum() - targetsum);
    std::cerr << "Gauss-Bonnet violation  after correction: " << Theta_hat.sum() - targetsum << std::endl;

    u.resize(n_v);
    cot_alpha.resize(n_h);
    alpha.resize(n_h);
    
    H_.resize(n_v, n_v);
  }
  
  void log(const char* c)
  {
    std::cout << c << std::endl;
  }
  
  void FindConformalMetric()
  {
    n_flips = 0;
    n_flips_s = 0;
    n_flips_t = 0;
    n_flips_q = 0;
    n_flips_12 = 0;
    n_solves = 0;
    n_g = 0;
    n_checks = 0;
    
    u.setZero();
    MakeDelaunay();
    compute_angles();
    while(!Converged())
    {
      VectorX d;
      VectorX currentg = g();
      //Eigen::SparseLU< Eigen::SparseMatrix<Scalar> > solver; // a bit more accurate than LDLT, but slower
      Eigen::SimplicialLDLT< Eigen::SparseMatrix<Scalar> > solver;
      solver.compute(H());
      n_solves++;
      if(solver.info()!=Eigen::Success)
      {
        log("Hessian decomposition failed.");
        d = -currentg; //fallback to gradient descent
      }
      else
      {
        d = -solver.solve(currentg);
        if(solver.info()!=Eigen::Success)
        {
          log("Hessian solve failed.");
          d = -currentg; //fallback to gradient descent
        }
        else
        {
          if(d.dot(currentg) >= 0) d = -currentg; //fallback to gradient descent
        }
      }
      d = d.array() - d.mean(); //subtract mean for numerical stability.
      d *= 0.99; // with factor 1 we very often have one backtrack step
      std::cerr << " d("<<d.minCoeff()<<" : "<<d.maxCoeff()<<")  ";
      std::cerr << " u("<<u.minCoeff()<<" : "<<u.maxCoeff()<<")  ";
      u += d;
      MakeDelaunay();
      compute_angles();
      std::cerr << " line-search ";
      int count = 0;
      while(d.dot(g()) > 0)
      {
        d /= 2;
        u -= d;
        MakeDelaunay();
        compute_angles();
        count++;
        std::cerr << count << " ";
        if(count >= 100) break;
      }
      if(count >= 100) break;
    }
    log("converged.");
    std::cerr << "\nn_checks = " << n_checks << "\nn_flips  = "<< n_flips <<"\nn_solves = " << n_solves <<"\nn_g      = " << n_g << "\n";
  }
  
  bool Converged()
  {
    VectorX currentg = g();
    Scalar error =  max(currentg.maxCoeff(), -currentg.minCoeff());
    std::cout << "current error: " << error << "max " << currentg.cwiseAbs().mean() <<"avg " << std::endl;
    #ifdef USE_MPFR
    // FIXME RYAN
      return error < 1e-2;
      return error < pow(2,-MPFR_PREC)*1e6;
    #else
      return error < 1e-10;
    #endif
  }
  
  VectorX g()
  {
    n_g++;
    return Theta_hat - Theta();
  }
  
  Eigen::SparseMatrix<Scalar>& H()
  {
    typedef Eigen::Triplet<Scalar> Trip;
    std::vector<Trip> trips;
    trips.clear();
    trips.resize(n_h*2);
    #pragma omp parallel for
    for(int h = 0; h < n_h; h++)
    {
      int v0 = m.v0(h);
      int v1 = m.v1(h);
      Scalar w = (cot_alpha[h] + cot_alpha[m.opp(h)])/2;
      trips[h*2]   = Trip(v0, v1, -w);
      trips[h*2+1] = Trip(v0, v0, w);
    }
        
    H_.setFromTriplets(trips.begin(), trips.end());
    return H_;
  }
  
  VectorX Theta()
  {
    VectorX t(n_v);
    t.setZero();
    for(int h = 0; h < n_h; h++)
    {
      t[m.to[m.n[h]]] += alpha[h];
    }
    return t;
  }
    
  void MakeDelaunay()
  {
    std::set<int> q;
    for(int i = 0; i < n_e; i++)
    {
      int type0 = m.type[m.h0(i)];
      int type1 = m.type[m.h1(i)];
      if(type0 == 0 || type0 == 1 || type1 == 1 || type0 == 3) //type 22 edges are flipped below; type 44 edges (virtual diagonals) are never flipped.
        q.insert(i);
    }
    while(!q.empty())
    {
      int e = *(q.begin());
      q.erase(q.begin());
      int type0 = m.type[m.h0(e)];
      int type1 = m.type[m.h1(e)];
      if(!(type0 == 2 && type1 == 2) && !(type0 == 4) && NonDelaunay(e))
      {
        int Re = -1;
        if(type0 == 1 && type1 == 1)
          Re = m.e(m.R[m.h0(e)]);
        PtolemyFlip(e,0);
        int hn = m.n[m.h0(e)];
        q.insert(m.e(hn));
        q.insert(m.e(m.n[hn]));
        hn = m.n[m.h1(e)];
        q.insert(m.e(hn));
        q.insert(m.e(m.n[hn]));
        if(type0 == 1 && type1 == 1) // flip mirror edge on sheet 2
        {
          int e = Re;
          PtolemyFlip(e,1);
          int hn = m.n[m.h0(e)];
          q.insert(m.e(hn));
          q.insert(m.e(m.n[hn]));
          hn = m.n[m.h1(e)];
          q.insert(m.e(hn));
          q.insert(m.e(m.n[hn]));
        }
       // checkR();
      }
    }
  }
  
  bool NonDelaunay(int e)
  {
   if(m.type[m.h0(e)] == 4) return false; //virtual diagonal of symmetric trapezoid
   n_checks++;
   int hij = m.h0(e);
   int hjk = m.n[hij];
   int hki = m.n[hjk];
   int hji = m.h1(e);
   int him = m.n[hji];
   int hmj = m.n[him];
   int i = m.to[hji];
   int j = m.to[hij];
   int k = m.to[hjk];
   int n = m.to[him];
   Scalar ui = u[i];
   Scalar uj = u[j];
   Scalar uk = u[k];
   Scalar um = u[n];
   Scalar ljk = ell(m.e(hjk),uj,uk);
   Scalar lki = ell(m.e(hki),uk,ui);
   Scalar lij = ell(m.e(hij),ui,uj);
   Scalar ljm = ell(m.e(hmj),uj,um);
   Scalar lmi = ell(m.e(him),um,ui);
   return (ljk*ljk + lki*lki - lij*lij)/(ljk*lki) + (ljm*ljm + lmi*lmi - lij*lij)/(ljm*lmi) < 0;
  }
    
    
    
  Scalar ell(int e, Scalar u0, Scalar u1)
  {
    return m.l[e] * exp((u0+u1)/2);
  }
    
  void PtolemyFlip(int e, int tag)
  {
    int hij = m.h0(e);
    int hjk = m.n[hij];
    int hki = m.n[hjk];
    int hji = m.h1(e);
    int him = m.n[hji];
    int hmj = m.n[him];
    
    std::vector<char>& type = m.type;
    
    std::vector<int> to_flip;
    if(type[hij] > 0) // skip in non-symmetric mode for efficiency
    {
      int types;
      bool reverse = true;
      if(type[hki] <= type[hmj]) { types = type[hki]*100000 + type[hjk]*10000 + type[hij]*1000 + type[hji]*100 + type[him]*10 + type[hmj]; reverse = false; }
      else types = type[hmj]*100000 + type[him]*10000 + type[hji]*1000 + type[hij]*100 + type[hjk]*10 + type[hki];
      
      if(types == 231123 || types == 231132 || types == 321123) return; // t1t irrelevant
      if(types == 132213 || types == 132231 || types == 312213) return; // t2t irrelevant
      if(types == 341143) return; // q1q irrelevant
      if(types == 342243) return; // q2q irrelevant
      
      if(types != 111111 && types != 222222) std::cerr << "["<<types<<"."<<tag<<"] ";
      if(types == 111222 || types == 123312) n_flips_s++;
      if(types == 111123 || types == 111132) n_flips_t++;
      if(types == 213324 || types == 123314 || types == 111143 || types == 413324 || types == 23314) n_flips_q++;
      if(types == 111111) n_flips_12++;
      switch(types)
      {
        case 111222: // (1|2)
          type[hij] = type[hji] = 3;
          m.R[hij] = hij;
          m.R[hji] = hji;
          break;
        case 123312: // (t,_,t)
          type[hij] = type[hki]; type[hji] = type[hmj];
          m.R[hij] = hji;
          m.R[hji] = hij;
          break;
        case 111123: // (1,1,t)
          type[hij] = type[hji] = 4;
          m.R[hij] = hij;
          m.R[hji] = hji;
          break;
        case 111132: // (1,1,t) mirrored
          type[hij] = type[hji] = 4;
          m.R[hij] = hij;
          m.R[hji] = hji;
          break;
        case 222214: // (2,2,t) following (1,1,t) mirrored
          type[hij] = type[hji] = 3; to_flip.push_back(6); // to make sure all fake diagonals are top left to bottom right
          m.R[hij] = hij;
          m.R[hji] = hji;
          break;
        case 142222: // (2,2,t) following (1,1,t)
          type[hij] = type[hji] = 3;
          m.R[hij] = hij;
          m.R[hji] = hji;
          break;
        case 213324: // (t,_,q)
          type[hij] = type[hji] = 2; to_flip.push_back(6);
          break;
        case 134412: // (t,_,q) 2nd
          type[hij] = type[hji] = 1;
          if(!reverse)
          {
            m.R[hji] = hmj;
            m.R[hmj] = hji;
            m.R[m.opp(hji)] = m.opp(hmj);
            m.R[m.opp(hmj)] = m.opp(hji);
          }
          else
          {
            m.R[hij] = hki;
            m.R[hki] = hij;
            m.R[m.opp(hij)] = m.opp(hki);
            m.R[m.opp(hki)] = m.opp(hij);
          }
          break;
        case 123314: // (q,_,t)
          type[hij] = type[hji] = 1; to_flip.push_back(6);
          break;
        case 124432: // (q,_,t) 2nd
          type[hij] = type[hji] = 2;
          if(!reverse)
          {
            m.R[hki] = hij;
            m.R[hij] = hki;
            m.R[m.opp(hki)] = m.opp(hij);
            m.R[m.opp(hij)] = m.opp(hki);
          }
          else
          {
            m.R[hmj] = hji;
            m.R[hji] = hmj;
            m.R[m.opp(hmj)] = m.opp(hji);
            m.R[m.opp(hji)] = m.opp(hmj);
          }
          break;
        case 111143: // (1,1,q)
          type[hij] = type[hji] = 4;
          m.R[hij] = hij;
          m.R[hji] = hji;
          break;
        case 222243: // (2,2,q) following (1,1,q)
          type[hij] = type[hji] = 4; to_flip.push_back(5);
          m.R[hij] = hij;
          m.R[hji] = hji;
          break;
        case 144442: // (1,1,q)+(2,2,q) 3rd
          type[hij] = type[hji] = 3;
          m.R[hij] = hij;
          m.R[hji] = hji;
          break;
        case 413324: // (q,_,q)
          type[hij] = type[hji] = 4; to_flip.push_back(6); to_flip.push_back(1);
          m.R[hij] = hij;
          m.R[hji] = hji;
          break;
        case 423314: // (q,_,q) opp
          type[hij] = type[hji] = 4; to_flip.push_back(1); to_flip.push_back(6);
          m.R[hij] = hij;
          m.R[hji] = hji;
          break;
        case 134414: // (q,_,q) 2nd
          type[hij] = type[hji] = 1;
          break;
        case 234424: // (q,_,q) 3rd
          type[hij] = type[hji] = 2;
          if(!reverse)
          {
            m.R[hji] = m.n[m.n[m.opp(m.n[m.n[hji]])]]; // attention: hji is not yet flipped here, hence twice .n[]
            m.R[m.n[m.n[m.opp(m.n[m.n[hji]])]]] = hji;
            m.R[m.opp(hji)] = m.opp(m.R[hji]);
            m.R[m.opp(m.R[hji])] = m.opp(hji);
          }
          else
          {
            m.R[hij] = m.n[m.n[m.opp(m.n[m.n[hij]])]];
            m.R[m.n[m.n[m.opp(m.n[m.n[hij]])]]] = hij;
            m.R[m.opp(hij)] = m.opp(m.R[hij]);
            m.R[m.opp(m.R[hij])] = m.opp(hij);
          }
          break;
        case 314423: // fake diag switch following (2,2,t) following (1,1,t) mirrored
          break;
        case 324413: // fake diag switch (opp) following (2,2,t) following (1,1,t) mirrored
          break;
        case 111111:
          break;
        case 222222:
          break;
        case 000000:
          type[hij] = type[hji] = 0; // for non-symmetric mode
          break;
        default: std::cerr << " (attempted to flip edge that should never be non-Delaunay (type: "<<types<<")) "; return;
      }
      
      if(reverse)
      {
        for(int i = 0; i < to_flip.size(); i++)
          to_flip[i] = 7-to_flip[i];
      }
    }
    
    n_flips++;
    if(!m.flip_ccw(hij)) { std::cerr << " EDGE COULD NOT BE FLIPPED! "; }
    if(tag == 1) { m.flip_ccw(hij); m.flip_ccw(hij); } // to make it cw on side 2
    m.l[e] = (m.l[m.e(hjk)]*m.l[m.e(him)] + m.l[m.e(hki)]*m.l[m.e(hmj)])/m.l[m.e(hij)];
    
    for(int i = 0; i < to_flip.size(); i++)
    {
      if(to_flip[i] == 1) PtolemyFlip(m.e(hki),2);
      if(to_flip[i] == 2) PtolemyFlip(m.e(hjk),2);
      if(to_flip[i] == 5) PtolemyFlip(m.e(him),2);
      if(to_flip[i] == 6) PtolemyFlip(m.e(hmj),2);
    }
  }
  
  void compute_angles() // compute alpha and cot_alpha from scaled edge lengths
  {
    #pragma omp parallel for
    for(int f = 0; f < n_f; f++)
    {
      int hi = m.h[f];
      int hj = m.n[hi];
      int hk = m.n[hj];
      int i = m.to[hj];
      int j = m.to[hk];
      int k = m.to[hi];
      Scalar ui = u[i];
      Scalar uj = u[j];
      Scalar uk = u[k];
      Scalar li = ell(m.e(hi),uj,uk);
      Scalar lj = ell(m.e(hj),uk,ui);
      Scalar lk = ell(m.e(hk),ui,uj);
      // (following "A Cotangent Laplacian for Images as Surfaces")
      Scalar s = (li+lj+lk)/2.0;
      Scalar Aijk4 = 4.0*sqrt(max(s*(s-li)*(s-lj)*(s-lk),0.0));
      Scalar Ijk = (-li*li+lj*lj+lk*lk);
      Scalar iJk = (li*li-lj*lj+lk*lk);
      Scalar ijK = (li*li+lj*lj-lk*lk);
      cot_alpha[hi] = Aijk4 == 0.0 ? copysign(cot_infty,Ijk) : (Ijk/Aijk4);
      cot_alpha[hj] = Aijk4 == 0.0 ? copysign(cot_infty,iJk) : (iJk/Aijk4);
      cot_alpha[hk] = Aijk4 == 0.0 ? copysign(cot_infty,ijK) : (ijK/Aijk4);
      
      alpha[hi] = acos(min(max(Ijk/(2.0*lj*lk), -1.0), 1.0));
      alpha[hj] = acos(min(max(iJk/(2.0*lk*li), -1.0), 1.0));
      alpha[hk] = acos(min(max(ijK/(2.0*li*lj), -1.0), 1.0));
    }
  }
  
  void compute_layout(std::vector<double>& _u, std::vector<double>& _v) //metric -> parametrization
  {
    _u.resize(n_h);
    _v.resize(n_h);
    
    std::vector<double> phi(n_h);
    
    //set starting point
    int h = 0;
    phi[h] = 0.0;
    double offset = u[m.to[h]]TODOUBLE;
    _u[h] = 0.0;
    _v[h] = 0.0;
    h = m.n[h];
    phi[h] = u[m.to[h]]TODOUBLE - offset;
    _u[h] = m.l[m.e(h)]TODOUBLE*std::exp(phi[h]/2);
    _v[h] = 0.0;
    
    // layout the rest of the mesh by BFS
    std::vector<bool> visited(n_f, false);
    std::queue<int> q;
    q.push(h);
    visited[m.f[h]] = true;
    while(!q.empty())
    {
      h = q.front();
      q.pop();
      
      int hn = m.n[h];
      int hp = m.n[hn];
      
      phi[hn] = u[m.to[hn]]TODOUBLE - offset;
      
      double len = m.l[m.e(hn)]TODOUBLE * std::exp((phi[h]+phi[hn])/2);
      
      double ud = _u[hp]-_u[h];
      double vd = _v[hp]-_v[h];
      double d = std::sqrt(ud*ud + vd*vd);
      double co = cos(alpha[hp])TODOUBLE;
      double si = sin(alpha[hp])TODOUBLE;
      
      _u[hn] = _u[h] + (co*ud + si*vd)*len/d;
      _v[hn] = _v[h] + (co*vd - si*ud)*len/d;
      
      int hno = m.opp(hn);
      int hpo = m.opp(hp);
      if(!visited[m.f[hno]])
      {
        visited[m.f[hno]] = true;
        phi[hno] = phi[h];
        phi[m.n[m.n[hno]]] = phi[hn];
        _u[hno] = _u[h];
        _v[hno] = _v[h];
        _u[m.n[m.n[hno]]] = _u[hn];
        _v[m.n[m.n[hno]]] = _v[hn];
        q.push(hno);
      }
      if(!visited[m.f[hpo]])
      {
        visited[m.f[hpo]] = true;
        phi[hpo] = phi[hn];
        phi[m.n[m.n[hpo]]] = phi[hp];
        _u[hpo] = _u[hn];
        _v[hpo] = _v[hn];
        _u[m.n[m.n[hpo]]] = _u[hp];
        _v[m.n[m.n[hpo]]] = _v[hp];
        q.push(hpo);
      }
    }
  }
};


void python_to_cpp_mesh(std::vector<int>& _n,
                        std::vector<int>& _to,
                        std::vector<int>& _f,
                        std::vector<int>& _h,
                        std::vector<int>& _out,
                        std::vector<int>& _type,
                        std::vector<int>& _R,
                        std::vector<double>& _l,
                        Mesh& m)
{
    m.n = _n;
    m.to = _to;
    m.f = _f;
    m.h = _h;
    m.out = _out;
    
    // Convert the type vector to char
    std::vector<char> type_char(_type.begin(), _type.end());
    m.type = type_char;
    m.R = _R;

    // Convert the length vector to Scalar
    std::vector<ConformalMetricDelaunay::Scalar> _l_Scalar(_l.begin(), _l.end());
    m.l = _l_Scalar;
    
    return;
}

std::tuple<std::vector<int>,
           std::vector<int>,
           std::vector<int>,
           std::vector<int>,
           std::vector<int>,
           std::vector<int>,
           std::vector<int>,
           std::vector<double>,
           Eigen::Matrix<double,Eigen::Dynamic,1>,
           int>
make_delaunay_cpp(std::vector<int>& _n,
                  std::vector<int>& _to,
                  std::vector<int>& _f,
                  std::vector<int>& _h,
                  std::vector<int>& _out,
                  std::vector<int>& _type,
                  std::vector<int>& _R,
                  std::vector<double>& _l,
                  Eigen::Matrix<double,Eigen::Dynamic,1>& _phi,
                  const std::vector<double>& _Theta_hat)
{
    Mesh m;
    python_to_cpp_mesh(_n, _to, _f, _h, _out, _type, _R, _l, m);

    // Create a ConformalMetricDelaunay instance and make the mesh Delaunay
    ConformalMetricDelaunay CMD(m, _Theta_hat);
    CMD.u.resize(_phi.size());
    for(int i = 0; i < _phi.size(); ++i)
      CMD.u[i] = _phi[i];
    CMD.MakeDelaunay();
    
    // Convert the CMD type vector into int
    std::vector<int> type_int(m.type.begin(),m.type.end());
    
    // Convert the CMD length vector back to double
    std::vector<double> l_double(m.l.begin(),m.l.end());
    
    // Convert the CMD phi vector back to a double vector
    Eigen::Matrix<double,Eigen::Dynamic,1> phi;
    phi.resize(CMD.u.size());
    for(int i = 0; i < phi.size(); ++i)
      phi[i] = CMD.u[i]TODOUBLE;
        
    auto T = std::make_tuple(m.n,
                             m.to,
                             m.f,
                             m.h,
                             m.out,
                             type_int,
                             m.R,
                             l_double,
                             phi,
                             CMD.n_flips);
    return T;
}

std::tuple<std::vector<int>,
           std::vector<int>,
           std::vector<int>,
           std::vector<int>,
           std::vector<int>,
           std::vector<int>,
           std::vector<int>,
           std::vector<double>,
           Eigen::Matrix<double,Eigen::Dynamic,1>,
           int>
find_conformal_metric_cpp(std::vector<int> _n,
              std::vector<int>& _to,
              std::vector<int>& _f,
              std::vector<int>& _h,
              std::vector<int>& _out,
              std::vector<int> _type,
              std::vector<int>& _R,
              std::vector<double>& _l,
              Eigen::Matrix<double,Eigen::Dynamic,1>& _phi,
              const std::vector<double>& _Theta_hat)
{
    Mesh m;
    //python_to_cpp_mesh(_n, _to, _f, _h, _out, _type, _R, _l, m);
    m.n = _n;
    m.to = _to;
    m.f = _f;
    m.h = _h;
    m.out = _out;
    
    // Convert the type vector to char
    std::vector<char> type_char(_type.begin(), _type.end());
    m.type = type_char;
    m.R = _R;

    // Convert the length vector to Scalar
    std::vector<ConformalMetricDelaunay::Scalar> _l_Scalar(_l.begin(), _l.end());
    m.l = _l_Scalar;

    // Create a ConformalMetricDelaunay instance and find a conformal metric
    ConformalMetricDelaunay CMD(m, _Theta_hat);
    CMD.u.resize(_phi.size());
    for(int i = 0; i < _phi.size(); ++i)
      CMD.u[i] = _phi[i];
    CMD.FindConformalMetric();
    
    // Convert the CMD type vector into int
    std::vector<int> type_int(m.type.begin(),m.type.end());
    
    // Convert the CMD length vector back to double
    std::vector<double> l_double(m.l.begin(),m.l.end());
    
    // Convert the CMD phi vector back to a double vector
    Eigen::Matrix<double,Eigen::Dynamic,1> phi;
    phi.resize(CMD.u.size());
    for(int i = 0; i < phi.size(); ++i)
      phi[i] = CMD.u[i]TODOUBLE;
        
    auto T = std::make_tuple(m.n,
                             m.to,
                             m.f,
                             m.h,
                             m.out,
                             type_int,
                             m.R,
                             l_double,
                             phi,
                             CMD.n_flips);
    return T;
}

// Function for debugging to verify that passing between mesh structures in python
// and cpp works correctly
std::tuple<std::vector<int>,
           std::vector<int>,
           std::vector<int>,
           std::vector<int>,
           std::vector<int>,
           std::vector<int>,
           std::vector<int>,
           std::vector<double>,
           Eigen::Matrix<double,Eigen::Dynamic,1>,
           int>
conversion_cpp(std::vector<int> _n,
              std::vector<int>& _to,
              std::vector<int>& _f,
              std::vector<int>& _h,
              std::vector<int>& _out,
              std::vector<int> _type,
              std::vector<int>& _R,
              std::vector<double>& _l,
              Eigen::Matrix<double,Eigen::Dynamic,1>& _phi,
              const std::vector<double>& _Theta_hat)
{
    Mesh m;
    python_to_cpp_mesh(_n, _to, _f, _h, _out, _type, _R, _l, m);

    // Create a ConformalMetricDelaunay instance
    ConformalMetricDelaunay CMD(m, _Theta_hat);
    CMD.u.resize(_phi.size());
    for(int i = 0; i < _phi.size(); ++i)
      CMD.u[i] = _phi[i];
    
    // Convert the CMD type vector into int
    std::vector<int> type_int(m.type.begin(),m.type.end());
    
    // Convert the CMD length vector back to double
    std::vector<double> l_double(m.l.begin(),m.l.end());
    
    // Convert the CMD phi vector back to a double vector
    Eigen::Matrix<double,Eigen::Dynamic,1> phi;
    phi.resize(CMD.u.size());
    for(int i = 0; i < phi.size(); ++i)
      phi[i] = CMD.u[i]TODOUBLE;
        
    auto T = std::make_tuple(m.n,
                             m.to,
                             m.f,
                             m.h,
                             m.out,
                             type_int,
                             m.R,
                             l_double,
                             phi,
                             CMD.n_flips);
    return T;
}
#endif