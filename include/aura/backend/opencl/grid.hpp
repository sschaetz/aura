#ifndef AURA_BACKEND_OPENCL_GRID_HPP
#define AURA_BACKEND_OPENCL_GRID_HPP

#include <array>

namespace aura {
namespace backend_detail {
namespace opencl {

// type to hold the grid size
template <std::size_t N>
struct grid_t {
  typedef std::array<std::size_t, N> type;
};

// generate 1 dimensional grid size
grid_t<1>::type grid(std::size_t s0) {
  grid_t<1>::type r;
  r[0] = s0;
  return r;
}

// generate 2 dimensional grid size
grid_t<2>::type grid(std::size_t s0, std::size_t s1) {
  grid_t<2>::type r;
  r[0] = s0;
  r[1] = s1;
  return r;
}

// generate 3 dimensional grid size
grid_t<3>::type grid(std::size_t s0, std::size_t s1, std::size_t s2) {
  grid_t<3>::type r;
  r[0] = s0;
  r[1] = s1;
  r[2] = s2;
  return r;
}

} // opencl 
} // backend_detail
} // aura



#endif // AURA_BACKEND_OPENCL_GRID_HPP

