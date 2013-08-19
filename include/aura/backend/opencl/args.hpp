#ifndef AURA_BACKEND_OPENCL_ARGS_HPP
#define AURA_BACKEND_OPENCL_ARGS_HPP

#include <array>
#include <utility>

namespace aura {
namespace backend_detail {
namespace opencl {

// type to hold kernel arguments
template <std::size_t N>
struct args_t {
  typedef std::array<std::pair<void *, std::size_t>, N> type;
};

// generate 1 kernel argument
template <typename T0>
args_t<1>::type args(T0 & a0) {
  args_t<1>::type r;
  r[0] = std::pair<void*, std::size_t>(&a0, sizeof(T0));
  return r;
}

// generate 2 kernel arguments
template <typename T0, typename T1>
args_t<2>::type args(T0 & a0, T1 & a1) {
  args_t<2>::type r;
  r[0] = std::pair<void*, std::size_t>(&a0, sizeof(T0));
  r[1] = std::pair<void*, std::size_t>(&a1, sizeof(T1));
  return r;
}

// generate 3 kernel arguments
template <typename T0, typename T1, typename T2>
args_t<3>::type args(T0 & a0, T1 & a1, T2 & a2) {
  args_t<3>::type r;
  r[0] = std::pair<void*, std::size_t>(&a0, sizeof(T0));
  r[1] = std::pair<void*, std::size_t>(&a1, sizeof(T1));
  r[2] = std::pair<void*, std::size_t>(&a2, sizeof(T2));
  return r;
}

} // opencl 
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_ARGS_HPP

