#ifndef AURA_BACKEND_CUDA_ARGS_HPP
#define AURA_BACKEND_CUDA_ARGS_HPP

#include <array>
#include <utility>

namespace aura {
namespace backend_detail {
namespace cuda {

// type to hold kernel arguments
template <std::size_t N>
struct args_t {
  typedef std::array<void *, N> type;
};

// generate 1 kernel argument
template <typename T0>
args_t<1>::type args(T0 & a0) {
  args_t<1>::type r;
  r[0] = (void*)&a0;
  return r;
}

// generate 2 kernel arguments
template <typename T0, typename T1>
args_t<2>::type args(T0 & a0, T1 & a1) {
  args_t<2>::type r;
  r[0] = (void*)&a0;
  r[1] = (void*)&a1;
  return r;
}

// generate 3 kernel arguments
template <typename T0, typename T1, typename T2>
args_t<3>::type args(T0 & a0, T1 & a1, T2 & a2) {
  args_t<3>::type r;
  r[0] = (void*)&a0;
  r[1] = (void*)&a1;
  r[2] = (void*)&a2;
  return r;
}

} // cuda 
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_ARGS_HPP

