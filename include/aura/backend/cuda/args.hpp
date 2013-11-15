#ifndef AURA_BACKEND_CUDA_ARGS_HPP
#define AURA_BACKEND_CUDA_ARGS_HPP

#include <array>
#include <utility>

#include <aura/detail/svec.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

typedef void * arg_t;
typedef svec<arg_t, AURA_MAX_KERNEL_ARGS> args_t;

template <typename T0>
args_t args(T0 & a0) {
  return args_t((void*)&a0);
}

template <typename T0, typename T1>
args_t args(T0 & a0, T1 & a1) {
  return args_t(
    (void*)&a0,
    (void*)&a1
  );
}

template <typename T0, typename T1, typename T2>
args_t args(T0 & a0, T1 & a1, T2 & a2) {
  return args_t(
    (void*)&a0,
    (void*)&a1,
    (void*)&a2
  );
}

template <typename T0, typename T1, typename T2, typename T3>
args_t args(T0 & a0, T1 & a1, T2 & a2, T3 & a3) {
  return args_t(
    (void*)&a0,
    (void*)&a1,
    (void*)&a2,
    (void*)&a3
  );
}

template <typename T0, typename T1, typename T2, typename T3,
          typename T4>
args_t args(T0 & a0, T1 & a1, T2 & a2, T3 & a3, T4 & a4) {
  return args_t(
    (void*)&a0,
    (void*)&a1,
    (void*)&a2,
    (void*)&a3,
    (void*)&a4
  );
}

template <typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5>
args_t args(T0 & a0, T1 & a1, T2 & a2, T3 & a3, T4 & a4, T5 & a5) {
  return args_t(
    (void*)&a0,
    (void*)&a1,
    (void*)&a2,
    (void*)&a3,
    (void*)&a4,
    (void*)&a5
  );
}

template <typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6>
args_t args(T0 & a0, T1 & a1, T2 & a2, T3 & a3, T4 & a4, T5 & a5, 
  T6 & a6) {
  return args_t(
    (void*)&a0,
    (void*)&a1,
    (void*)&a2,
    (void*)&a3,
    (void*)&a4,
    (void*)&a5,
    (void*)&a6
  );
}

template <typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7>
args_t args(T0 & a0, T1 & a1, T2 & a2, T3 & a3, T4 & a4, T5 & a5, 
  T6 & a6, T7 & a7) {
  return args_t(
    (void*)&a0,
    (void*)&a1,
    (void*)&a2,
    (void*)&a3,
    (void*)&a4,
    (void*)&a5,
    (void*)&a6,
    (void*)&a7
  );
}

template <typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7,
          typename T8>
args_t args(T0 & a0, T1 & a1, T2 & a2, T3 & a3, T4 & a4, T5 & a5, 
  T6 & a6, T7 & a7, T8 & a8) {
  return args_t(
    (void*)&a0,
    (void*)&a1,
    (void*)&a2,
    (void*)&a3,
    (void*)&a4,
    (void*)&a5,
    (void*)&a6,
    (void*)&a7,
    (void*)&a8
  );
}

template <typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9>
args_t args(T0 & a0, T1 & a1, T2 & a2, T3 & a3, T4 & a4, T5 & a5, 
  T6 & a6, T7 & a7, T8 & a8, T9 & a9) {
  return args_t(
    (void*)&a0,
    (void*)&a1,
    (void*)&a2,
    (void*)&a3,
    (void*)&a4,
    (void*)&a5,
    (void*)&a6,
    (void*)&a7,
    (void*)&a8,
    (void*)&a9
  );
}

} // cuda 
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_ARGS_HPP

