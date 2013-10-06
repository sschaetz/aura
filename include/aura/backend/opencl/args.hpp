#ifndef AURA_BACKEND_OPENCL_ARGS_HPP
#define AURA_BACKEND_OPENCL_ARGS_HPP

#include <array>
#include <utility>

#include <aura/detail/svec.hpp>

namespace aura {
namespace backend_detail {
namespace opencl {

typedef std::pair<void *, std::size_t> arg_t;
typedef svec<arg_t, AURA_MAX_KERNEL_ARGS> args_t;

template <typename T0>
args_t args(T0 & a0) {
  return args_t(arg_t(&a0, sizeof(T0)));
}

template <typename T0, typename T1>
args_t args(T0 & a0, T1 & a1) {
  return args_t(
    arg_t(&a0, sizeof(T0)),
    arg_t(&a1, sizeof(T1))
  );
}

template <typename T0, typename T1, typename T2>
args_t args(T0 & a0, T1 & a1, T2 & a2) {
  return args_t(
    arg_t(&a0, sizeof(T0)),
    arg_t(&a1, sizeof(T1)),
    arg_t(&a2, sizeof(T2))
  );
}

template <typename T0, typename T1, typename T2, typename T3>
args_t args(T0 & a0, T1 & a1, T2 & a2, T3 & a3) {
  return args_t(
    arg_t(&a0, sizeof(T0)),
    arg_t(&a1, sizeof(T1)),
    arg_t(&a2, sizeof(T2)),
    arg_t(&a3, sizeof(T3))
  );
}

} // opencl 
} // backend_detail
} // aura


#endif // AURA_BACKEND_OPENCL_ARGS_HPP

