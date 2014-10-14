#ifndef AURA_BACKEND_CUDA_ARGS_HPP
#define AURA_BACKEND_CUDA_ARGS_HPP

#include <array>
#include <utility>
#include <cstring>

#include <aura/detail/svec.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

typedef void * arg_t;
typedef svec<arg_t, AURA_MAX_KERNEL_ARGS> args_tt;

// we pass an array containing copies of the args as first
// and the actual args as second argument
typedef std::pair<void *, args_tt> args_t;

template <typename T0>
args_t args(const T0 & a0) {
  const std::size_t size = sizeof(T0); 
  char * ptr = (char *)malloc(size);
  std::size_t offset = 0;
  std::memcpy(&ptr[offset], &a0, sizeof(T0));
  char * a0p = ptr+offset;
  return args_t(ptr, 
    args_tt(arg_t(a0p)));
}

template <typename T0, typename T1>
args_t args(const T0 & a0, const T1 & a1) {
  const std::size_t size = sizeof(T0) + sizeof(T1); 
  char * ptr = (char *)malloc(size);
  std::size_t offset = 0;
  std::memcpy(&ptr[offset], &a0, sizeof(T0));
  char * a0p = ptr+offset;
  offset += sizeof(T0);
  std::memcpy(&ptr[offset], &a1, sizeof(T1));
  char * a1p = ptr+offset;
  return args_t(ptr, 
    args_tt(
      arg_t(a0p),
      arg_t(a1p)
    )
  );
}

template <typename T0, typename T1, typename T2>
args_t args(const T0 & a0, const T1 & a1, const T2 & a2) {
  const std::size_t size = sizeof(T0) + sizeof(T1) + sizeof(T2); 
  char * ptr = (char *)malloc(size);
  std::size_t offset = 0;
  std::memcpy(&ptr[offset], &a0, sizeof(T0));
  char * a0p = ptr+offset;
  offset += sizeof(T0);
  std::memcpy(&ptr[offset], &a1, sizeof(T1));
  char * a1p = ptr+offset;
  offset += sizeof(T1);
  std::memcpy(&ptr[offset], &a2, sizeof(T2));
  char * a2p = ptr+offset;
  return args_t(ptr,
    args_tt(
      arg_t(a0p),
      arg_t(a1p),
      arg_t(a2p)
    )
  );
}

template <typename T0, typename T1, typename T2, typename T3>
args_t args(const T0 & a0, const T1 & a1, const T2 & a2, const T3 & a3) {
  const std::size_t size = sizeof(T0) + sizeof(T1) + sizeof(T2) +
    sizeof(T3);
  char * ptr = (char *)malloc(size);
  std::size_t offset = 0;
  std::memcpy(&ptr[offset], &a0, sizeof(T0));
  char * a0p = ptr+offset;
  offset += sizeof(T0);
  std::memcpy(&ptr[offset], &a1, sizeof(T1));
  char * a1p = ptr+offset;
  offset += sizeof(T1);
  std::memcpy(&ptr[offset], &a2, sizeof(T2));
  char * a2p = ptr+offset;
  offset += sizeof(T2);
  std::memcpy(&ptr[offset], &a3, sizeof(T3));
  char * a3p = ptr+offset;
  return args_t(ptr,
    args_tt(
      arg_t(a0p),
      arg_t(a1p),
      arg_t(a2p),
      arg_t(a3p)
    )
  );
}

template <typename T0, typename T1, typename T2, typename T3,
          typename T4>
args_t args(const T0 & a0, const T1 & a1, const T2 & a2, 
  const T3 & a3, const T4 & a4) {
  const std::size_t size = sizeof(T0) + sizeof(T1) + sizeof(T2) +
    sizeof(T3) + sizeof(T4);
  char * ptr = (char *)malloc(size);
  std::size_t offset = 0;
  std::memcpy(&ptr[offset], &a0, sizeof(T0));
  char * a0p = ptr+offset;
  offset += sizeof(T0);
  std::memcpy(&ptr[offset], &a1, sizeof(T1));
  char * a1p = ptr+offset;
  offset += sizeof(T1);
  std::memcpy(&ptr[offset], &a2, sizeof(T2));
  char * a2p = ptr+offset;
  offset += sizeof(T2);
  std::memcpy(&ptr[offset], &a3, sizeof(T3));
  char * a3p = ptr+offset;
  offset += sizeof(T3);
  std::memcpy(&ptr[offset], &a4, sizeof(T4));
  char * a4p = ptr+offset;
  return args_t(ptr,
    args_tt(
      arg_t(a0p),
      arg_t(a1p),
      arg_t(a2p),
      arg_t(a3p),
      arg_t(a4p)
    )
  );
}

template <typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5>
args_t args(const T0 & a0, const T1 & a1, const T2 & a2, const T3 & a3, 
  const T4 & a4, const T5 & a5) {
  const std::size_t size = sizeof(T0) + sizeof(T1) + sizeof(T2) +
    sizeof(T3) + sizeof(T4) + sizeof(T5);
  char * ptr = (char *)malloc(size);
  std::size_t offset = 0;
  std::memcpy(&ptr[offset], &a0, sizeof(T0));
  char * a0p = ptr+offset;
  offset += sizeof(T0);
  std::memcpy(&ptr[offset], &a1, sizeof(T1));
  char * a1p = ptr+offset;
  offset += sizeof(T1);
  std::memcpy(&ptr[offset], &a2, sizeof(T2));
  char * a2p = ptr+offset;
  offset += sizeof(T2);
  std::memcpy(&ptr[offset], &a3, sizeof(T3));
  char * a3p = ptr+offset;
  offset += sizeof(T3);
  std::memcpy(&ptr[offset], &a4, sizeof(T4));
  char * a4p = ptr+offset;
  offset += sizeof(T4);
  std::memcpy(&ptr[offset], &a5, sizeof(T5));
  char * a5p = ptr+offset;
  return args_t(ptr,
    args_tt(
      arg_t(a0p),
      arg_t(a1p),
      arg_t(a2p),
      arg_t(a3p),
      arg_t(a4p),
      arg_t(a5p)
    )
  );
}

template <typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6>
args_t args(const T0 & a0, const T1 & a1, const T2 & a2, const T3 & a3, 
  const T4 & a4, const T5 & a5, const T6 & a6) {
  const std::size_t size = sizeof(T0) + sizeof(T1) + sizeof(T2) +
    sizeof(T3) + sizeof(T4) + sizeof(T5) + sizeof(T6);
  char * ptr = (char *)malloc(size);
  std::size_t offset = 0;
  std::memcpy(&ptr[offset], &a0, sizeof(T0));
  char * a0p = ptr+offset;
  offset += sizeof(T0);
  std::memcpy(&ptr[offset], &a1, sizeof(T1));
  char * a1p = ptr+offset;
  offset += sizeof(T1);
  std::memcpy(&ptr[offset], &a2, sizeof(T2));
  char * a2p = ptr+offset;
  offset += sizeof(T2);
  std::memcpy(&ptr[offset], &a3, sizeof(T3));
  char * a3p = ptr+offset;
  offset += sizeof(T3);
  std::memcpy(&ptr[offset], &a4, sizeof(T4));
  char * a4p = ptr+offset;
  offset += sizeof(T4);
  std::memcpy(&ptr[offset], &a5, sizeof(T5));
  char * a5p = ptr+offset;
  offset += sizeof(T5);
  std::memcpy(&ptr[offset], &a6, sizeof(T6));
  char * a6p = ptr+offset;
  return args_t(ptr,
    args_tt(
      arg_t(a0p),
      arg_t(a1p),
      arg_t(a2p),
      arg_t(a3p),
      arg_t(a4p),
      arg_t(a5p),
      arg_t(a6p)
    )
  );
}

template <typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7>
args_t args(const T0 & a0, const T1 & a1, const T2 & a2, const T3 & a3, 
  const T4 & a4, const T5 & a5, const T6 & a6, const T7 & a7) {
  const std::size_t size = sizeof(T0) + sizeof(T1) + sizeof(T2) +
    sizeof(T3) + sizeof(T4) + sizeof(T5) + sizeof(T6) + sizeof(T7);
  char * ptr = (char *)malloc(size);
  std::size_t offset = 0;
  std::memcpy(&ptr[offset], &a0, sizeof(T0));
  char * a0p = ptr+offset;
  offset += sizeof(T0);
  std::memcpy(&ptr[offset], &a1, sizeof(T1));
  char * a1p = ptr+offset;
  offset += sizeof(T1);
  std::memcpy(&ptr[offset], &a2, sizeof(T2));
  char * a2p = ptr+offset;
  offset += sizeof(T2);
  std::memcpy(&ptr[offset], &a3, sizeof(T3));
  char * a3p = ptr+offset;
  offset += sizeof(T3);
  std::memcpy(&ptr[offset], &a4, sizeof(T4));
  char * a4p = ptr+offset;
  offset += sizeof(T4);
  std::memcpy(&ptr[offset], &a5, sizeof(T5));
  char * a5p = ptr+offset;
  offset += sizeof(T5);
  std::memcpy(&ptr[offset], &a6, sizeof(T6));
  char * a6p = ptr+offset;
  offset += sizeof(T6);
  std::memcpy(&ptr[offset], &a7, sizeof(T7));
  char * a7p = ptr+offset;
  return args_t(ptr,
    args_tt(
      arg_t(a0p),
      arg_t(a1p),
      arg_t(a2p),
      arg_t(a3p),
      arg_t(a4p),
      arg_t(a5p),
      arg_t(a6p),
      arg_t(a7p)
    )
  );
}

template <typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7,
          typename T8>
args_t args(const T0 & a0, const T1 & a1, const T2 & a2, const T3 & a3, 
  const T4 & a4, const T5 & a5, const T6 & a6, const T7 & a7, const T8 & a8) {
  const std::size_t size = sizeof(T0) + sizeof(T1) + sizeof(T2) +
    sizeof(T3) + sizeof(T4) + sizeof(T5) + sizeof(T6) + sizeof(T7) +
    sizeof(T8);
  char * ptr = (char *)malloc(size);
  std::size_t offset = 0;
  std::memcpy(&ptr[offset], &a0, sizeof(T0));
  char * a0p = ptr+offset;
  offset += sizeof(T0);
  std::memcpy(&ptr[offset], &a1, sizeof(T1));
  char * a1p = ptr+offset;
  offset += sizeof(T1);
  std::memcpy(&ptr[offset], &a2, sizeof(T2));
  char * a2p = ptr+offset;
  offset += sizeof(T2);
  std::memcpy(&ptr[offset], &a3, sizeof(T3));
  char * a3p = ptr+offset;
  offset += sizeof(T3);
  std::memcpy(&ptr[offset], &a4, sizeof(T4));
  char * a4p = ptr+offset;
  offset += sizeof(T4);
  std::memcpy(&ptr[offset], &a5, sizeof(T5));
  char * a5p = ptr+offset;
  offset += sizeof(T5);
  std::memcpy(&ptr[offset], &a6, sizeof(T6));
  char * a6p = ptr+offset;
  offset += sizeof(T6);
  std::memcpy(&ptr[offset], &a7, sizeof(T7));
  char * a7p = ptr+offset;
  offset += sizeof(T7);
  std::memcpy(&ptr[offset], &a8, sizeof(T8));
  char * a8p = ptr+offset;
  return args_t(ptr,
    args_tt(
      arg_t(a0p),
      arg_t(a1p),
      arg_t(a2p),
      arg_t(a3p),
      arg_t(a4p),
      arg_t(a5p),
      arg_t(a6p),
      arg_t(a7p),
      arg_t(a8p)
    )
  );
}

template <typename T0, typename T1, typename T2, typename T3,
          typename T4, typename T5, typename T6, typename T7,
          typename T8, typename T9>
args_t args(const T0 & a0, const T1 & a1, const T2 & a2, const T3 & a3, 
  const T4 & a4, const T5 & a5, const T6 & a6, const T7 & a7, 
  const T8 & a8, const T9 & a9) {
  const std::size_t size = sizeof(T0) + sizeof(T1) + sizeof(T2) +
    sizeof(T3) + sizeof(T4) + sizeof(T5) + sizeof(T6) + sizeof(T7) +
    sizeof(T8) + sizeof(T9);
  char * ptr = (char *)malloc(size);
  std::size_t offset = 0;
  std::memcpy(&ptr[offset], &a0, sizeof(T0));
  char * a0p = ptr+offset;
  offset += sizeof(T0);
  std::memcpy(&ptr[offset], &a1, sizeof(T1));
  char * a1p = ptr+offset;
  offset += sizeof(T1);
  std::memcpy(&ptr[offset], &a2, sizeof(T2));
  char * a2p = ptr+offset;
  offset += sizeof(T2);
  std::memcpy(&ptr[offset], &a3, sizeof(T3));
  char * a3p = ptr+offset;
  offset += sizeof(T3);
  std::memcpy(&ptr[offset], &a4, sizeof(T4));
  char * a4p = ptr+offset;
  offset += sizeof(T4);
  std::memcpy(&ptr[offset], &a5, sizeof(T5));
  char * a5p = ptr+offset;
  offset += sizeof(T5);
  std::memcpy(&ptr[offset], &a6, sizeof(T6));
  char * a6p = ptr+offset;
  offset += sizeof(T6);
  std::memcpy(&ptr[offset], &a7, sizeof(T7));
  char * a7p = ptr+offset;
  offset += sizeof(T7);
  std::memcpy(&ptr[offset], &a8, sizeof(T8));
  char * a8p = ptr+offset;
  offset += sizeof(T8);
  std::memcpy(&ptr[offset], &a9, sizeof(T9));
  char * a9p = ptr+offset;
  return args_t(ptr,
    args_tt(
      arg_t(a0p),
      arg_t(a1p),
      arg_t(a2p),
      arg_t(a3p),
      arg_t(a4p),
      arg_t(a5p),
      arg_t(a6p),
      arg_t(a7p),
      arg_t(a8p),
      arg_t(a9p)
    )
  );
}

} // cuda 
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_ARGS_HPP

