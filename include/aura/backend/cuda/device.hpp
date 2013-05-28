
#ifndef AURA_BACKEND_CUDA_DEVICE_HPP
#define AURA_BACKEND_CUDA_DEVICE_HPP


#include <cuda.h>
#include <aura/backend/cuda/call.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

/// device handle
typedef CUdevice device;

/**
 * create device handle from number
 *
 * @param ordinal device number to get handle for
 * @return the device handle
 */
inline device device_create(int ordinal) {
  device d;
  AURA_CUDA_SAFE_CALL(cuDeviceGet(&d, ordinal));
  return d;
}

/**
 * get number of devices available
 *
 * @return number of devices
 */
inline int device_get_count() {
  int num_devices;
  AURA_CUDA_SAFE_CALL(cuDeviceGetCount(&num_devices));
  return num_devices;
}


} // cuda
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_DEVICE_HPP

