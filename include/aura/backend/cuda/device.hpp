
#ifndef AURA_BACKEND_CUDA_DEVICE_HPP
#define AURA_BACKEND_CUDA_DEVICE_HPP

#include <boost/move/move.hpp>
#include <cuda.h>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/context.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

/**
 * device class
 *
 * every interaction with devices starts from this class
 */
class device {

private:
  BOOST_MOVABLE_BUT_NOT_COPYABLE(device)

public:

  /// create empty device object 
  inline explicit device() : context_(nullptr) {}
   
  /**
   * create device form ordinal
   *
   * @param ordinal device number
   */
  inline explicit device(std::size_t ordinal) : 
    context_(new detail::context(ordinal)) {}

  /// destroy device
  inline ~device() {
    finalize();
  }

  /**
   * move constructor, move device information here, invalidate other
   *
   * @param d device to move here
   */
  device(BOOST_RV_REF(device) d) : context_(d.context_) { 
    d.context_ = nullptr;
  }

  /**
   * move assignment, move device information here, invalidate other
   *
   * @param d device to move here
   */
  device& operator=(BOOST_RV_REF(device) d) 
  {
    finalize();
    context_ = d.context_;
    d.context_ = nullptr;
    return *this;
  }

  /// make device active
  inline void set() {
    context_->set();
  }
  
  /// undo make device active
  inline void unset() {
    context_->unset();
  }

  /**
   * pin 
   *
   * disable unset, device context stays associated with current thread
   * usefull for interoperability with other libraries that use a context
   * explicitly
   */
  inline void pin() {
    context_->pin();
  }
  
  /// unpin (reenable unset)
  inline void unpin() {
    context_->unpin();
  } 

  /// access the device handle
  inline const CUdevice & get_backend_device() const {
    return context_->get_backend_device(); 
  }
  
  /// access the context handle
  inline const CUcontext & get_backend_context() const {
    return context_->get_backend_context(); 
  }
  
  /// access the context handle
  inline detail::context * get_context() {
    return context_; 
  }

private:
  /// finalize object (called from dtor and move assign)
  void finalize() {
    if(nullptr != context_) {
      delete context_;
    }
  }

private:
  /// device context
  detail::context * context_;
};
  
/**
 * get number of devices available
 *
 * @return number of devices
 */
inline std::size_t device_get_count() {
  int num_devices;
  AURA_CUDA_SAFE_CALL(cuDeviceGetCount(&num_devices));
  return (std::size_t)num_devices;
}

/// print device info to stdout
inline void print_device_info() {
  for(std::size_t n=0; n<device_get_count(); n++) {
    CUdevice device; 
    AURA_CUDA_SAFE_CALL(cuDeviceGet(&device, n));
    char device_name[400];
    AURA_CUDA_SAFE_CALL(cuDeviceGetName(device_name, 400, device)); 
    printf("%lu: %s\n", n, device_name); 
  }
}

#include <aura/backend/shared/device_info.hpp>

/// return the device info 
device_info device_get_info(device & d) {
  device_info di;
  // name and vendor
  AURA_CUDA_SAFE_CALL(cuDeviceGetName(di.name, sizeof(di.name)-1,
    device.get_backend_device()));
  strncpy(di.vendor, "CUDA", sizeof(di.vendor)-1); 
  
  // mesh 
  int r;
  AURA_CUDA_SAFE_CALL(&r, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
    device.get_backend_device());
  di.max_grid.push_back(r);
  AURA_CUDA_SAFE_CALL(&r, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
    device.get_backend_device());
  di.max_grid.push_back(r);
  AURA_CUDA_SAFE_CALL(&r, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
    device.get_backend_device());
  di.max_grid.push_back(r);

  // bundle 
  AURA_CUDA_SAFE_CALL(&r, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
    device.get_backend_device());
  di.max_block.push_back(r);
  AURA_CUDA_SAFE_CALL(&r, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
    device.get_backend_device());
  di.max_block.push_back(r);
  AURA_CUDA_SAFE_CALL(&r, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
    device.get_backend_device());
  di.max_block.push_back(r);

  // fibers in bundle
  AURA_CUDA_SAFE_CALL(&r, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    device.get_backend_device());
  max_threads = r;
}

} // cuda
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_DEVICE_HPP

