
#ifndef AURA_BACKEND_CUDA_CONTEXT_HPP
#define AURA_BACKEND_CUDA_CONTEXT_HPP

#include <cuda.h>
#include <aura/backend/cuda/call.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {
namespace detail {

/**
 * context class
 *
 * holds device resources and provides basic device interaction
 * must not be instantiated by user
 */
class context {

public:

  /**
   * create context form ordinal
   *
   * @param ordinal context number
   */
  inline explicit context(std::size_t ordinal) : 
      ordinal_(ordinal), pinned_(false) {
    AURA_CUDA_SAFE_CALL(cuDeviceGet(&device_, ordinal));
    AURA_CUDA_SAFE_CALL(cuCtxCreate(&context_, 0, device_));
  }

  /// destroy context
  inline ~context() {
    AURA_CUDA_SAFE_CALL(cuCtxDestroy(context_));
  }

  /// make context active
  inline void set() const {
    AURA_CUDA_SAFE_CALL(cuCtxSetCurrent(context_));
  }
  
  /// undo make context active
  inline void unset() const {
    if(pinned_) {
      return;
    }
    AURA_CUDA_SAFE_CALL(cuCtxSetCurrent(NULL));
  }

  /**
   * pin 
   *
   * disable unset, context context stays associated with current thread
   * usefull for interoperability with other libraries that use a context
   * explicitly
   */
  inline void pin() {
    set();
    pinned_ = true;
  }
  
  /// unpin (reenable unset)
  inline void unpin() {
    pinned_ = false;
  } 

  /// access the device handle
  inline const CUdevice & get_backend_device() const {
    return device_; 
  }
  
  /// access the context handle
  inline const CUcontext & get_backend_context() const {
    return context_; 
  }

private:
  /// device ordinal
  std::size_t ordinal_;
  /// device handle
  CUdevice device_;
  /// context handle 
  CUcontext context_;
  /// flag indicating pinned or unpinned context
  bool pinned_;
};


} // detail
} // cuda
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_CONTEXT_HPP

