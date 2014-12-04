
#ifndef AURA_BACKEND_CUDA_CONTEXT_HPP
#define AURA_BACKEND_CUDA_CONTEXT_HPP

#include <cuda.h>
#include <boost/aura/backend/cuda/call.hpp>

namespace boost
{
namespace aura {
namespace backend_detail {
namespace cuda {
namespace detail {

/**
 * channel class
 *
 * efficient transfer of data between host memory and device memory
 *
 * Depending on usage scenario and supported backend, the internals
 * of ths class are diverse.
 *
 * A user can: 
 * provide host memory (might be pinned, depending on backend
 * and device support)
 * ask for host memory (might be pinned, depending on backend 
 * and device support)
 * specify seldom access (might be implemented as to peer to peer access)
 * speficy frequent access (might be implemented as memory mapping
 * or pinned memory copy)
 */
template <typename T>
class channel {

public:

  /**
   * create channel, let channel allocate host and device memory
   *
   * memory access indicats if memory is accessed frequently
   * after a transfer through channel was initiated or seldom (once)
   *
   * rationale: 
   * if memory is accessed frequently, pinned copy is more efficient
   * if memory is accessed seldom, p2p access from the kernel is
   * more efficient
   */
  inline explicit channel(device& d, bounds b, memory_access ma) 
  {
	/**
	 * seldom access:
	 * allocate pinned host memory, provide device accessible
	 * pointer to pinned host memory for device access
	 *
	 * frequent access:
	 * allocate pinned host memory and corresponding device memory
	 */
  }
	
  template <typename HostIterator>
  inline explicit channel(device& d, HostIterator& hi, 
		  bounds b, memory_access ma) 
  {
	/**
	 * seldom access:
	 * if possible provide device accessible pointer 
	 * to pinned host memory for device access
	 * otherwise allocate device memory
	 *
	 * frequent access:
	 * hmmmmmm 
	 */
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
  
  /// access the device ordinal
  inline std::size_t get_ordinal() const {
    return ordinal_;
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
} // boost

#endif // AURA_BACKEND_CUDA_CONTEXT_HPP

