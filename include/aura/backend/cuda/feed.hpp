#ifndef AURA_BACKEND_CUDA_FEED_HPP
#define AURA_BACKEND_CUDA_FEED_HPP

#include <boost/noncopyable.hpp>
#include <cuda.h>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/device.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

/**
 * feed class
 */
class feed : private boost::noncopyable {

public:
  /**
   * create device feed for device
   *
   * @param d device to create feed for
   */
  inline feed(const device & d) : device_(d), pinned_(false) {
    set();
    AURA_CUDA_SAFE_CALL(cuStreamCreate(&stream_, 0 /*CU_STREAM_NON_BLOCKING*/));
    unset(); 
  }

  /**
   * destroy feed
   */
  inline ~feed() {
    set();
    AURA_CUDA_SAFE_CALL(cuStreamDestroy(stream_));
    unset();
  }
  
  /**
   * wait until all commands in the feed have finished
   */
  inline void synchronize() {
    set();
    AURA_CUDA_SAFE_CALL(cuStreamSynchronize(stream_));
    unset();
  }
  
  /// make feed active
  inline void set() {
    device_.set(); 
  }
  
  /// undo make feed active
  inline void unset() {
    device_.unset(); 
  }
 
  /// get device 
  inline const CUdevice & get_device() const {
    return device_.get_device();
  }

  /// get context 
  inline const CUcontext & get_context() const {
    return device_.get_context();
  }

  /// get stream
  inline const CUstream & get_stream() const {
    return stream_;
  }

private:
  /// reference to device the feed was created for
  const device & device_;
  /// stream handle
  CUstream stream_;
  /// flag indicating pinned or unpinned context
  bool pinned_;

};

} // cuda
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_FEED_HPP

