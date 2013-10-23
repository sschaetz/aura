#ifndef AURA_BACKEND_CUDA_FEED_HPP
#define AURA_BACKEND_CUDA_FEED_HPP

#include <boost/move/move.hpp>
#include <cuda.h>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/device.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {

/**
 * feed class
 */
class feed {

private:
  BOOST_MOVABLE_BUT_NOT_COPYABLE(feed)

public:
 
  /**
   * create empty feed object without device and stream
   */
  inline explicit feed() : device_(0), stream_((CUstream)feed::no_stream) {}

  /**
   * create device feed for device
   *
   * @param d device to create feed for
   *
   * const device & is not allowed since an actual instance is needed
   */
  inline explicit feed(device & d) : device_(&d) {
    device_->set();
    AURA_CUDA_SAFE_CALL(cuStreamCreate(&stream_, 0 /*CU_STREAM_NON_BLOCKING*/));
    device_->unset(); 
  }

  /**
   * move constructor, move feed information here, invalidate other
   *
   * @param f feed to move here
   */
  feed(BOOST_RV_REF(feed) f) : 
    device_(f.device_), stream_(f.stream_) {  
    f.stream_ = (CUstream)feed::no_stream;
  }

  /**
   * move assignment, move feed information here, invalidate other
   *
   * @param f feed to move here
   */
  feed& operator=(BOOST_RV_REF(feed) f) { 
    stream_ = f.stream_;
    f.stream_ = (CUstream)feed::no_stream;
    return *this;
  }

  /**
   * destroy feed
   */
  inline ~feed() {
    if((CUstream)feed::no_stream != stream_) {
      device_->set();
      AURA_CUDA_SAFE_CALL(cuStreamDestroy(stream_));
      device_->unset(); 
    }
  }
  
  /**
   * wait until all commands in the feed have finished
   */
  inline void synchronize() const {
    device_->set();
    AURA_CUDA_SAFE_CALL(cuStreamSynchronize(stream_));
    device_->unset();
  }
  
  /// make feed active
  inline void set() const {
    device_->set(); 
  }
  
  /// undo make feed active
  inline void unset() const {
    device_->unset(); 
  }
 
  /// get device 
  inline const CUdevice & get_device() const {
    return device_->get_device();
  }

  /// get context 
  inline const CUcontext & get_context() const {
    return device_->get_context();
  }

  /// get stream
  inline const CUstream & get_stream() const {
    return stream_;
  }


private:
  /// reference to device the feed was created for
  device * device_;
  /// stream handle
  CUstream stream_;

  // 0 is probably not ok for CUDA, it is the default stream
  // we're relying here on implementation details of CUDA
  // this might not be the best way to do this
  /// const value indicating no stream
  static const int no_stream = -1;
};

/**
 * @brief wait for a feed to finish all operations
 *
 * @param f the feed to wait for
 */
void wait_for(feed & f) {
  f.synchronize();
}

} // cuda
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_FEED_HPP

