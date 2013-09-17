#ifndef AURA_BACKEND_CUDA_FFT_HPP
#define AURA_BACKEND_CUDA_FFT_HPP

#include <boost/move/move.hpp>
#include <cuda.h>
#include <cufft.h>
#include <aura/backend/cuda/call.hpp>
#include <aura/backend/cuda/device.hpp>

namespace aura {
namespace backend_detail {
namespace cuda {



/**
 * fft class
 */
class fft {

private:
  BOOST_MOVABLE_BUT_NOT_COPYABLE(fft)

public:
 
  /**
   * create empty fft object without device and stream
   */
  inline explicit fft() : device_(0), stream_((CUstream)fft::no_stream) {}

  /**
   * create device fft for device
   *
   * @param d device to create fft for
   */
  inline explicit fft(device & d) : device_(&d) {
    device_->set();
    AURA_CUDA_SAFE_CALL(cuStreamCreate(&stream_, 0 /*CU_STREAM_NON_BLOCKING*/));
    device_->unset(); 
  }

  /**
   * move constructor, move fft information here, invalidate other
   *
   * @param f fft to move here
   */
  fft(BOOST_RV_REF(fft) f) : 
    device_(f.device_), stream_(f.stream_)
  {  
    f.stream_ = (CUstream)fft::no_stream;
  }

  /**
   * move assignment, move fft information here, invalidate other
   *
   * @param f fft to move here
   */
  fft& operator=(BOOST_RV_REF(fft) f) 
  { 
    stream_ = f.stream_;
    f.stream_ = (CUstream)fft::no_stream;
    return *this;
  }

  /**
   * destroy fft
   */
  inline ~fft() {
    if((CUstream)fft::no_stream != stream_) {
      device_->set();
      AURA_CUDA_SAFE_CALL(cuStreamDestroy(stream_));
      device_->unset(); 
    }
  }
  
  /**
   * wait until all commands in the fft have finished
   */
  inline void synchronize() const {
    device_->set();
    AURA_CUDA_SAFE_CALL(cuStreamSynchronize(stream_));
    device_->unset();
  }
  
  /// make fft active
  inline void set() const {
    device_->set(); 
  }
  
  /// undo make fft active
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
  /// reference to device the fft was created for
  device * device_;
  /// stream handle
  CUstream stream_;

  // 0 is probably not ok for CUDA, it is the default stream
  // we're relying here on implementation details of CUDA
  // this might not be the best way to do this
  /// const value indicating no stream
  static const int no_stream = -1;
};

} // cuda
} // backend_detail
} // aura

#endif // AURA_BACKEND_CUDA_FFT_HPP

