#ifndef AURA_BACKEND_OPENCL_FEED_HPP
#define AURA_BACKEND_OPENCL_FEED_HPP

#include <boost/move/move.hpp>
#include <CL/cl.h>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/device.hpp>

namespace aura {
namespace backend_detail {
namespace opencl {

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
  inline explicit feed() : empty_(true) {}
  
  /**
   * create device feed for device
   *
   * @param d device to create feed for
   *
   * const device & is not allowed since an actual instance is needed
   */
  inline feed(device & d) : device_(&d), empty_(false) {
    int errorcode = 0;
    stream_ = clCreateCommandQueue(device_->get_context(), 
      device_->get_device(), 0, &errorcode);
    AURA_OPENCL_CHECK_ERROR(errorcode);
  }

  /**
   * move constructor, move feed information here, invalidate other
   *
   * @param f feed to move here
   */
  feed(BOOST_RV_REF(feed) f) : 
    device_(f.device_), stream_(f.stream_), empty_(false) {  
    f.empty_ = true;
  }

  /**
   * move assignment, move feed information here, invalidate other
   *
   * @param f feed to move here
   */
  feed& operator=(BOOST_RV_REF(feed) f) { 
    finalize();
    stream_ = f.stream_;
    empty_ = false;
    f.empty_ = false;
    return *this;
  }

  /**
   * destroy feed
   */
  inline ~feed() {
    finalize();
  }
  
  /**
   * wait until all commands in the feed have finished
   */
  inline void synchronize() const {
    AURA_OPENCL_SAFE_CALL(clFinish(stream_));
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
  inline const cl_device_id & get_device() const {
    return device_->get_device();
  }

  /// get context 
  inline const cl_context & get_context() const {
    return device_->get_context();
  }

  /// get stream
  inline const cl_command_queue & get_stream() const {
    return stream_;
  }
  
private:
  /// finalize object (called from dtor and move assign)
  void finalize() {
    if(empty_) {
      return;
    }
    AURA_OPENCL_SAFE_CALL(clReleaseCommandQueue(stream_)); 
  }

private:
  /// reference to device the feed was created for
  device * device_;
  /// stream handle
  cl_command_queue stream_;
  /// empty marker
  bool empty_;
};

/**
 * @brief wait for a feed to finish all operations
 *
 * @param f the feed to wait for
 */
void wait_for(feed & f) {
  f.synchronize();
}

} // opencl
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_FEED_HPP

