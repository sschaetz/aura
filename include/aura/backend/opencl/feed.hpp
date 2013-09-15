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
  inline explicit feed() : device_(0), stream_((cl_command_queue)feed::no_stream) {}
  
  /**
   * create device feed for device
   *
   * @param d device to create feed for
   */
  inline feed(device & d) : device_(&d) {
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
    device_(f.device_), stream_(f.stream_)
  {  
    f.stream_ = (cl_command_queue)feed::no_stream;
  }

  /**
   * move assignment, move feed information here, invalidate other
   *
   * @param f feed to move here
   */
  feed& operator=(BOOST_RV_REF(feed) f) 
  { 
    stream_ = f.stream_;
    f.stream_ = (cl_command_queue)feed::no_stream;
    return *this;
  }

  /**
   * destroy feed
   */
  inline ~feed() {
    AURA_OPENCL_SAFE_CALL(clReleaseCommandQueue(stream_)); 
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
    return device_->get_context();;
  }

  /// get stream
  inline const cl_command_queue & get_stream() const {
    return stream_;
  }


private:
  /// reference to device the feed was created for
  device * device_;
  /// stream handle
  cl_command_queue stream_;
  
  /// const value indicating no stream
  static const int no_stream = 0;
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

