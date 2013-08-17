#ifndef AURA_BACKEND_OPENCL_FEED_HPP
#define AURA_BACKEND_OPENCL_FEED_HPP

#include <boost/noncopyable.hpp>
#include <CL/cl.h>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/device.hpp>

namespace aura {
namespace backend_detail {
namespace opencl {

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
  inline feed(const device & d) : device_(d) {
    int errorcode = 0;
    stream_ = clCreateCommandQueue(device_.get_context(), 
      device_.get_device(), 0, &errorcode);
    AURA_OPENCL_CHECK_ERROR(errorcode);
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
    device_.set();
  }
  
  /// undo make feed active
  inline void unset() const {
    device_.unset();
  }

  /// get device 
  inline const cl_device_id & get_device() const {
    return device_.get_device();
  }

  /// get context 
  inline const cl_context & get_context() const {
    return device_.get_context();;
  }

  /// get stream
  inline const cl_command_queue & get_stream() const {
    return stream_;
  }


private:
  /// reference to device the feed was created for
  const device & device_;
  /// stream handle
  cl_command_queue stream_;

};

} // opencl
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_FEED_HPP

