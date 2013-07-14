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
  inline feed(const device & d) : 
    device_(d.get_device()), context_(d.get_context()), pinned_(false) {
    int errorcode = 0;
    stream_ = clCreateCommandQueue(context_, device_, 0, &errorcode);
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
  inline void synchronize() {
    cl_event event;
    AURA_OPENCL_SAFE_CALL(clEnqueueMarker(stream_, &event));
    AURA_OPENCL_SAFE_CALL(clEnqueueWaitForEvents(stream_, 1, &event));
  }
  
  /// make feed active
  inline void set() {
  }
  
  /// undo make feed active
  inline void unset() {
  }

  /// pin (make pinned, deactivate set/unset)
  inline void pin() {
  }
  
  /// unpin (make unpinned, activate set/unset)
  inline void unpin() {
  }

  /// get device 
  inline const cl_device_id & get_device() const {
    return device_;
  }

  /// get context 
  inline const cl_context & get_context() const {
    return context_;
  }

  /// get stream
  inline const cl_command_queue & get_stream() const {
    return stream_;
  }


private:
  /// reference to device handle
  const cl_device_id & device_;
  /// reference to context handle
  const cl_context & context_;
  /// stream handle
  cl_command_queue stream_;
  /// flag indicating pinned or unpinned context
  bool pinned_;

};

} // opencl
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_FEED_HPP

