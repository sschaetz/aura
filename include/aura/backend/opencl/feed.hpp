#ifndef AURA_BACKEND_OPENCL_FEED_HPP
#define AURA_BACKEND_OPENCL_FEED_HPP

#include <boost/move/move.hpp>
#include <CL/cl.h>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/device.hpp>

namespace aura {
namespace backend_detail {
namespace opencl {

#ifndef CL_VERSION_1_2
class mark;
#endif

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
  inline explicit feed() : context_(nullptr) {}
  
  /**
   * create device feed for device
   *
   * @param d device to create feed for
   *
   * const device & is not allowed since an actual instance is needed
   */
  inline feed(device & d) : context_(d.get_context()) {
    int errorcode = 0;
    stream_ = clCreateCommandQueue(context_->get_backend_context(), 
      context_->get_backend_device(), 0, &errorcode);
    AURA_OPENCL_CHECK_ERROR(errorcode);
  }

  /**
   * move constructor, move feed information here, invalidate other
   *
   * @param f feed to move here
   */
  feed(BOOST_RV_REF(feed) f) : 
    context_(f.context_), stream_(f.stream_) {  
    f.context_ = nullptr;
  }

  /**
   * move assignment, move feed information here, invalidate other
   *
   * @param f feed to move here
   */
  feed& operator=(BOOST_RV_REF(feed) f) { 
    finalize();
    context_ = f.context_;
    stream_ = f.stream_;
    f.context_ = nullptr;
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
  inline void set() const { }
  
  /// undo make feed active
  inline void unset() const { }

  /// get device 
  inline const cl_device_id & get_backend_device() const {
    return context_->get_backend_device();
  }

  /// get context 
  inline const cl_context & get_backend_context() const {
    return context_->get_backend_context();
  }

  /// get stream
  inline const cl_command_queue & get_backend_stream() const {
    return stream_;
  }

#ifndef CL_VERSION_1_2
protected:
  inline void insert_event(cl_event* event) {
	  // insert a dummy memory copy
	AURA_OPENCL_SAFE_CALL(clEnqueueCopyBuffer(stream_,
		context_->get_dummy_mem(), context_->get_dummy_mem(),
		0, 1, 1, 0, 0, event));
  }
#endif // CL_VERSION_1_2 

private:
  /// finalize object (called from dtor and move assign)
  void finalize() {
    if(nullptr != context_) {
      AURA_OPENCL_SAFE_CALL(clReleaseCommandQueue(stream_)); 
    }
  }

private:
  /// reference to device context the feed was created for
  detail::context * context_;
  /// stream handle
  cl_command_queue stream_;


#ifndef CL_VERSION_1_2
friend class mark;
friend void insert(feed& f, mark& m);
#endif // CL_VERSION_1_2
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

