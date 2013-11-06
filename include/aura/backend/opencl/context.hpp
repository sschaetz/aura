
#ifndef AURA_BACKEND_OPENCL_CONTEXT_HPP
#define AURA_BACKEND_OPENCL_CONTEXT_HPP


#include <boost/move/move.hpp>
#include <vector>
#include <CL/cl.h>
#include <aura/backend/opencl/call.hpp>

namespace aura {
namespace backend_detail {
namespace opencl {
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
  inline context (int ordinal) : ordinal_(ordinal) {
    // get platforms
    unsigned int num_platforms = 0;
    AURA_OPENCL_SAFE_CALL(clGetPlatformIDs(0, 0, &num_platforms));
    std::vector<cl_platform_id> platforms(num_platforms);
    AURA_OPENCL_SAFE_CALL(clGetPlatformIDs(num_platforms, &platforms[0], NULL));
    
    // find device 
    unsigned int num_devices = 0;
    for(unsigned int i=0; i<num_platforms; i++) {
      unsigned int num_devices_platform = 0;
      AURA_OPENCL_SAFE_CALL(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 
        0, 0, &num_devices_platform));
      
      // check if we found the device we want
      if(num_devices+num_devices_platform > (unsigned)ordinal) {
        std::vector<cl_device_id> devices(num_devices_platform);
        AURA_OPENCL_SAFE_CALL(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 
          num_devices_platform, &devices[0], 0));
        
        device_ = devices[ordinal-num_devices];
      }
    }
   
    int errorcode = 0;
    context_ = clCreateContext(NULL, 1, &device_, NULL, NULL, &errorcode);
    AURA_OPENCL_CHECK_ERROR(errorcode);
  }

  /**
   * destroy context 
   */
  inline ~context() {
    AURA_OPENCL_SAFE_CALL(clReleaseContext(context_));
  }


  /// make context active
  inline void set() const {}
  
  /// undo make context active
  inline void unset() const {}

  /// pin
  inline void pin() {}
  
  /// unpin 
  inline void unpin() {} 
  
  /// access the device handle
  inline const cl_device_id & get_backend_device() const {
    return device_; 
  }
  
  /// access the context handle
  inline const cl_context & get_backend_context() const {
    return context_; 
  }
  
  /// access the device ordinal
  inline std::size_t get_ordinal() const {
    return ordinal_;
  }

private:
  /// device ordinal
  int ordinal_;
  /// device handle
  cl_device_id device_;
  /// context handle
  cl_context context_;
};
 
} // detail
} // opencl 
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_CONTEXT_HPP

