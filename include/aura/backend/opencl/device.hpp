
#ifndef AURA_BACKEND_OPENCL_DEVICE_HPP
#define AURA_BACKEND_OPENCL_DEVICE_HPP


#include <boost/noncopyable.hpp>
#include <vector>
#include <CL/cl.h>
#include <aura/backend/opencl/call.hpp>

namespace aura {
namespace backend_detail {
namespace opencl {


/**
 * device class
 *
 * every interaction with devices starts from this class
 */
class device : private boost::noncopyable {

public:
  /**
   * create device form ordinal, also creates a context
   *
   * @param ordinal device number
   */
  inline device(int ordinal) {
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
   * destroy device (context)
   */
  inline ~device() {
    AURA_OPENCL_SAFE_CALL(clReleaseContext(context_));
  }

  /// make device active
  inline void set() {
  }
  
  /// undo make device active
  inline void unset() {
  }

  /// pin (make pinned, deactivate set/unset)
  inline void pin() {
  }
  
  /// unpin (make unpinned, activate set/unset)
  inline void unpin() {
  } 
  
  inline const cl_device_id & get_device() const {
    return device_; 
  }
  
  inline const cl_context & get_context() const {
    return context_; 
  }

private:
  /// device handle
  cl_device_id device_;
  /// context handle
  cl_context context_;
};
 

/**
 * get number of devices available
 *
 * @return number of devices
 */
inline int device_get_count() {
  // get platforms
  unsigned int num_platforms = 0;
  AURA_OPENCL_SAFE_CALL(clGetPlatformIDs(0, 0, &num_platforms));
  std::vector<cl_platform_id> platforms(num_platforms);
  AURA_OPENCL_SAFE_CALL(clGetPlatformIDs(num_platforms, &platforms[0], 0));
  
  // find device 
  int num_devices = 0;
  for(unsigned int i=0; i<num_platforms; i++) {
    unsigned int num_devices_platform = 0;
    AURA_OPENCL_SAFE_CALL(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 
      0, 0, &num_devices_platform));
    num_devices += num_devices_platform; 
  }
  return num_devices;
}


} // opencl 
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_DEVICE_HPP

