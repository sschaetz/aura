
#ifndef AURA_BACKEND_OPENCL_DEVICE_HPP
#define AURA_BACKEND_OPENCL_DEVICE_HPP


#include <boost/move/move.hpp>
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
class device {

private:
  BOOST_MOVABLE_BUT_NOT_COPYABLE(device)

public:

  /**
   * create empty device object without device and context
   */
  inline explicit device() : 
    device_(device::no_device), context_(device::no_context), pinned_(false) {}
 
  /**
   * create device form ordinal, also creates a context
   *
   * @param ordinal device number
   */
  inline device(int ordinal) : pinned_(false) {
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
   * move constructor, move device information here, invalidate other
   *
   * @param d device to move here
   */
  device(BOOST_RV_REF(device) d) : 
    device_(d.device_), context_(d.context_), pinned_(d.pinned_)
  {  
    d.device_ = device::no_device;
    d.context_ = device::no_context;
    d.pinned_ = false; 
  }

  /**
   * move assignment, move device information here, invalidate other
   *
   * @param d device to move here
   */
  device& operator=(BOOST_RV_REF(device) d) 
  { 
    device_ = d.device_;
    context_ = d.context_;
    pinned_ = d.pinned_;
    d.device_ = device::no_device;
    d.context_ = device::no_context;
    d.pinned_ = false;
    return *this;
  }

  /**
   * destroy device (context)
   */
  inline ~device() {
    AURA_OPENCL_SAFE_CALL(clReleaseContext(context_));
  }


  /// make device active
  inline void set() const {
  }
  
  /// undo make device active
  inline void unset() const {
  }

  /// pin
  inline void pin() {
    pinned_ = true;
  }
  
  /// unpin 
  inline void unpin() {
    pinned_ = false;
  } 
  
  /// access the device handle
  inline const cl_device_id & get_device() const {
    return device_; 
  }
  
  /// access the context handle
  inline const cl_context & get_context() const {
    return context_; 
  }

private:
  /// device handle
  cl_device_id device_;
  /// context handle
  cl_context context_;
  /// flag indicating pinned or unpinned context
  bool pinned_;

  /// const value indicating no device
  static int const no_device = 0;

  // bit of a hack, should be 
  // static constexpr cl_context no_context = -1;
  /// const value indicationg no context
  static int const no_context = 0; 

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

