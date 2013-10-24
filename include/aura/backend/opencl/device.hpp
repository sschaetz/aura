
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
  inline explicit device() : empty_(true) {}
 
  /**
   * create device form ordinal, also creates a context
   *
   * @param ordinal device number
   */
  inline device(int ordinal) : pinned_(false), empty_(false) {
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
    d.empty_ = true;
  }

  /**
   * move assignment, move device information here, invalidate other
   *
   * @param d device to move here
   */
  device& operator=(BOOST_RV_REF(device) d) 
  { 
    finalize();
    device_ = d.device_;
    context_ = d.context_;
    pinned_ = d.pinned_;
    empty_ = false;
    d.empty_ = true;
    return *this;
  }

  /**
   * destroy device (context)
   */
  inline ~device() {
    finalize();
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
  /// finalize object (called from dtor and move assign)
  void finalize() {
    if(empty_) {
      return;
    }
    AURA_OPENCL_SAFE_CALL(clReleaseContext(context_));
  }

private:
  /// device handle
  cl_device_id device_;
  /// context handle
  cl_context context_;
  /// flag indicating pinned or unpinned context
  bool pinned_;
  /// empty marker
  int empty_;
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

/**
 * print device info to stdout
 */
inline void print_device_info() {
  // get platforms
  unsigned int num_platforms = 0;
  AURA_OPENCL_SAFE_CALL(clGetPlatformIDs(0, 0, &num_platforms));
  std::vector<cl_platform_id> platforms(num_platforms);
  AURA_OPENCL_SAFE_CALL(clGetPlatformIDs(num_platforms, &platforms[0], 0));
  
  unsigned int num = 0;

  // find device 
  for(unsigned int i=0; i<num_platforms; i++) {
    unsigned int num_devices_platform = 0;
    AURA_OPENCL_SAFE_CALL(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 
      0, 0, &num_devices_platform));
    std::vector<cl_device_id> devices;
    devices.reserve(num_devices_platform);
    AURA_OPENCL_SAFE_CALL(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 
      num_devices_platform, &devices[0], NULL));
    for(unsigned int j=0; j<num_devices_platform; j++) {
      std::size_t namelen;
      AURA_OPENCL_SAFE_CALL(clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, 
        NULL, &namelen));
      std::vector<char> name;
      name.reserve(namelen);
      AURA_OPENCL_SAFE_CALL(clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 
        namelen, &name[0], NULL));
      printf("platform %u device %u (ordinal %u): %s\n", i, j, num, &name[0]);
      num++;
    }
  }
}


} // opencl 
} // backend_detail
} // aura

#endif // AURA_BACKEND_OPENCL_DEVICE_HPP

