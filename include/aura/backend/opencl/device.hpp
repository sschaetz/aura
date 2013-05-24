#ifndef AURA_BACKEND_OPENCL_DEVICE_HPP
#define AURA_BACKEND_OPENCL_DEVICE_HPP

#include <vector>
#include <CL/cl.h>

namespace aura {
namespace backend {
namespace opencl {

/// device handle
typedef cl_device_id device;

/**
 * create device handle from number
 *
 * since we don't expose the platform concept at the moment
 * this function is rather messy
 *
 * @param ordinal device number to get handle from
 * @return the device handle
 */
inline device device_create(int ordinal) {
  // get platforms
  unsigned int num_platforms = 0;
  AURA_OPENCL_SAFE_CALL(clGetPlatformIDs(NULL, NULL, &num_platforms));
  std::vector<cl_platform_id> platforms(num_platforms);
  AURA_OPENCL_SAFE_CALL(clGetPlatformIDs(num_platforms, &platforms[0], NULL));
  
  // find device 
  unsigned int num_devices = 0;
  cl_platform_id p;
  for(unsigned int i=0; i<num_platforms; i++) {
    unsigned int num_devices_platform = 0;
    AURA_OPENCL_SAFE_CALL(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 
      NULL, NULL, &num_devices_platform));
    
    // check if we found the device we want
    if(num_devices+num_devices_platform > ordinal) {
      std::vector<cl_device_id> devices(num_devices_platform);
      AURA_OPENCL_SAFE_CALL(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 
        num_devices_platform, &devices[0], NULL));
      return devices[ordinal-num_devices];
    }
  }
}

} // opencl 
} // backend
} // aura


#endif // AURA_BACKEND_OPENCL_DEVICE_HPP

