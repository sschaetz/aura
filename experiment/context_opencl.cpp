/**
 * basic code from [0], copyright Matthew Scarpino, August 03, 2011 
 * [0] http://www.drdobbs.com/parallel/a-gentle-introduction-to-opencl/231002854
 */

#include <stdio.h>
#include <CL/cl.h>

int main(void) {
  
  cl_platform_id platform;
  cl_device_id dev;
  int err;
   
  /* Identify a platform */
  err = clGetPlatformIDs(1, &platform, NULL);
  if(err < 0) {
    perror("Couldn't identify a platform");
    exit(1);
  } 

  /* Platform information */
  char platform_name[251];
  clGetPlatformInfo (platform, CL_PLATFORM_NAME, 250, platform_name, NULL);
  printf("platform name: %s\n", platform_name);

  /* Access a device */
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
  if(err == CL_DEVICE_NOT_FOUND) {
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
  }
  if(err < 0) {
    perror("Couldn't access any devices");
    exit(1); 
  }

  

}
