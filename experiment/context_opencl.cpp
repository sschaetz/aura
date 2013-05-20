#include <vector>
#include <stdio.h>
#include <CL/cl.h>
#include <boost/thread.hpp>

#define TESTSIZE 512*512

void worker1(cl_context * context) {
  int err; 
  cl_mem device_memory;
  device_memory = clCreateBuffer(*context, CL_MEM_READ_WRITE, 
    sizeof(int) * TESTSIZE, NULL, &err);
  if(err < 0) {
    perror("Could not allocate memory in worker1");
    exit(1); 
  }
  err = clReleaseMemObject(device_memory);
  if(err < 0) {
    perror("Could not free memory in worker1");
    exit(1); 
  }
}

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

  /* Create a context */
  cl_context context = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
  if(err < 0) {
    perror("Could not create context");
    exit(1); 
  }

  /* Create command queue */
  cl_command_queue commands = clCreateCommandQueue(context, dev, 0, &err);
  if(err < 0) {
    perror("Could not create commands");
    exit(1); 
  }

  /* Run memory test */
  std::vector<int> a1(TESTSIZE, 42);
  std::vector<int> a2(TESTSIZE);

  cl_mem device_memory;
  device_memory = clCreateBuffer(context, CL_MEM_READ_WRITE, 
    sizeof(int) * TESTSIZE, NULL, &err);
  if(err < 0) {
    perror("Could not allocate memory");
    exit(1); 
  }
 
  err = clEnqueueWriteBuffer(commands, device_memory, CL_TRUE, 0, 
    sizeof(int) * TESTSIZE, &a1[0], 0, NULL, NULL); 
  if(err < 0) {
    perror("Could not write buffer");
    exit(1); 
  }
  err = clEnqueueReadBuffer(commands, device_memory, CL_TRUE, 0, 
    sizeof(int) * TESTSIZE, &a2[0], 0, NULL, NULL); 
  if(err < 0) {
    perror("Could not read buffer");
    exit(1); 
  }

  if(std::equal(a1.begin(), a1.end(), a2.begin())) {
     printf("Copy test ok.\n");
  } else {
    fprintf(stderr, "Copy test not ok.\n");
  }

  err = clReleaseMemObject(device_memory);
  if(err < 0) {
    perror("Could not free memory");
    exit(1); 
  }

  boost::thread wt1(worker1, &context);
  wt1.join();

  /* Release the context */ 
  err = clReleaseContext(context);
  if(err < 0) {
    perror("Could not create context");
    exit(1); 
  }
  

}
