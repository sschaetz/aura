#pragma once

#include <boost/aura/base/opencl/safecall.hpp>

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace opencl
{

class device
{
public:
        /// @copydoc boost::aura::base::cuda::device::device()
        inline explicit device(std::size_t ordinal)
                : ordinal_(ordinal)
        {
                // Get platforms.
                unsigned int num_platforms = 0;
                AURA_OPENCL_SAFE_CALL(clGetPlatformIDs(0, 0, &num_platforms));
                std::vector<cl_platform_id> platforms(num_platforms);
                AURA_OPENCL_SAFE_CALL(
                        clGetPlatformIDs(num_platforms, &platforms[0], NULL));

                // Find device.
                unsigned int num_devices = 0;
                for (unsigned int i = 0; i < num_platforms; i++)
                {
                        unsigned int num_devices_platform = 0;
                        AURA_OPENCL_SAFE_CALL(
                                clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,
                                        0, 0, &num_devices_platform));

                        // Check if we found the device we want.
                        if (num_devices + num_devices_platform >
                                (unsigned)ordinal)
                        {
                                std::vector<cl_device_id> devices(
                                        num_devices_platform);
                                AURA_OPENCL_SAFE_CALL(clGetDeviceIDs(
                                        platforms[i], CL_DEVICE_TYPE_ALL,
                                        num_devices_platform, &devices[0], 0));

                                device_ = devices[ordinal - num_devices];
                        }
                }

                int errorcode = 0;
                context_ = clCreateContext(
                        NULL, 1, &device_, NULL, NULL, &errorcode);
                AURA_OPENCL_CHECK_ERROR(errorcode);

#ifndef CL_VERSION_1_2
                dummy_mem_ = clCreateBuffer(
                        context_, CL_MEM_READ_WRITE, 2, 0, &errorcode);
                AURA_OPENCL_CHECK_ERROR(errorcode);
#endif // CL_VERSION_1_2
        }

        /// @copydoc boost::aura::base::cuda::device::~device()
        inline ~device()
        {
#ifndef CL_VERSION_1_2
                AURA_OPENCL_SAFE_CALL(clReleaseMemObject(dummy_mem_));
#endif // CL_VERSION_1_2
                AURA_OPENCL_SAFE_CALL(clReleaseContext(context_));
        }

        /// @copydoc boost::aura::base::cuda::device::get_base_device()
        inline const cl_device_id &get_base_device() const
        {
                return device_;
        }

        /// @copydoc boost::aura::base::cuda::device::get_base_conext()
        inline const cl_context &get_base_context() const
        {
                return context_;
        }

        /// @copydoc boost::aura::base::cuda::device::get_ordinal()
        inline std::size_t get_ordinal() const
        {
                return ordinal_;
        }

        /// @copydoc boost::aura::base::cuda::device::activate()
        inline void activate() const
        {
                // Pass
        }

        /// @copydoc boost::aura::base::cuda::device::deactivate()
        inline void deactivate() const
        {
                // Pass
        }

private:
        /// Device ordinal
        std::size_t ordinal_;

        /// Device handle
        cl_device_id device_;

        /// Context handle
        cl_context context_;
};

} // namespace opencl
} // namespace base_detail
} // namespace aura
} // namespace boost
