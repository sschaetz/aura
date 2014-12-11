
#ifndef AURA_BACKEND_HPP
#define AURA_BACKEND_HPP

#include <boost/aura/config.hpp>

#if defined AURA_BACKEND_CUDA
	#if defined __CUDACC__
		#include <boost/aura/backend/cuda/kernel/atomic.hpp>
		#include <boost/aura/backend/cuda/kernel/complex.hpp>
		#include <boost/aura/backend/cuda/kernel/kernel_helper.hpp>
	#else
		#include <boost/aura/backend/cuda/args.hpp>
		#include <boost/aura/backend/cuda/bundle.hpp>
		#include <boost/aura/backend/cuda/call.hpp>
		#include <boost/aura/backend/cuda/device.hpp>
		#include <boost/aura/backend/cuda/device_ptr.hpp>
		#include <boost/aura/backend/cuda/mark.hpp>
		#include <boost/aura/backend/cuda/feed.hpp>
		#include <boost/aura/backend/cuda/fft.hpp>
		#include <boost/aura/backend/cuda/mesh.hpp>
		#include <boost/aura/backend/cuda/init.hpp>
		#include <boost/aura/backend/cuda/invoke.hpp>
		#include <boost/aura/backend/cuda/kernel.hpp>
		#include <boost/aura/backend/cuda/memory.hpp>
		#include <boost/aura/backend/cuda/module.hpp>
		#include <boost/aura/backend/cuda/p2p.hpp>
	#endif // defined __CUDACC__

#elif defined AURA_BACKEND_OPENCL

	#if defined __OPENCL_VERSION__
		#include <boost/aura/backend/opencl/kernel/atomic.hpp>
		#include <boost/aura/backend/opencl/kernel/complex.hpp>
		#include <boost/aura/backend/opencl/kernel/kernel_helper.hpp>
	#else
		#include <boost/aura/backend/opencl/args.hpp>
		#include <boost/aura/backend/opencl/bundle.hpp>
		#include <boost/aura/backend/opencl/call.hpp>
		#include <boost/aura/backend/opencl/device.hpp>
		#include <boost/aura/backend/opencl/device_ptr.hpp>
		#include <boost/aura/backend/opencl/feed.hpp>
	#if AURA_FFT_CLFFT
		#include <boost/aura/backend/opencl/fft.hpp>
	#endif // AURA_FFT_CLFFT
		#include <boost/aura/backend/opencl/mesh.hpp>
		#include <boost/aura/backend/opencl/init.hpp>
		#include <boost/aura/backend/opencl/invoke.hpp>
		#include <boost/aura/backend/opencl/kernel.hpp>
		#include <boost/aura/backend/opencl/mark.hpp>
		#include <boost/aura/backend/opencl/memory.hpp>
		#include <boost/aura/backend/opencl/module.hpp>
	#endif // defined __OPENCL_VERSION__

#endif

#if !defined __OPENCL_VERSION__ && !defined __CUDACC__
#include <boost/aura/backend/shared/call.hpp>

namespace boost
{
namespace aura 
{

#ifdef AURA_BACKEND_CUDA
namespace backend = backend_detail::cuda;
#elif AURA_BACKEND_OPENCL
namespace backend = backend_detail::opencl;
#endif

using backend::initialize;

using backend::device;
using backend::device_info;
using backend::device_get_info;
using backend::device_get_count;
using backend::print_device_info;

using backend::feed;
using backend::wait_for;

using backend::device_ptr;

using backend::module;
using backend::kernel;
using backend::create_kernel;
using backend::mesh;
using backend::bundle;
using backend::args;

using backend::device_malloc;

} // namespace aura
} // namespace boost 

#endif // !defined __OPENCL_VERSION__ && !defined __CUDACC__

#endif // AURA_BACKEND_HPP

