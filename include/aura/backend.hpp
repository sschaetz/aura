
#ifndef AURA_BACKEND_HPP
#define AURA_BACKEND_HPP

#include <aura/config.hpp>

#if defined AURA_BACKEND_CUDA
	#if defined __CUDACC__
		#include <aura/backend/cuda/kernel_helper.hpp>
	#else
		#include <aura/backend/cuda/args.hpp>
		#include <aura/backend/cuda/bundle.hpp>
		#include <aura/backend/cuda/call.hpp>
		#include <aura/backend/cuda/device.hpp>
		#include <aura/backend/cuda/device_ptr.hpp>
		#include <aura/backend/cuda/mark.hpp>
		#include <aura/backend/cuda/feed.hpp>
		#include <aura/backend/cuda/fft.hpp>
		#include <aura/backend/cuda/mesh.hpp>
		#include <aura/backend/cuda/init.hpp>
		#include <aura/backend/cuda/invoke.hpp>
		#include <aura/backend/cuda/kernel.hpp>
		#include <aura/backend/cuda/memory.hpp>
		#include <aura/backend/cuda/module.hpp>
		#include <aura/backend/cuda/p2p.hpp>
		#include <aura/backend/cuda/detail/host_allocator.hpp>
	#endif // defined __CUDACC__

#elif defined AURA_BACKEND_OPENCL

	#if defined __OPENCL_VERSION__
		#include <aura/backend/opencl/kernel_helper.hpp>
	#else
		#include <aura/backend/opencl/args.hpp>
		#include <aura/backend/opencl/bundle.hpp>
		#include <aura/backend/opencl/call.hpp>
		#include <aura/backend/opencl/device.hpp>
		#include <aura/backend/opencl/device_ptr.hpp>
		#include <aura/backend/opencl/feed.hpp>
	#if AURA_FFT_CLFFT
		#include <aura/backend/opencl/fft.hpp>
	#endif // AURA_FFT_CLFFT
		#include <aura/backend/opencl/mesh.hpp>
		#include <aura/backend/opencl/init.hpp>
		#include <aura/backend/opencl/invoke.hpp>
		#include <aura/backend/opencl/kernel.hpp>
		#include <aura/backend/opencl/mark.hpp>
		#include <aura/backend/opencl/memory.hpp>
		#include <aura/backend/opencl/module.hpp>
		#include <aura/backend/opencl/detail/host_allocator.hpp>
	#endif // defined __OPENCL_VERSION__

#endif

#if !defined __OPENCL_VERSION__ && !defined __CUDACC__
#include <aura/backend/shared/call.hpp>

namespace aura {

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

} // namespace aura

#include <aura/detail/host_allocator_traits.hpp>

#endif // !defined __OPENCL_VERSION__ && !defined __CUDACC__

#endif // AURA_BACKEND_HPP

