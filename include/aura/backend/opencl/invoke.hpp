#ifndef AURA_BACKEND_OPENCL_INVOKE_HPP
#define AURA_BACKEND_OPENCL_INVOKE_HPP

#include <assert.h>
#include <cstddef>
#include <CL/cl.h>
#include <aura/bounds.hpp>
#include <aura/detail/svec.hpp>
#include <aura/backend/opencl/kernel.hpp>
#include <aura/backend/opencl/call.hpp>
#include <aura/backend/opencl/feed.hpp>
#include <aura/backend/opencl/mesh.hpp>
#include <aura/backend/opencl/bundle.hpp>
#include <aura/backend/opencl/args.hpp>
#include <aura/backend/shared/calc_mesh_bundle.hpp>

namespace aura {
namespace backend_detail {
namespace opencl {

namespace detail {

void invoke_impl(kernel & k, const mesh & m, const bundle & b, 
	const args_t & a, feed & f) 
{
	// set parameters
	for (std::size_t i=0; i<a.second.size(); i++) {
		AURA_OPENCL_SAFE_CALL(clSetKernelArg(k, i, 
					a.second[i].second, a.second[i].first));
	}
	// handling for non 3-dimensional mesh and bundle sizes
	mesh tm;
	tm.push_back(m[0]);
	tm.push_back(1);
	tm.push_back(1);
	bundle tb; 
	tb.push_back(b[0]);
	tb.push_back(1);
	tb.push_back(1);

	if (m.size() > 1) {
		tm[1] = m[1];
	}
	if (m.size() > 2) {
		tm[2] = m[2];
	}

	if (b.size() > 1) {
		tb[1] = b[1];
	}
	if (b.size() > 2) {
		tb[2] = b[2];
	}

	// call kernel
	AURA_OPENCL_SAFE_CALL(clEnqueueNDRangeKernel(
		f.get_backend_stream(), k, tm.size(), NULL, 
		&tm[0], &tb[0], 0, NULL, NULL)); 
	free(a.first);
} 

void invoke_impl(kernel & k, const bounds& b, const args_t & a, feed & f)
{
	// set parameters
	for (std::size_t i=0; i<a.second.size(); i++) {
		AURA_OPENCL_SAFE_CALL(clSetKernelArg(k, i, 
					a.second[i].second, a.second[i].first));
	}
	std::array<std::size_t, 4> mb = {{1, 1, 1, 1}};
	const std::array<std::size_t, 4> max_mb = {{
		AURA_OPENCL_MAX_BUNDLE, 
		AURA_OPENCL_MAX_MESH0, 
		AURA_OPENCL_MAX_MESH1, 
		AURA_OPENCL_MAX_MESH2 
	}};

	// OpenCL has a different behaviour here
	std::array<bool, 4> mask = {{false, true, false, false}};
	
	aura::detail::calc_mesh_bundle(product(b), 2, 
			mb.begin(), max_mb.begin(), mask.begin());

	mesh tm;
	tm.push_back(mb[1]);
	tm.push_back(mb[2]);
	tm.push_back(mb[3]);
	bundle tb; 
	tb.push_back(mb[0]);
	tb.push_back(1);
	tb.push_back(1);
	
	// call kernel
	AURA_OPENCL_SAFE_CALL(clEnqueueNDRangeKernel(
		f.get_backend_stream(), k, tm.size(), NULL, 
		&tm[0], &tb[0], 0, NULL, NULL)); 
	free(a.first);
}

} // namespace detail


/// invoke kernel without args
void invoke(kernel & k, const mesh & m, const bundle & b, feed & f) 
{
	detail::invoke_impl(k, m, b, args_t(), f);
}

/// invoke kernel with args
void invoke(kernel & k, const mesh & m, const bundle & b, 
		const args_t & a, feed & f) 
{
	detail::invoke_impl(k, m, b, a, f);
}

/// invoke kernel with bounds and args
void invoke(kernel& k, const bounds& b, const args_t& a, feed& f) 
{
	detail::invoke_impl(k, b, a, f);
}

} // namespace opencl
} // namespace backend_detail
} // namespace aura

#endif // AURA_BACKEND_OPENCL_INVOKE_HPP

