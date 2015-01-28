#ifndef AURA_BACKEND_CUDA_FFT_HPP
#define AURA_BACKEND_CUDA_FFT_HPP

#include <boost/move/move.hpp>
#include <cuda.h>
#include <cufft.h>
#include <boost/aura/backend/cuda/call.hpp>
#include <boost/aura/backend/cuda/device_ptr.hpp>
#include <boost/aura/detail/svec.hpp>
#include <boost/aura/backend/cuda/device.hpp>
#include <boost/aura/bounds.hpp>

namespace boost
{
namespace aura
{
namespace backend_detail
{
namespace cuda
{

typedef int fft_size;
typedef svec<fft_size, 3> fft_embed;
using ::boost::aura::bounds;

/**
 * fft class
 */
class fft
{

private:
	BOOST_MOVABLE_BUT_NOT_COPYABLE(fft)

public:

	enum type {
		r2c,  // real to complex
		c2r,  // complex to real
		c2c,  // complex to complex
		d2z,  // double to double-complex
		z2d,  // double-complex to double
		z2z   // double-complex to double-complex
	};

	enum direction {
		fwd = CUFFT_FORWARD,
		inv = CUFFT_INVERSE
	};

	/**
	 * create empty fft object without device and stream
	 */
	inline explicit fft() : context_(nullptr)
	{
	}

	/**
	 * create fft
	 *
	 * @param d device to create fft for
	 */
	inline explicit fft(device& d, feed& f, 
			const bounds& dim, const fft::type& type, 
			std::size_t batch = 1,
	                const fft_embed& iembed = fft_embed(),
	                std::size_t istride = 1, std::size_t idist = 0,
	                const fft_embed& oembed = fft_embed(),
	                std::size_t ostride = 1, std::size_t odist = 0) :
		context_(d.get_context()), type_(type),
		dim_(dim), batch_(batch)
	{
		initialize(iembed, istride, idist,
				oembed, ostride, odist);
	}

	/**
	 * create fft
	 *
	 * @param d device to create fft for
	 */
	inline explicit fft(device& d, feed& f, 
			std::tuple<bounds, bounds> const & dim, 
			const fft::type& type, 
	                const fft_embed& iembed = fft_embed(),
	                std::size_t istride = 1, std::size_t idist = 0,
	                const fft_embed& oembed = fft_embed(),
	                std::size_t ostride = 1, std::size_t odist = 0) :
		context_(d.get_context()), type_(type),
		dim_(std::get<0>(dim)), 
		batch_(std::max(1, product(std::get<1>(dim))))
	{
		initialize(iembed, istride, idist,
				oembed, ostride, odist);
	}

	/**
	 * move constructor, move fft information here, invalidate other
	 *
	 * @param f fft to move here
	 */
	fft(BOOST_RV_REF(fft) f) : 
		context_(f.context_), handle_(f.handle_), type_(f.type_)
	{
		f.context_ = nullptr;
	}

	/**
	 * move assignment, move fft information here, invalidate other
	 *
	 * @param f fft to move here
	 */
	fft& operator=(BOOST_RV_REF(fft) f)
	{
		finalize();
		context_= f.context_;
		handle_ = f.handle_;
		type_ = f.type_;
		f.context_ = nullptr;
		return *this;
	}

	/**
	 * destroy fft
	 */
	inline ~fft()
	{
		finalize();
	}

	/**
	 * set feed
	 */
	void set_feed(const feed& f)
	{
		AURA_CUFFT_SAFE_CALL(cufftSetStream(handle_, 
					f.get_backend_stream()));
	}

	/**
	 * return fft handle
	 */
	const cufftHandle& get_handle() const
	{
		return handle_;
	}

	/**
	 * return fft type
	 */
	const type& get_type() const
	{
		return type_;
	}

	/// map fft type to cufftType
	cufftType map_type(fft::type type)
	{
		switch(type) {
			case r2c:
				return CUFFT_R2C;
			case c2r:
				return CUFFT_C2R;
			case c2c:
				return CUFFT_C2C;
			case d2z:
				return CUFFT_D2Z;
			case z2d:
				return CUFFT_Z2D;
			case z2z:
				return CUFFT_Z2Z;
			default:
				return (cufftType)0;
		}
	}

private:
	inline void initialize(const fft_embed& iembed = fft_embed(),
	                std::size_t istride = 1, std::size_t idist = 0,
	                const fft_embed& oembed = fft_embed(),
	                std::size_t ostride = 1, std::size_t odist = 0)
	{
		context_->set();
		AURA_CUFFT_SAFE_CALL(
		        cufftPlanMany(
		                &handle_,
		                dim_.size(),
		                const_cast<int*>(&dim_[0]),
		                0 == iembed.size() ? NULL : 
					const_cast<int*>(&iembed[0]),
		                istride,
		                0 == idist ? product(dim_) : idist,
		                0 == oembed.size() ? NULL : 
					const_cast<int*>(&oembed[0]),
		                ostride,
		                0 == odist ? product(dim_) : odist,
		                map_type(type_),
		                batch_
		        )
		);
		context_->unset();
	}

	/// finalize object (called from dtor and move assign)
	void finalize()
	{
		if(nullptr != context_) {
			context_->set();
			AURA_CUFFT_SAFE_CALL(cufftDestroy(handle_));
			context_->unset();
		}
	}

	/**
	 * check if the handle is a valid handle
	 */
	bool valid() 
	{
		if (context_ == nullptr) {
			return false;	
		}
		return true;
	}

protected:
	/// device context
	detail::context * context_;

private:
	/// fft handle
	cufftHandle handle_;
	/// fft type
	type type_;
	/// fft dims
	bounds dim_;
	/// batch
	std::size_t batch_;

	template <typename T1, typename T2>
	friend void fft_forward(device_ptr<T2> src, device_ptr<T1> dst,
	                        fft& plan, const feed& f);
	template <typename T1, typename T2>
	friend void fft_inverse(device_ptr<T2> src, device_ptr<T1> dst,
	                        fft& plan, const feed& f);
};

/// initialize fft library
inline void fft_initialize()
{
}
/// finalize fft library and release all associated resources
inline void fft_terminate()
{
}

/**
 * @brief calculate forward fourier transform
 *
 * @param dst pointer to result of fourier transform
 * @param src pointer to input of fourier transform
 * @param plan that is used to calculate the fourier transform
 * @param f feed the fourier transform should be calculated in
 */
template <typename T1, typename T2>
void fft_forward(device_ptr<T2> src, device_ptr<T1> dst,
                 fft& plan, const feed& f)
{
	plan.context_->set();
	plan.set_feed(f);
	switch(plan.get_type()) {
	case fft::type::r2c: {
		AURA_CUFFT_SAFE_CALL(
		        cufftExecR2C(
		                plan.get_handle(),
		                (cufftReal*)src.get(),
		                (cufftComplex*)dst.get())
		);
		break;
	}
	case fft::type::c2r: {
		assert(false); // FIXME
		break;
	}
	case fft::type::c2c: {
		AURA_CUFFT_SAFE_CALL(
		        cufftExecC2C(
		                plan.get_handle(),
		                (cufftComplex*)src.get(),
		                (cufftComplex*)dst.get(),
		                fft::direction::fwd)
		);
		break;
	}
	case fft::type::d2z: {
		AURA_CUFFT_SAFE_CALL(
		        cufftExecD2Z(
		                plan.get_handle(),
		                (cufftDoubleReal*)src.get(),
		                (cufftDoubleComplex*)dst.get())
		);
		break;
	}
	case fft::type::z2d: {
		assert(false); // FIXME
		break;
	}
	case fft::type::z2z: {
		AURA_CUFFT_SAFE_CALL(
		        cufftExecZ2Z(
		                plan.get_handle(),
		                (cufftDoubleComplex*)src.get(),
		                (cufftDoubleComplex*)dst.get(),
		                fft::direction::fwd)
		);
		break;
	}
	}
	plan.context_->unset();
}


/**
 * @brief calculate forward fourier transform
 *
 * @param dst pointer to result of fourier transform
 * @param src pointer to input of fourier transform
 * @param plan that is used to calculate the fourier transform
 * @param f feed the fourier transform should be calculated in
 */
template <typename T1, typename T2>
void fft_inverse(device_ptr<T2> src, device_ptr<T1> dst,
                 fft& plan, const feed& f)
{
	plan.context_->set();
	plan.set_feed(f);
	switch(plan.get_type()) {
	case fft::type::r2c: {
		assert(false); // FIXME
		break;
	}
	case fft::type::c2r: {
		AURA_CUFFT_SAFE_CALL(
		        cufftExecC2R(
		                plan.get_handle(),
		                (cufftComplex*)src.get(),
		                (cufftReal*)dst.get())
		);
		break;
	}
	case fft::type::c2c: {
		AURA_CUFFT_SAFE_CALL(
		        cufftExecC2C(
		                plan.get_handle(),
		                (cufftComplex*)src.get(),
		                (cufftComplex*)dst.get(),
		                fft::direction::inv)
		);
		break;
	}
	case fft::type::d2z: {
		assert(false); // FIXME
		break;
	}
	case fft::type::z2d: {
		AURA_CUFFT_SAFE_CALL(
		        cufftExecZ2D(
		                plan.get_handle(),
		                (cufftDoubleComplex*)src.get(),
		                (cufftDoubleReal*)dst.get())
		);
		break;
	}
	case fft::type::z2z: {
		AURA_CUFFT_SAFE_CALL(
		        cufftExecZ2Z(
		                plan.get_handle(),
		                (cufftDoubleComplex*)src.get(),
		                (cufftDoubleComplex*)dst.get(),
		                fft::direction::inv)
		);
		break;
	}
	}
	plan.context_->unset();
}
} // cuda
} // backend_detail
} // aura
} // boost

#endif // AURA_BACKEND_CUDA_FFT_HPP

