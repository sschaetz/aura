#ifndef AURA_BENCH_FFT_FFTW_HPP
#define AURA_BENCH_FFT_FFTW_HPP

#include <boost/move/move.hpp>
#include <boost/aura/detail/svec.hpp>
#include <boost/aura/bounds.hpp>

namespace boost
{
namespace aura
{
namespace fftw 
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
		fwd = FFTW_FORWARD,
		inv = FFTW_BACKWARD
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
	inline explicit fft(const bounds& dim, const fft::type& type, 
			std::size_t batch = 1,
	                const fft_embed& iembed = fft_embed(),
	                std::size_t istride = 1, std::size_t idist = 0,
	                const fft_embed& oembed = fft_embed(),
	                std::size_t ostride = 1, std::size_t odist = 0) :
		context_(d.get_context()), type_(type),
		dim_(dim), batch_(batch)
	{
		initialize(iembed, istride, idist, oembed, ostride, odist);
	}

	/**
	 * create fft
	 *
	 * @param d device to create fft for
	 */
	inline explicit fft(std::tuple<bounds, bounds> const & dim,
			const fft::type& type,
	                const fft_embed& iembed = fft_embed(),
	                std::size_t istride = 1, std::size_t idist = 0,
	                const fft_embed& oembed = fft_embed(),
	                std::size_t ostride = 1, std::size_t odist = 0) :
		type_(type),
		dim_(std::get<0>(dim)), 
		batch_(std::max(1, product(std::get<1>(dim))))
	{
		initialize(iembed, istride, idist, oembed, ostride, odist);
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
	 * return fft type
	 */
	const type& get_type() const
	{
		return type_;
	}

	/// true if single precision FFT is requested 
	bool is_single(fft::type type)
	{	
		switch(type) {
			case r2c:
				return true;
			case c2r:
				return true;
			case c2c:
				return true;
			default:
				return false;
		}
	}

	/// true if double precision FFT is requested
	bool is_double(fft::type type)
	{
		return !is_single(type);
	}

	/// true if complex to complex (single or double) FFT is requested
	bool is_c2c(fft::type type)
	{
		switch(type) {
			case c2c:
				return true;
			case z2z:
				return true;
			default:
				return false;
		}
	}

	/// true if real to complex (single or double) FFT is requested
	bool is_r2c(fft::type type)
	{
		switch(type) {
			case r2c:
				return true;
			case d2z:
				return true;
			default:
				return false;
		}
	}
	
	/// true if complex to real (single or double) FFT is requested
	bool is_c2r(fft::type type)
	{
		switch(type) {
			case c2r:
				return true;
			case z2d:
				return true;
			default:
				return false;
		}
	}

private:
	inline void initialize(const fft_embed& iembed = fft_embed(),
	                std::size_t istride = 1, std::size_t idist = 0,
	                const fft_embed& oembed = fft_embed(),
	                std::size_t ostride = 1, std::size_t odist = 0)
	{
		if (is_single(type_)) {
			if (is_c2c(type_)) {
				handle_single_fwd_ = 
					fftwf_plan_many_dft(
						dim_.size(), 
						const_cast<int*>(&dim_[0]), 
						batch_,
						NULL, 
						0 == iembed.size() ? NULL : 
							const_cast<int*>(
								&iembed[0]),
						istride,
						0 == idist ? product(dim_) : 
							idist,
						NULL,
						0 == oembed.size() ? NULL : 
							const_cast<int*>(
								&oembed[0]),
						ostride,
						0 == odist ? product(dim_) : 
							odist,
						forward,
						FFTW_ESTIMATE|FFTW_UNALIGNED);

				handle_single_bwd_ = 
					fftwf_plan_many_dft(
						dim_.size(), 
						const_cast<int*>(&dim_[0]), 
						batch_,
						NULL, 
						0 == iembed.size() ? NULL : 
							const_cast<int*>(
								&iembed[0]),
						istride,
						0 == idist ? product(dim_) : 
							idist,
						NULL,
						0 == oembed.size() ? NULL : 
							const_cast<int*>(
								&oembed[0]),
						ostride,
						0 == odist ? product(dim_) : 
							odist,
						backward,
						FFTW_ESTIMATE|FFTW_UNALIGNED);
			}
		}
	}

	/// finalize object (called from dtor and move assign)
	void finalize()
	{
		
	}

private:
	/// fft handle
	fftwf_plan handle_single_fwd_;
	fftwf_plan handle_single_bwd_;
	fftw_plan handle_double_fwd_;
	fftw_plan handle_double_bwd_;

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
	fftw_init_threads();
}

/// finalize fft library and release all associated resources
inline void fft_terminate()
{
	fftw_cleanup_threads();
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
}

} // fftw 
} // aura
} // boost

#endif // AURA_BENCH_FFT_FFTW_HPP

