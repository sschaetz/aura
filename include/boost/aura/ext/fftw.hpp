#ifndef AURA_BENCH_FFT_FFTW_HPP
#define AURA_BENCH_FFT_FFTW_HPP

#include <fftw3.h>
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
	inline explicit fft() : 
		handle_single_fwd_(nullptr),
		handle_single_inv_(nullptr),
		handle_double_fwd_(nullptr),
		handle_double_inv_(nullptr)
	{}

	/**
	 * create fft
	 */
	inline explicit fft(const bounds& dim, const fft::type& type, 
			std::size_t batch = 1,
	                const fft_embed& iembed = fft_embed(),
	                std::size_t istride = 1, std::size_t idist = 0,
	                const fft_embed& oembed = fft_embed(),
	                std::size_t ostride = 1, std::size_t odist = 0) :
		handle_single_fwd_(nullptr),
		handle_single_inv_(nullptr),
		handle_double_fwd_(nullptr),
		handle_double_inv_(nullptr),
		type_(type),
		dim_(dim), batch_(batch)
	{
		initialize(iembed, istride, idist, oembed, ostride, odist);
	}

	/**
	 * create fft with measured plan
	 */
	template <typename IT1, typename IT2>
	inline explicit fft(const bounds& dim, const fft::type& type, 
			IT1 in, IT2 out,		
			std::size_t batch = 1,
	                const fft_embed& iembed = fft_embed(),
	                std::size_t istride = 1, std::size_t idist = 0,
	                const fft_embed& oembed = fft_embed(),
	                std::size_t ostride = 1, std::size_t odist = 0) :
		handle_single_fwd_(nullptr),
		handle_single_inv_(nullptr),
		handle_double_fwd_(nullptr),
		handle_double_inv_(nullptr),
		type_(type),
		dim_(dim), 
		batch_(batch)
	{
		initialize(in, out, iembed, istride, idist, 
				oembed, ostride, odist);
	}

	/**
	 * create fft
	 */
	inline explicit fft(std::tuple<bounds, bounds> const & dim,
			const fft::type& type,
	                const fft_embed& iembed = fft_embed(),
	                std::size_t istride = 1, std::size_t idist = 0,
	                const fft_embed& oembed = fft_embed(),
	                std::size_t ostride = 1, std::size_t odist = 0) :
		handle_single_fwd_(nullptr),
		handle_single_inv_(nullptr),
		handle_double_fwd_(nullptr),
		handle_double_inv_(nullptr),
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
		handle_single_fwd_(f.handle_single_fwd_),
		handle_single_inv_(f.handle_single_inv_),
		handle_double_fwd_(f.handle_double_fwd_),
		handle_double_inv_(f.handle_double_inv_),
		type_(f.type_),
		dim_(f.dim_),
		batch_(f.batch_)
	{
		f.handle_single_fwd_ = nullptr;
		f.handle_single_inv_ = nullptr;
		f.handle_double_fwd_ = nullptr;
		f.handle_double_inv_ = nullptr;
	}

	/**
	 * move assignment, move fft information here, invalidate other
	 *
	 * @param f fft to move here
	 */
	fft& operator=(BOOST_RV_REF(fft) f)
	{
		finalize();

		handle_single_fwd_ = f.handle_single_fwd_;
		handle_single_inv_ = f.handle_single_inv_;
		handle_double_fwd_ = f.handle_double_fwd_;
		handle_double_inv_ = f.handle_double_inv_;
		type_ = f.type_;
		dim_ = f.dim_;
		batch_ = f.batch_;

		f.handle_single_fwd_ = nullptr;
		f.handle_single_inv_ = nullptr;
		f.handle_double_fwd_ = nullptr;
		f.handle_double_inv_ = nullptr;

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

	bool is_single()
	{
		return is_single(type_);
	}

	/// true if double precision FFT is requested
	bool is_double(fft::type type)
	{
		return !is_single(type);
	}

	bool is_double()
	{
		return is_double(type_);
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

	bool is_c2c()
	{
		return is_c2c(type_);
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
	
	bool is_r2c() 
	{
		return is_r2c(type_);
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

	bool is_c2r()
	{
		return is_c2r(type_);
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
						fwd,
						FFTW_ESTIMATE|FFTW_UNALIGNED);

				handle_single_inv_ = 
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
						inv,
						FFTW_ESTIMATE|FFTW_UNALIGNED);
			}
		}
	}
	
	template <typename IT1, typename IT2>
	inline void initialize(IT1 in, IT2 out,
			const fft_embed& iembed = fft_embed(),
	                std::size_t istride = 1, std::size_t idist = 0,
	                const fft_embed& oembed = fft_embed(),
	                std::size_t ostride = 1, std::size_t odist = 0)
	{
		typedef fftwf_complex* sptr;
		if (is_single(type_)) {
			if (is_c2c(type_)) {
				handle_single_fwd_ = 
					fftwf_plan_many_dft(
						dim_.size(), 
						const_cast<int*>(&dim_[0]), 
						batch_,
						reinterpret_cast<sptr>(&(*in)),
						0 == iembed.size() ? NULL : 
							const_cast<int*>(
								&iembed[0]),
						istride,
						0 == idist ? product(dim_) : 
							idist,
						reinterpret_cast<sptr>(&(*out)), 
						0 == oembed.size() ? NULL : 
							const_cast<int*>(
								&oembed[0]),
						ostride,
						0 == odist ? product(dim_) : 
							odist,
						fwd,
						FFTW_PATIENT);

				handle_single_inv_ = 
					fftwf_plan_many_dft(
						dim_.size(), 
						const_cast<int*>(&dim_[0]), 
						batch_,
						reinterpret_cast<sptr>(&(*in)),
						0 == iembed.size() ? NULL : 
							const_cast<int*>(
								&iembed[0]),
						istride,
						0 == idist ? product(dim_) : 
							idist,
						reinterpret_cast<sptr>(&(*out)),
						0 == oembed.size() ? NULL : 
							const_cast<int*>(
								&oembed[0]),
						ostride,
						0 == odist ? product(dim_) : 
							odist,
						inv,
						FFTW_PATIENT);
			}
		}
	}

	/// finalize object (called from dtor and move assign)
	void finalize()
	{
		
		if (handle_single_fwd_ != nullptr) {
			fftwf_destroy_plan(handle_single_fwd_);
		}
		if (handle_single_inv_ != nullptr) {
			fftwf_destroy_plan(handle_single_inv_);
		}
		if (handle_double_fwd_ != nullptr) {
			fftw_destroy_plan(handle_double_fwd_);
		}
		if (handle_double_inv_ != nullptr) {
			fftw_destroy_plan(handle_double_inv_);
		}
	}

private:
	/// fft handle
	fftwf_plan handle_single_fwd_;
	fftwf_plan handle_single_inv_;
	fftw_plan handle_double_fwd_;
	fftw_plan handle_double_inv_;

	/// fft type
	type type_;
	/// fft dims
	bounds dim_;
	/// batch
	std::size_t batch_;

	template <typename IT1, typename IT2>
	friend void fft_forward(IT1 in, IT2 out, fft& plan);
	template <typename IT1, typename IT2>
	friend void fft_inverse(IT1 in, IT2 out, fft& plan);
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
template <typename IT1, typename IT2>
void fft_forward(IT1 in, IT2 out, fft& plan)
{
	// C-style cast because FFTW interface wants this type non-const
	if (plan.is_single()) {
		if (plan.is_c2c()) {
			fftwf_execute_dft(plan.handle_single_fwd_,
				(fftwf_complex*)(&(*in)),
				reinterpret_cast<fftwf_complex*>(&(*out)));
		}
	}
}


/**
 * @brief calculate forward fourier transform
 *
 * @param dst pointer to result of fourier transform
 * @param src pointer to input of fourier transform
 * @param plan that is used to calculate the fourier transform
 * @param f feed the fourier transform should be calculated in
 */
template <typename IT1, typename IT2>
void fft_inverse(IT1 in, IT2 out, fft& plan)
{
	if (plan.is_single()) {
		if (plan.is_c2c()) {
			fftwf_execute_dft(plan.handle_single_inv_,
				(fftwf_complex*)(&(*in)), 
				reinterpret_cast<fftwf_complex*>(&(*out)));
		}
	}
}

} // fftw 
} // aura
} // boost

#endif // AURA_BENCH_FFT_FFTW_HPP

