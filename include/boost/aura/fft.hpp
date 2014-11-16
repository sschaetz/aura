#ifndef AURA_FFT_HPP
#define AURA_FFT_HPP

#include <boost/aura/backend.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/bounds.hpp>

namespace boost {
namespace aura {

using backend::fft_size;
using backend::fft_embed;

using backend::fft;
using backend::fft_initialize;
using backend::fft_terminate;
using backend::fft_forward;
using backend::fft_inverse;

template <typename T1, typename T2>
void fft_forward(device_array<T2> & src, device_array<T1> & dst,
		fft& plan, const feed& f)
{
	fft_forward<T1, T2>(src.begin(), dst.begin(), plan, f);
}

template <typename T1, typename T2>
void fft_inverse(device_array<T2> & src, device_array<T1> & dst,
		fft& plan, const feed& f)
{
	fft_inverse<T1, T2>(src.begin(), dst.begin(), plan, f);
}

} // aura
} // boost 

#endif // AURA_FFT_HPP

