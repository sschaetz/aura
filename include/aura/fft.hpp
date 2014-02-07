#ifndef AURA_FFT_HPP
#define AURA_FFT_HPP

#include <aura/backend.hpp>
#include <aura/device_array.hpp>
#include <aura/bounds.hpp>

namespace aura {

using backend::fft_size;
using backend::fft_embed;

using backend::fft;
using backend::fft_initialize;
using backend::fft_terminate;
using backend::fft_forward;
using backend::fft_inverse;

template <typename T1, typename T2>
void fft_forward(device_array<T1> & dst, device_array<T2> & src,
		fft& plan, const feed& f)
{
	fft_forward<T1, T2>(dst.begin(), src.begin(), plan, f);
}

template <typename T1, typename T2>
void fft_inverse(device_array<T1> & dst, device_array<T2> & src,
		fft& plan, const feed& f)
{
	fft_inverse<T1, T2>(dst.begin(), src.begin(), plan, f);
}

} // aura

#endif // AURA_FFT_HPP

