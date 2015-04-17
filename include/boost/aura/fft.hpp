#ifndef AURA_FFT_HPP
#define AURA_FFT_HPP

#include <boost/aura/backend.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/bounds.hpp>
#include <boost/aura/math/basic/mul.hpp>

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
void fft_forward(const device_array<T1> & src, device_array<T2> & dst,
		fft& plan, const feed& f)
{
	fft_forward<T1, T2>(src.begin(), dst.begin(), plan, f);
}

template <typename T1, typename T2>
void fft_inverse(const device_array<T1> & src, device_array<T2> & dst,
		fft& plan, const feed& f)
{
	fft_inverse<T1, T2>(src.begin(), dst.begin(), plan, f);
}

template <typename T1, typename T2>
void fft_forward(const device_range<T1> & src, device_range<T2> & dst,
		fft& plan, const feed& f)
{
	fft_forward<T1, T2>(src.begin(), dst.begin(), plan, f);
}

template <typename T1, typename T2>
void fft_inverse(const device_range<T1> & src, device_range<T2> & dst,
		fft& plan, const feed& f)
{
	fft_inverse<T1, T2>(src.begin(), dst.begin(), plan, f);
}

template <typename T1, typename T2>
void fft_forward(const device_range<T1> & src, device_array<T2> & dst,
        fft& plan, const feed& f)
{
    fft_forward<T1, T2>(src.begin(), dst.begin(), plan, f);
}

template <typename T1, typename T2>
void fft_inverse(const device_range<T1> & src, device_array<T2> & dst,
        fft& plan, const feed& f)
{
    fft_inverse<T1, T2>(src.begin(), dst.begin(), plan, f);
}

template <typename T1, typename T2>
void fft_forward(const device_array<T1> & src, device_range<T2> & dst,
        fft& plan, const feed& f)
{
    fft_forward<T1, T2>(src.begin(), dst.begin(), plan, f);
}

template <typename T1, typename T2>
void fft_inverse(const device_array<T1> & src, device_range<T2> & dst,
        fft& plan, const feed& f)
{
    fft_inverse<T1, T2>(src.begin(), dst.begin(), plan, f);
}

template <typename deviceRangeType1, typename deviceRangeType2>
void fft_forward_scaled(const deviceRangeType1 & src, deviceRangeType2 & dst,
        fft& plan, feed& f)
{
    // derive scaling factor and copy it to GPU
    bounds b = src.get_bounds();
    float cpuScale = 1/(sqrt(b[0])*sqrt(b[1]));
    device_array<float> scale(1,dst.get_device());
    scale.set_value(cpuScale,f);

    // perform fft
    fft_forward(src, dst, plan, f);

    // perform scaling
    math::mul(dst,scale,dst,f);

}
template <typename deviceRangeType1, typename deviceRangeType2>

void fft_inverse_scaled(const deviceRangeType1 & src, deviceRangeType2 & dst,
        fft& plan, feed& f)
{
    // derive scaling factor and copy it to GPU
    bounds b = src.get_bounds();
    float cpuScale = sqrt(b[0])*sqrt(b[1]);
    device_array<float> scale(1,dst.get_device());
    scale.set_value(cpuScale,f);

    // perform ifft
    fft_inverse(src, dst, plan, f);

    // perform scaling
    math::mul(dst,scale,dst,f);
}


} // aura
} // boost 

#endif // AURA_FFT_HPP

