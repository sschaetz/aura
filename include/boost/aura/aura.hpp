#ifndef AURA_AURA_HPP
#define AURA_AURA_HPP

/*
get list using:
ls *.hpp | xargs -n1 basename | \
awk '{print "#include <boost/aura/"$0">"}'
*/

#include <boost/aura/backend.hpp>
#include <boost/aura/bounds.hpp>
#include <boost/aura/config.hpp>
#include <boost/aura/copy.hpp>
#include <boost/aura/device_array.hpp>
#include <boost/aura/device_buffer.hpp>
#include <boost/aura/device_map.hpp>
#include <boost/aura/device_view.hpp>
#include <boost/aura/error.hpp>

#if defined AURA_FFT_CLFFT || defined AURA_FFT_CUFFT
	#include <boost/aura/fft.hpp>
#endif

#endif // AURA_AURA_HPP

