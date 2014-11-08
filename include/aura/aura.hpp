#ifndef AURA_AURA_HPP
#define AURA_AURA_HPP

/*
get list using:
ls *.hpp | xargs -n1 basename | \
awk '{print "#include <aura/"$0">"}'
*/

#include <aura/backend.hpp>
#include <aura/bounds.hpp>
#include <aura/config.hpp>
#include <aura/copy.hpp>
#include <aura/device_array.hpp>
#include <aura/device_buffer.hpp>
#include <aura/device_map.hpp>
#include <aura/device_view.hpp>
#include <aura/error.hpp>

#if defined AURA_FFT_CLFFT || defined AURA_FFT_CUFFT
	#include <aura/fft.hpp>
#endif

#include <aura/stream.hpp>

#endif // AURA_AURA_HPP

