#pragma once

namespace boost
{
namespace aura
{
namespace platform
{

// Shared memory.
#if defined AURA_BASE_CUDA
        constexpr bool supports_shared_memory = false;
#elif defined AURA_BASE_METAL
        constexpr bool supports_shared_memory = true;
#elif defined AURA_BASE_OPENCL
        constexpr bool supports_shared_memory = false;
#endif

// Preferred memory alignment.
#if defined AURA_BASE_CUDA
        constexpr std::size_t memory_alignment = 32;
#elif defined AURA_BASE_METAL
        // On metal if we align to page size we can
        // use CPU memory directly on GPU.
        constexpr std::size_t memory_alignment = 16384;
#elif defined AURA_BASE_OPENCL
        constexpr std::size_t memory_alignment = 32;
#endif

} // platform
} // aura
} // boost
