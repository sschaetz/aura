#pragma once

#import <Metal/Metal.h>

namespace boost {
namespace aura {
namespace base_detail {
namespace metal {

class device
{
public:
        /**
         * Create device form ordinal.
         *
         * @param ordinal Device number
         */
        inline explicit device(std::size_t ordinal)
                : ordinal_(ordinal)
        {
                device_ = MTLCreateSystemDefaultDevice();
        }

private:
        /// Device ordinal
        std::size_t ordinal_;

        /// Device handle
        id<MTLDevice> device_;
};

} // namespace metal
} // namespace base_detail
} // namespace aura
} // namespace boost
