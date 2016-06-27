#pragma once

#include <boost/aura/base/metal/device.hpp>
#include <boost/aura/base/metal/safecall.hpp>
#include <boost/aura/io.hpp>

#include <iostream>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace metal
{


class library
{
public:
        /// Create empty library.
        inline explicit library()
                : device_(nullptr)
        {
        }

        /// Prevent copies.
        library(const library&) = delete;
        void operator=(const library&) = delete;

        /// Create library from string.
        inline explicit library(const std::string& kernelstring, device& d,
                const std::string& options = "")
                : device_(&d)
        {
                create_from_string(kernelstring, options);
        }

        /// Create library from file.
        inline explicit library(
                boost::aura::path p, device& d, const std::string& options = "")
                : device_(&d)
        {
                auto kernelstring = boost::aura::read_all(p);
                create_from_string(kernelstring, options);
        }


        /// Access device.
        const device& get_device()
        {
                return *device_;
        }

        ~library()
        {
                finalize();
        }

private:
        /// Create a library from a string.
        void create_from_string(
                const std::string& kernelstring, const std::string& opt)
        {
                // Prepend AURA define to kernel.
                auto kernelstring_with_define =
                        std::string("#define AURA_BASE_METAL\n") + kernelstring;


                NSError* err;
                library_ = [device_->get_base_device()
                        newLibraryWithSource:
                                [NSString stringWithUTF8String:
                                                  kernelstring_with_define
                                                          .c_str()]
                                     options:nil
                                       error:&err];

                if (!library_)
                {
                        NSLog(@"Error: %@ %@", err, [err userInfo]);
                }
                AURA_METAL_CHECK_ERROR(library_);
        }

        /// Finalize object.
        void finalize()
        {
                if (device_ != nullptr)
                {
                        library_ = nil;
                }
        }

        /// Pointer to device the feed was created for
        device* device_;

        /// Library
        id<MTLLibrary> library_;

        /// Library compile log
        std::string log_;
};


} // metal
} // base_detail
} // aura
} // boost
