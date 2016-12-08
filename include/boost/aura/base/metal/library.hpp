#pragma once

#include <boost/aura/base/alang.hpp>
#include <boost/aura/base/metal/alang.hpp>
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
                : initialized_(false)
        {}

        /// Prevent copies.
        library(const library&) = delete;
        void operator=(const library&) = delete;

        /// Create library from string.
        inline explicit library(
                const std::string& kernelstring,
                device& d,
                bool inject_aura_preamble = true,
                const std::string& options = "")
                : initialized_(true)
                , device_(&d)
        {
                create_from_string(kernelstring, options, inject_aura_preamble);
        }

        /// Create library from file.
        inline explicit library(
                boost::aura::path p,
                device& d,
                bool inject_aura_preamble = true,
                const std::string& options = "")
                : initialized_(true)
                , device_(&d)
        {
                auto kernelstring = boost::aura::read_all(p);
                create_from_string(kernelstring, options, inject_aura_preamble);
        }

        /// Move construct.
        library(library&& other)
                : initialized_(other.initialized_)
                , device_(other.device_)
                , library_(other.library_)
                , log_(other.log_)
        {
                other.initialized_ = false;
                other.device_ = nullptr;
                other.library_ = nullptr;
                other.log_ = "";
        }

        /// Move assign.
        library& operator=(library&& other)
        {
                reset();

                initialized_ = other.initialized_;
                device_ = other.device_;
                library_ = other.library_;
                log_ = other.log_;

                other.initialized_ = false;
                other.device_ = nullptr;
                other.library_ = nullptr;
                other.log_ = "";
                return *this;
        }

        /// Access device.
        const device& get_device()
        {
                AURA_CHECK_INITIALIZED(initialized_);
                return *device_;
        }

        /// Access library.
        id<MTLLibrary> get_base_library()
        {
                AURA_CHECK_INITIALIZED(initialized_);
                return library_;
        }

        const id<MTLLibrary> get_base_library() const
        {
                AURA_CHECK_INITIALIZED(initialized_);
                return library_;
        }

        /// Destructor.
        ~library() { reset(); }

        /// Finalize object.
        void reset()
        {
                if (initialized_)
                {
                        library_ = nullptr;
                        initialized_ = false;
                }
                device_ = nullptr;
                log_ = "";
        }

private:
        /// Create a library from a string.
        void create_from_string(
                const std::string& kernelstring,
                const std::string& opt,
                bool inject_aura_preamble)
        {
                shared_alang_header salh;
                alang_header alh;

                std::string kernelstring_with_preamble = kernelstring;
                if (inject_aura_preamble)
                {
                        // Prepend AURA define to kernel.
                        kernelstring_with_preamble =
                                std::string("#define AURA_BASE_METAL\n") +
                                salh.get() +
                                std::string("\n") +
                                alh.get() +
                                std::string("\n") +
                                kernelstring_with_preamble;
                }

                NSError* err;
                library_ = [device_->get_base_device()
                        newLibraryWithSource:
                                [NSString stringWithUTF8String:
                                                  kernelstring_with_preamble
                                                          .c_str()]
                                     options:nil
                                       error:&err];

                if (!library_)
                {
                        NSLog(@"Error: %@ %@", err, [err userInfo]);
                }
                AURA_METAL_CHECK_ERROR(library_);
        }

        /// Initialized flag
        bool initialized_;

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
