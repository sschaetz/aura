#pragma once

#include <boost/aura/bounds.hpp>
#include <boost/aura/device.hpp>
#include <boost/aura/device_ptr.hpp>

namespace boost
{
namespace aura
{

namespace detail
{

/// Deleter used for unique ptr to deallocate device data.
template <typename T>
struct device_array_deleter
{
        void operator()(device_ptr<T>* p) const
        {
                device_free(*p);
                free(p);
        }
};

} // namespace detail

/// Device array, owns device memory, can have multiple dimensions.
template <typename T, typename BoundsType = bounds>
class device_array
{

public:
        /// Convenience types
        typedef device_ptr<T> iterator;
        typedef const device_ptr<T> const_iterator;
        typedef T value_type;

        // Prevent copies
        device_array(const device_array&) = delete;
        void operator=(const device_array&) = delete;

        /// Create empty array
        device_array()
        {
        }

        /// Create one-dimensional array of size on device
        device_array(std::size_t size, device& d)
                : bounds_({size})
        {
                allocate(size, d);
        }

        /// Create multi-dimensional array of of bound size b on device
        device_array(const BoundsType& b, device& d)
                : bounds_(b)
        {
                allocate(product(b), d);
        }

        /// destroy object
        ~device_array()
        {
        }

        // move constructor, move device_array here, invalidate other
        // @param da device_array to move here
        device_array(device_array&& da)
                : bounds_(std::move(da.bounds_))
                , data_(std::move(da.data_))
        {
                da.bounds_.clear();
        }

        /// move assignment, move device_array here, invalidate other
        /// @param da device_array to move here
        device_array& operator=(device_array&& da)
        {
                bounds_ = da.bounds_;
                data_ = boost::move(da.data_);
                da.bounds_.clear();
                return *this;
        }

private:
        typedef detail::device_array_deleter<T> deleter_t;
        typedef std::unique_ptr<device_ptr<T>, deleter_t> data_t;

        data_t allocate(std::size_t size, device& d)
        {
                auto ptr = new device_ptr<T>;
                *ptr = device_malloc<T>(size, d);
                return data_t(ptr, deleter_t());
        }

        /// Stores the bounds.
        BoundsType bounds_;

        /// Holds data.
        data_t data_;
};

} // namespace aura
} // namespace boost
