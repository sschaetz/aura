#pragma once

#include <boost/aura/bounds.hpp>
#include <boost/aura/device.hpp>
#include <boost/aura/device_ptr.hpp>
#include <boost/aura/feed.hpp>

#include <memory>

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
                : data_()
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

        /// Create multi-dimensional array from initializer list.
        device_array(
                const std::initializer_list<
                        typename BoundsType::value_type
                >& dimensions,
                device& d)
                : bounds_(dimensions)
        {
                allocate(product(bounds_), d);
        }

        /// move constructor, move device_array here, invalidate other
        // @param da device_array to move here
        device_array(device_array&& da)
                : bounds_(std::move(da.bounds_))
                , data_(std::move(da.data_))
        {
                da.bounds_.clear();
        }

        /// destroy object
        ~device_array() {}

        /// Resize vector (optionally disallow shrinking).
        void resize(std::size_t size, device& d, bool shrink=true)
        {
                resize_impl(size, d, shrink);
                bounds_ = BoundsType({size});
        }

        void resize(const BoundsType& b, device& d, bool shrink=true)
        {
                resize_impl(product(b), d, shrink);
                bounds_ = b;
        }

        void resize(
                const std::initializer_list<
                        typename BoundsType::value_type
                >& dimensions,
                device& d,
                bool shrink=true
        )
        {
                boost::aura::bounds b(dimensions);
                resize_impl(product(b), d, shrink);
                bounds_ = b;
        }

        /// Zero vector (fill with \0 bytes).
        void zero(feed& f)
        {
                device_memset(*(data_.get()), 0, product(bounds_)*sizeof(T), f);
        }

        /// move assignment, move device_array here, invalidate other
        /// @param da device_array to move here
        device_array& operator=(device_array&& da)
        {
                bounds_ = da.bounds_;
                data_ = std::move(da.data_);
                da.bounds_.clear();
                return *this;
        }

        /// Access bounds and size
        BoundsType bounds() const { return bounds_; }

        std::size_t size() const { return product(bounds_); }

        /// Begin
        iterator begin() { return *(data_.get()); }
        const_iterator begin() const { return *(data_.get()); }

        /// End
        iterator end() { return *(data_.get()) + product(bounds_); }
        const_iterator end() const { return *(data_.get()) + product(bounds_); }

        /// Access base pointers.
        typename iterator::base_type get_base_ptr()
        {
                return data_.get()->get_base_ptr();
        }

        typename const_iterator::const_base_type get_base_ptr() const
        {
                return data_.get()->get_base_ptr();
        }

        /// Access host pointers.
        T* get_host_ptr() { return data_.get()->get_host_ptr(); }

        const T* get_host_ptr() const { return data_.get()->get_host_ptr(); }

        /// Indicate if memory hold by array is shared with host or not.
        bool is_shared_memory() const
        {
                return data_.get()->is_shared_memory();
        }

private:
        /// Implementation of resize method
        void resize_impl(std::size_t size, device& d, bool shrink=true)
        {
                if (!initialized_)
                {
                        allocate(size, d);
                }
                else
                {
                        auto current_size = product(bounds_);
                        if (current_size > size && shrink)
                        {
                                allocate(size, d);
                        }
                        else if (current_size < size)
                        {
                                allocate(size, d);
                        }
                }
        }

        /// Deleter type
        typedef detail::device_array_deleter<T> deleter_t;

        /// Data type
        typedef std::unique_ptr<device_ptr<T>, deleter_t> data_t;

        /// Allocation helper function
        void allocate(std::size_t size, device& d)
        {
                data_ = data_t(new device_ptr<T>(device_malloc<T>(size, d)),
                        deleter_t());
                initialized_ = true;
        }

        /// Initialized flag
        bool initialized_ { false };

        /// Stores the bounds
        BoundsType bounds_;

        /// Holds data
        data_t data_;
};

} // namespace aura
} // namespace boost
