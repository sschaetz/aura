#pragma once

#include <boost/aura/device.hpp>
#include <boost/aura/memory_tag.hpp>

#include <cstddef>
namespace boost
{
namespace aura
{
namespace detail
{

/// Base device pointer does not manage memory.
template <typename T, typename BaseType>
struct base_device_ptr
{
        /// Types required to make this an iterator.
        typedef std::random_access_iterator_tag iterator_category;
        typedef T value_type;
        typedef std::ptrdiff_t difference_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef std::shared_ptr<T> safe_pointer;
        typedef std::shared_ptr<T> const safe_const_pointer;
        typedef T& reference;

        /// Base handle type
        /// Can not be assigned to nullptr since
        /// it is defined as long long unsigned int.
        typedef BaseType base_type;

        /// Base handle type
        typedef const BaseType const_base_type;

        /// @brief Create pointer that points nowhere.
        base_device_ptr()
                : memory_()
                , offset_(0)
                , device_(nullptr)
                , tag_(memory_access_tag::rw)
                , is_shared_memory_(false)
        {
        }

        /// @brief Create pointer that points nowhere.
        base_device_ptr(std::nullptr_t)
                : memory_()
                , offset_(0)
                , device_(nullptr)
                , tag_(memory_access_tag::rw)
                , is_shared_memory_(false)
        {
        }

        /// @brief Create device pointer that points to memory.
        /// @param m Memory that identifies device memory
        /// @param d Device the memory is allocated on
        base_device_ptr(base_type& m, device& d,
                memory_access_tag tag = memory_access_tag::rw,
                bool is_shared_memory = false)
                : memory_(m)
                , offset_(0)
                , device_(&d)
                , tag_(tag)
                , is_shared_memory_(is_shared_memory)
        {
        }

        /// @brief Create device pointer that points to memory.
        /// @param m Memory that identifies device memory
        /// @param o Offset of memory object
        /// @param d Device the memory is allocated on
        base_device_ptr(const_base_type& m, const std::size_t& o, device& d,
                memory_access_tag tag = memory_access_tag::rw,
                bool is_shared_memory = false)
                : memory_(m)
                , offset_(o)
                , device_(&d)
                , tag_(tag)
                , is_shared_memory_(is_shared_memory)
        {
        }

        /// Invalidate pointer (sets everything to null).
        void reset()
        {
                device_ = nullptr;
                offset_ = 0;
                memory_.reset();
                tag_ = memory_access_tag::rw;
                is_shared_memory_ = false;
        }

        /// Returns a pointer to the device memory.
        base_type get_base_ptr() { return memory_; }
        const_base_type get_base_ptr() const { return memory_; }

        /// Returns a pointer to the device memory.
        std::size_t get_offset() const { return offset_; }

        /// Returns a pointer to the device memory.
        device& get_device() { return *device_; }
        const device& get_device() const { return *device_; }

        /// Returns the memory tag.
        memory_access_tag get_memory_access_tag() const { return tag_; }

        /// Indicate if memory held by pointer is shared with host or not.
        bool is_shared_memory() const
        {
                return is_shared_memory_;
        }

        /// Access host pointer (returns a nullptr if it is not available).
        pointer get_host_ptr()
        {
                return memory_.get_host_ptr();

        }

        const_pointer get_host_ptr() const
        {
                return memory_.get_host_ptr();

        }

        safe_pointer get_safe_host_ptr()
        {
                return memory_.get_safe_host_ptr();

        }

        safe_const_pointer get_safe_host_ptr() const
        {
                return memory_.get_safe_host_ptr();

        }

        /// Assign operator.
        base_device_ptr<T, BaseType>& operator=(
                base_device_ptr<T, BaseType> const& b)
        {
                memory_ = b.memory_;
                offset_ = b.offset_;
                device_ = b.device_;
                tag_ = b.tag_;
                is_shared_memory_ = b.is_shared_memory_;
                return *this;
        }

        /// Assign nullptr operator.
        base_device_ptr<T, BaseType>& operator=(std::nullptr_t)
        {
                reset();
                return *this;
        }

        /// Addition operator.
        base_device_ptr<T, BaseType> operator+(const std::size_t& b) const
        {
                return base_device_ptr<T, BaseType>(
                        memory_, offset_ + b, *device_, tag_,
                        is_shared_memory_);
        }

        std::ptrdiff_t operator+(const base_device_ptr<T, BaseType>& b) const
        {
                return offset_ + b.offset_;
        }

        /// Addition assignment operator
        base_device_ptr<T, BaseType>& operator+=(const std::size_t& b)
        {
                offset_ += b;
                return *this;
        }

        /// Prefix addition operator
        base_device_ptr<T, BaseType>& operator++()
        {
                ++offset_;
                return *this;
        }

        /// postfix addition operator
        base_device_ptr<T, BaseType> operator++(int)
        {
                return base_device_ptr<T, BaseType>(
                        memory_, offset_ + 1, *device_, tag_,
                        is_shared_memory_);
        }

        /// subtraction operator
        base_device_ptr<T, BaseType> operator-(const std::size_t& b) const
        {
                return *this + (-b);
        }

        std::ptrdiff_t operator-(const base_device_ptr<T, BaseType>& b) const
        {
                return offset_ - b.offset_;
        }

        /// subtraction assignment operator
        base_device_ptr<T, BaseType>& operator-=(const std::size_t& b)
        {
                return *this += (-b);
        }

        /// prefix subtraction operator
        base_device_ptr<T, BaseType>& operator--()
        {
                --offset_;
                return *this;
        }

        /// postfix subtraction operator
        base_device_ptr<T, BaseType> operator--(int)
        {
                return base_device_ptr<T, BaseType>(
                        memory_, offset_ - 1, *device_, tag_,
                        is_shared_memory_);
        }

        /// equal to operator
        bool operator==(const base_device_ptr<T, BaseType>& b) const
        {
                if (nullptr == device_ || nullptr == b.device_)
                {
                        return (nullptr == device_ && nullptr == b.device_ &&
                                offset_ == b.offset_ && memory_ == b.memory_ &&
                                tag_ == b.tag_ &&
                                is_shared_memory_ == b.is_shared_memory_);
                }
                else
                {
                        return (device_->get_ordinal() ==
                                        b.device_->get_ordinal() &&
                                offset_ == b.offset_ && memory_ == b.memory_ &&
                                tag_ == b.tag_ &&
                                is_shared_memory_ == b.is_shared_memory_);
                }
        }

        /// Less than operator.
        bool operator <(const base_device_ptr<T, BaseType>& b) const
        {
                if (nullptr == device_ && nullptr != b.device_)
                {
                        // This is no initialized, is always smaller.
                        return true;
                }
                else if (nullptr == b.device_)
                {
                        // The other is not initialized.
                        return false;
                }
                else if (memory_ < b.memory_)
                {
                        // Compare pointers.
                        return true;
                }
                else if (memory_ == b.memory_)
                {
                        // Same ptr, compare offset.
                        return offset_ < b.offset_;
                }
                else
                {
                        return false;
                }
        }

        bool operator==(std::nullptr_t) const
        {
                return (nullptr == device_ && 0 == offset_ && 0 == memory_);
        }

        /// not equal to operator
        bool operator!=(const base_device_ptr<T, BaseType>& b) const
        {
                return !(*this == b);
        }

        bool operator!=(std::nullptr_t) const { return !(*this == nullptr); }

private:
        /// actual pointer that identifies device memory
        base_type memory_;

        /// the offset (OpenCL does not support arithmetic on the pointer)
        std::size_t offset_;

        /// reference to device the pointer points to
        device* device_;

        /// read+write readonly writeonly?
        memory_access_tag tag_;

        /// is memory shared beween GPU and CPU?
        bool is_shared_memory_;
};


/// equal to operator (reverse order)
template <typename T, typename BaseType>
bool operator==(std::nullptr_t, const base_device_ptr<T, BaseType>& ptr)
{
        return (ptr == nullptr);
}

/// not equal to operator (reverse order)
template <typename T, typename BaseType>
bool operator!=(std::nullptr_t, const base_device_ptr<T, BaseType>& ptr)
{
        return (ptr != nullptr);
}

} // detail
} // aura
} // boost
