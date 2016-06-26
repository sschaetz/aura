#pragma once

#include <boost/aura/base/cuda/device.hpp>
#include <boost/aura/memory_tag.hpp>

#include <cuda.h>

#include <cstddef>

namespace boost
{
namespace aura
{
namespace base_detail
{
namespace cuda
{

/// Device pointer does not manage memory.
template <typename T>
struct device_ptr
{

        /// Base handle type
        /// Can not be assigned to nullptr since
        /// it is defined as long long unsigned int.
        typedef CUdeviceptr base_type;

        /// Base handle type
        typedef const CUdeviceptr const_base_type;

public:
        /// @brief Create pointer that points nowhere.
        device_ptr()
                : memory_(0)
                , offset_(0)
                , device_(nullptr)
                , tag_(memory_access_tag::rw)
        {
        }

        /// @brief Create pointer that points nowhere.
        device_ptr(std::nullptr_t)
                : memory_(0)
                , offset_(0)
                , device_(nullptr)
                , tag_(memory_access_tag::rw)
        {
        }

        /// @brief Create device pointer that points to memory.
        ///
        /// @param m Memory that identifies device memory
        /// @param d Device the memory is allocated on
        device_ptr(base_type &m, device &d, memory_access_tag tag = memory_access_tag::rw)
                : memory_(m)
                , offset_(0)
                , device_(&d)
                , tag_(tag)
        {
        }

        /// @brief Create device pointer that points to memory.
        ///
        /// @param m Memory that identifies device memory
        /// @param o Offset of memory object
        /// @param d Device the memory is allocated on
        device_ptr(const_base_type &m, const std::size_t &o, device &d,
                memory_access_tag tag = memory_access_tag::rw)
                : memory_(m)
                , offset_(o)
                , device_(&d)
                , tag_(tag)
        {
        }

        /// Invalidate pointer (sets everything to null).
        void reset()
        {
                memory_ = 0;
                device_ = nullptr;
                offset_ = 0;
                tag_ = memory_access_tag::rw;
        }

        /// Returns a pointer to the device memory.
        base_type get_base_ptr()
        {
                return memory_;
        }
        const_base_type get_base_ptr() const
        {
                return memory_;
        }

        /// Returns a pointer to the device memory.
        std::size_t get_offset() const
        {
                return offset_;
        }

        /// Returns a pointer to the device memory.
        device &get_device()
        {
                return *device_;
        }
        const device &get_device() const
        {
                return *device_;
        }

        /// Returns the memory tag.
        memory_access_tag get_memory_access_tag() const
        {
                return tag_;
        }

        /// Assign operator.
        device_ptr<T> &operator=(device_ptr<T> const &b)
        {
                memory_ = b.memory_;
                offset_ = b.offset_;
                device_ = b.device_;
                tag_ = b.tag_;
                return *this;
        }

        /// Assign nullptr operator.
        device_ptr<T> &operator=(std::nullptr_t)
        {
                reset();
                return *this;
        }

        /// Addition operator.
        device_ptr<T> operator+(const std::size_t &b) const
        {
                return device_ptr<T>(memory_, offset_ + b, *device_, tag_);
        }

        /// Addition assignment operator
        device_ptr<T> &operator+=(const std::size_t &b)
        {
                offset_ += b;
                return *this;
        }

        /// Prefix addition operator
        device_ptr<T> &operator++()
        {
                ++offset_;
                return *this;
        }

        /// postfix addition operator
        device_ptr<T> operator++(int)
        {
                return device_ptr<T>(memory_, offset_ + 1, *device_);
        }

        /// subtraction operator
        device_ptr<T> operator-(const std::size_t &b) const
        {
                return *this + (-b);
        }

        /// subtraction assignment operator
        device_ptr<T> &operator-=(const std::size_t &b)
        {
                return *this += (-b);
        }

        /// prefix subtraction operator
        device_ptr<T> &operator--()
        {
                --offset_;
                return *this;
        }

        /// postfix subtraction operator
        device_ptr<T> operator--(int)
        {
                return device_ptr<T>(memory_, offset_ - 1, *device_, tag_);
        }

        /// equal to operator
        bool operator==(const device_ptr<T> &b) const
        {
                if (nullptr == device_ || nullptr == b.device_)
                {
                        return (nullptr == device_ && nullptr == b.device_ &&
                                offset_ == b.offset_ && memory_ == b.memory_ &&
                                tag_ == b.tag_);
                }
                else
                {
                        return (device_->get_ordinal() ==
                                        b.device_->get_ordinal() &&
                                offset_ == b.offset_ && memory_ == b.memory_ &&
                                tag_ == b.tag_);
                }
        }

        bool operator==(std::nullptr_t) const
        {
                return (nullptr == device_ && 0 == offset_ && 0 == memory_);
        }

        /// not equal to operator
        bool operator!=(const device_ptr<T> &b) const
        {
                return !(*this == b);
        }

        bool operator!=(std::nullptr_t) const
        {
                return !(*this == nullptr);
        }

private:
        /// actual pointer that identifies device memory
        base_type memory_;

        /// the offset (OpenCL does not support arithmetic on the pointer)
        std::size_t offset_;

        /// reference to device the pointer points to
        device *device_;

        /// read+write readonly writeonly?
        memory_access_tag tag_;
};

/// equal to operator (reverse order)
template <typename T> bool operator==(std::nullptr_t, const device_ptr<T> &ptr)
{
        return (ptr == nullptr);
}

/// not equal to operator (reverse order)
template <typename T> bool operator!=(std::nullptr_t, const device_ptr<T> &ptr)
{
        return (ptr != nullptr);
}


/// Allocate device memory.
template <typename T> device_ptr<T> device_malloc(std::size_t size, device &d)
{
        d.activate();
        typename device_ptr<T>::base_type m;
        AURA_CUDA_SAFE_CALL(cuMemAlloc(&m, size * sizeof(T)));
        d.deactivate();
        return device_ptr<T>(m, d);
}

/// Free device memory.
template <typename T> void device_free(device_ptr<T> &ptr)
{
        ptr.get_device().activate();
        AURA_CUDA_SAFE_CALL(cuMemFree(ptr.get_base_ptr()));
        ptr.get_device().deactivate();
        ptr.reset();
}

} // cuda
} // base_detail
} // aura
} // boost
