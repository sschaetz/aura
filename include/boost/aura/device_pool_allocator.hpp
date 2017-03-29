#pragma once

#include <boost/core/ignore_unused.hpp>

#include <boost/aura/device.hpp>
#include <boost/aura/device_ptr.hpp>

#include <unordered_map>
#include <vector>

namespace boost
{
namespace aura
{

/// General allocator for device memory.
template <class T>
struct device_pool_allocator
{
        using value_type = T;
        using pointer = device_ptr<T>;
        using const_pointer = const device_ptr<T>;

        /// Construct empty allocator.
        device_pool_allocator()
        {}

        /// Construct allocator.
        /// @param d Device.
        /// @param max_elements Maximum number of elements this pool can hold.
        device_pool_allocator(device& d,
                const std::size_t& max_elements = 10 * 1024 * 1024 / sizeof(T))
                : device_(&d)
                , max_elements_(max_elements)
        {}

        /// Copy construct allocator.
        template <class U>
        device_pool_allocator(const device_pool_allocator<U>& other)
                : device_(other.device_)
                , max_elements_(other.max_elements_)
        {}

        /// Move construct allocator.
        template <class U>
        device_pool_allocator(device_pool_allocator<U>&& other)
                : device_(other.device_)
                , max_elements_(other.max_elements_)
        {
                other.device_ = nullptr;
                other.max_elements_ = 0;
        }

        ~device_pool_allocator()
        {
                assert(in_use_memory_.size() == 0);
                // Allow no elements in the object and purge.
                max_elements_= 0;
                purge_();
        }

        /// Allocate memory.
        pointer allocate(std::size_t n)
        {
                assert(device_);
                assert(n < max_elements_);
                pointer ptr;

                // Check if we have available memory of that size.
                auto it = available_memory_.find(n);

                if (it != available_memory_.end())
                {
                        ptr = it->second.back();

                        // Add to in_use.
                        in_use_memory_[ptr] = it->first;

                        // Erase from free.
                        if (it->second.size() == 1)
                        {
                                // Remove this size entirely.
                                available_memory_.erase(it);
                        }
                        else
                        {
                                // Only remove this one pointer from size.
                                it->second.pop_back();
                        }
                }
                else
                {
                        ptr = device_malloc<T>(n, *device_);
                        num_elements_ += n;
                        in_use_memory_[ptr] = n;
                        if (num_elements_ >= max_elements_)
                        {
                                assert(!available_memory_.empty());
                                purge_();
                        }
                }
                return ptr;
        }

        /// Deallocate memory.
        void deallocate(pointer& p, std::size_t n)
        {
                assert(device_);
                auto it = in_use_memory_.find(p);

                // in_use_memory_ must contain this pointer.
                if (it != in_use_memory_.end())
                {

                        assert(it->second == n);
                        auto it2 = available_memory_.find(n);
                        if (it2 != available_memory_.end())
                        {
                                // Add to existing vector.
                                it2->second.push_back(it->first);
                        }
                        else
                        {
                                // Create new vector.
                                available_memory_[it->second] = {it->first};
                        }
                        in_use_memory_.erase(it);
                }
                else
                {
                        assert(false);
                }
        }

private:
        /// Methos used to purge the allocator if it has grown too big.
        void purge_()
        {
                while (num_elements_ > max_elements_)
                {
                        auto available_it = available_memory_.begin();
                        auto ptr_it = available_it->second.begin();
                        device_free(*ptr_it);
                        num_elements_ -= available_it->first;
                        available_it->second.erase(ptr_it);

                        // If this size is not available any longer, remove everythinig.
                        if (available_it->second.size() < 1)
                        {
                                available_memory_.erase(available_it);
                        }
                }

        }

        /// A single allocation is stored as it's size and a pointer.
        using available_storage_t = std::tuple<std::size_t, pointer>;

        /// Device we allocate memory from.
        device* device_;

        /// Flag that indicates if allocator is initialized or not.
        bool initialized_ { false };

        /// Maximum number of elements this allocator should hold.
        std::size_t max_elements_;

        /// Num elements held
        std::size_t num_elements_ = { 0 };

        /// List of free storage elements.
        std::map<std::size_t, std::vector<pointer>> available_memory_;

        /// List of in-use storage elements.
        std::unordered_map<pointer, std::size_t> in_use_memory_;
};

} // namespace aura
} // namespace boost
