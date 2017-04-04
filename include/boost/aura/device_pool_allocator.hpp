#pragma once

#include <boost/core/ignore_unused.hpp>

#include <boost/aura/device.hpp>
#include <boost/aura/device_ptr.hpp>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace boost
{
namespace aura
{

/// device_pool_allocator
/// There is not really a pool in this allocator.
/// This is more of a memoization: if a user
/// alloc(1024), free(1024) and alloc(1024)
/// the second alloc will just return the freed memory from before
/// (it is not really freed). The allocator memoizes up to a limit;
/// if the limit is hit it starts evicting/puring memory.
///
/// In real-time pipelines allocation sizes are typically very similar
/// (low to no variance) so this allocator should give a speed-up as
/// subsequent alloc calls are cached.
///
/// The allocator stores a multimapmap<size, vector<ptr>> for free memory
/// blocks size can be looked up and the pointer can be looked up immediately.
/// It can store multiple pointers for a specific size because there might be
/// identical. It stores a multimap<pointer, size_t> for in-use memory so
/// pointers can be looked up quickly when deallocation should occur and move
/// the pointer over to the free map.
/// @tparam T Type the allocator allocates.
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

        /// Move construct allocator.
        template <class U>
        device_pool_allocator(device_pool_allocator<U>&& other)
        {
                std::lock(mutex_, other.mutex_);
                std::lock_guard<std::mutex> guard0(
                        mutex_, std::adopt_lock
                );
                std::lock_guard<std::mutex> guard1(
                        other.mutex_, std::adopt_lock
                );

                device_ = other.device_;
                max_elements_ = other.max_elements;
                initialized_ = other.initialized_;
                num_elements_ = other.num_elements_;
                available_memory_ = other.available_memory_;
                in_use_memory_ = other.in_use_memory_;

                other.device_ = nullptr;
                other.initialized_ = false;
                other.num_elements = 0;
                other.available_memory_.clear();
                other.in_use_memory_.clear();
        }

        ~device_pool_allocator()
        {
                std::lock_guard<std::mutex> guard(mutex_);

                // Allow no elements in the object and purge.
                max_elements_= 0;
                purge_();
                purge_in_use_memory_();
        }

        /// Allocate memory.
        pointer allocate(std::size_t n)
        {
                std::lock_guard<std::mutex> guard(mutex_);
                assert(device_);
                assert(n < max_elements_);
                pointer ptr;

                // Check if we have available memory of that size.
                auto it = available_memory_.find(n);

                if (it != available_memory_.end())
                {
                        ptr = it->second;
                        // Add to in_use.
                        in_use_memory_[it->second] = it->first;
                        available_memory_.erase(it);
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
                std::lock_guard<std::mutex> guard(mutex_);
                assert(device_);
                auto it = in_use_memory_.find(p);

                // in_use_memory_ must contain this pointer.
                if (it != in_use_memory_.end())
                {
                        assert(it->second == n);
                        available_memory_.emplace(std::make_pair(n, it->first));
                        in_use_memory_.erase(it);
                }
                else
                {
                        assert(false);
                }
                p.reset();
        }

private:
        /// Method used to purge the allocator if it has grown too big.
        void purge_()
        {
                while (num_elements_ > max_elements_ &&
                        !available_memory_.empty())
                {
                        auto available_it = available_memory_.begin();
                        auto ptr = available_it->second;
                        num_elements_ -= available_it->first;
                        available_memory_.erase(available_it);
                        device_free(ptr);
                }
        }

        /// Method used to free all memory in case the allocator is destroyed.
        void purge_in_use_memory_()
        {
                for (auto memory : in_use_memory_)
                {
                        auto ptr = memory.first;
                        device_free(ptr);
                }
                in_use_memory_.clear();
        }

        /// Mutex used to allow multi-threaded access to class.
        std::mutex mutex_;

        /// Device we allocate memory from.
        device* device_;

        /// Flag that indicates if allocator is initialized or not.
        bool initialized_ { false };

        /// Maximum number of elements this allocator should hold.
        std::size_t max_elements_;

        /// Num elements held
        std::size_t num_elements_ = { 0 };

        /// List of free storage elements.
        std::unordered_multimap<std::size_t, pointer>
        available_memory_;

        /// List of in-use storage elements.
        std::unordered_map<pointer, std::size_t> in_use_memory_;
};

} // namespace aura
} // namespace boost
