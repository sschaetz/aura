#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

namespace boost
{
namespace aura
{
namespace detail
{

/// Tracks memory allocations.
/// Can be deactivated completely
/// by defining BOOST_AURA_NO_TRACK_ALLOCATIONS.
class allocation_tracker
{
public:
        /// Default constructor.
        allocation_tracker(bool active = false)
                : active_(active)
        {}

        /// Activate allocation tracking.
        void activate()
        {
                active_ = true;
        }

        /// Deactivate allocation tracking.
        void deactivate()
        {
                active_ = false;
        }

        /// Add information about an allocation.
        template <typename T>
        inline void add(const T * ptr, const std::size_t size)
        {
                add(reinterpret_cast<const uintptr_t>(ptr), size);
        }

        inline void add(const uintptr_t ptr, const std::size_t size)
        {
#ifdef BOOST_AURA_NO_TRACK_ALLOCATIONS
                boost::ignore_unused(ptr);
                boost::ignore_unused(size);
#else
                if (!active_)
                {
                        return;
                }
                std::lock_guard<std::mutex> guard(mutex_);
                assert(active_allocations_.count(ptr) == 0);
                active_allocations_[ptr] = size;
#endif
        }

        /// Add information about a de-allocation.
        template <typename T>
        inline void remove(const T* ptr)
        {
                remove(reinterpret_cast<const uintptr_t>(ptr));
        }

        inline void remove(const uintptr_t ptr)
        {
#ifdef BOOST_AURA_NO_TRACK_ALLOCATIONS
                boost::ignore_unused(ptr);
#else
                if (!active_)
                {
                        return;
                }
                std::lock_guard<std::mutex> guard(mutex_);
                // Find, move to old erase.
                auto r = active_allocations_.find(ptr);
                assert(r != active_allocations_.end());
                old_allocations_.push_back(r->second);
                active_allocations_.erase(r);
#endif
        }

        /// Return number of active allocations.
        inline std::size_t count_active() const
        {
#ifdef BOOST_AURA_NO_TRACK_ALLOCATIONS
                return 0;
#else
                std::lock_guard<std::mutex> guard(mutex_);
                return active_allocations_.size();
#endif
        }

        /// Return number of old allocations.
        inline std::size_t count_old() const
        {
#ifdef BOOST_AURA_NO_TRACK_ALLOCATIONS
                return 0;
#else
                std::lock_guard<std::mutex> guard(mutex_);
                return old_allocations_.size();
#endif
        }

private:
#ifndef BOOST_AURA_NO_TRACK_ALLOCATIONS
        /// Flag that indicates if allocation tracker is active or not.
        std::atomic<bool> active_;

        /// Tracks active allocations and their size.
        std::map<uintptr_t, std::size_t> active_allocations_;

        /// Tracks old allocations.
        std::vector<std::size_t> old_allocations_;

        /// Mutex that allows multi-threaded access.
        mutable std::mutex mutex_;
#endif

};

} // detail
} // aura
} // boost
