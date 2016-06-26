#pragma once

namespace boost
{
namespace aura
{

/// Indicates how memory is accessed in application.
enum class memory_access_tag
{
        ro,
        wo,
        rw
};

/// Indicates how memory is used in application.
enum class memory_usage_tag
{
        seldom,
        frequent
};

} // namespace aura
} // boost
