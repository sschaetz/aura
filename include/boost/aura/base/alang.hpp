#pragma once

namespace boost
{
namespace aura
{

struct shared_alang_header
{
        static const std::string& get()
        {
                static std::string v = R"(
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
                        )";
                return v;
        }
};


} // namespace aura
} // namespace boost
