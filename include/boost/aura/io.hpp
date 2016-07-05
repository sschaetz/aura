#pragma once

#include <boost/aura/check.hpp>

#include <boost/test/unit_test.hpp>

#include <fstream>
#include <string>

namespace boost
{
namespace aura
{

/// Wrapper type to indicate a string should be interpreted as a path.
struct path
{
        path(const std::string& p)
                : str(p)
        {
        }

        std::string str;
};

/// Read contens of a file into a string and return it.
inline std::string read_all(path p)
{
        // Read contents of file.
        std::ifstream in(p.str, std::ios::in);
        AURA_CHECK_ERROR(in);
        in.seekg(0, std::ios::end);
        std::string str;
        str.resize(in.tellg());
        in.seekg(0, std::ios::beg);
        in.read(&str[0], str.size());
        in.close();
        return str;
}

} // namespace aura
} // namespace boost
