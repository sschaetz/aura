#pragma once

#include <boost/regex.hpp>
#include <iostream>
#include <iomanip>

namespace boost
{
namespace aura
{
namespace detail
{

template <typename T>
inline std::string value_to_string(const T &value)
{
        std::ostringstream out;
        out << std::setprecision(std::numeric_limits<T>::digits10 + 1) << value;
        return out.str();
}

template <>
inline std::string value_to_string(const float &value)
{
        std::ostringstream out;
        out << std::fixed <<
                std::setprecision(std::numeric_limits<float>::digits10 + 1) <<
                value << 'f';
        return out.str();
}

template <typename T>
inline std::string value_to_string(const std::vector<T> &values)
{
        std::string s("{");
        s += value_to_string(values[0]);

        for (auto it = values.begin() + 1; it != values.end(); it++)
        {
                s += ",";
                s += value_to_string(*it);
        }
        return s += "}";
}

inline std::string value_to_string(const char *value)
{
        return std::string(value);
}

inline std::string value_to_string(const std::string &value)
{
        return value;
}

} // namespace detail


struct preprocessor
{
        inline preprocessor() : re_("(<<<([A-z0-9_])+>>>)")
        {
        }

        /// Add define to preprocessor.
        template <typename T>
        void add_define(const std::string &name, const T &value)
        {
                auto t = name;
                t.insert(0, "<<<");
                t.append(">>>");
                defines_[t] = detail::value_to_string(value);
        }

        /// Take in string and output preprocessed string.
        inline std::string operator()(const std::string &s)
        {
                return boost::regex_replace(s, re_,
                        [this](const boost::smatch &m)
                        {
                                return defines_[m[0]];
                        });
        }

private:
        /// Regexp that is replaced.
        const boost::regex re_;

        /// Map of defines that maps keys to replacement.
        std::map<std::string, std::string> defines_;
};

} // namespace aura
} // namespace boost

