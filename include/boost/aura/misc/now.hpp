#ifndef AURA_MISC_NOW_HPP
#define AURA_MISC_NOW_HPP

#include <chrono>

namespace boost
{
namespace aura 
{

inline double now() 
{
	return std::chrono::duration<double, std::micro>(
			std::chrono::high_resolution_clock::now().
			time_since_epoch()).count();
}

} // namespace aura
} // boost
 
#endif // AURA_MISC_NOW_HPP

