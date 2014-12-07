#ifndef AURA_COMPLEX_HPP
#define AURA_COMPLEX_HPP

#include <boost/aura/backend.hpp>


#define T float
#include "complex_impl.ipp"
#undef T
#define T double
#include "complex_impl.ipp"
#undef T

#endif // AURA_COMPLEX_HPP

