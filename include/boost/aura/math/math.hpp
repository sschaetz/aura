#ifndef BOOST_AURA_MATH_MATH_HPP
#define BOOST_AURA_MATH_MATH_HPP

// basic
#include <boost/aura/math/basic/add.hpp>
#include <boost/aura/math/basic/conj.hpp>
#include <boost/aura/math/basic/div.hpp>
#include <boost/aura/math/basic/fma.hpp>
#include <boost/aura/math/basic/fma.hpp>
#include <boost/aura/math/basic/mul.hpp>
#include <boost/aura/math/basic/sqrt.hpp>
#include <boost/aura/math/basic/sub.hpp>
#include <boost/aura/math/basic/exp.hpp>

// blas
#include <boost/aura/math/blas/axpy.hpp>
#include <boost/aura/math/blas/dot.hpp>
#include <boost/aura/math/blas/norm2.hpp>
#include <boost/aura/math/blas/sum.hpp>

// special
#include <boost/aura/math/special/reduced_sum.hpp>
#include <boost/aura/math/special/ndmul.hpp>
//#include <boost/aura/math/special/nlinv_operators_1.hpp>
#include <boost/aura/math/special/nlinv_operators.hpp>


// basic linear algebra
#include <boost/aura/math/complex.hpp>
#include <boost/aura/math/memset_zero.hpp>
#include <boost/aura/math/memset_ones.hpp>
#include <boost/aura/math/split_interleaved.hpp>
#include <boost/aura/math/support_functions.hpp>

// numerical optimization
#include <boost/aura/math/conjgrad.hpp>
#include <boost/aura/math/gauss_newton.hpp>

#endif // BOOST_AURA_MATH_MATH_HPP

