#ifndef AURA_DETAIL_SVEC_HPP
#define AURA_DETAIL_SVEC_HPP

#include <assert.h>
#include <array>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/type_traits/has_multiplies_assign.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/less_equal.hpp>
#include <aura/config.hpp>

namespace aura 
{

/**
* @brief svec small vector class
*/
template <typename T, std::size_t max_size_>
class svec 
{

// make sure we have constructors for the specified size
BOOST_MPL_ASSERT((
  boost::mpl::less_equal< 
    boost::mpl::int_<max_size_>, 
    boost::mpl::int_<AURA_SVEC_MAX_SIZE> >
));

public:

  /**
   * @brief empty small vector 
   */
  inline svec() : size_(0) { }

  // FIXME ctor should throw or better not compile if number of args > max_size

  /**
   * @brief construct dim with T
   *
   * There are more constructors that take more Ts, with the maximum
   * being the the template argument max_size_ 
   */
  inline explicit svec(const T & i0) : size_(1) { dim_[0] = i0; }

  #define AURA_SVEC_ARGS(z, n, _) , const T & id ## n
  #define AURA_SVEC_ASSIGN(z, n, _) dim_[n] = id ## n;
  
  #define AURA_SVEC_CTOR(z, n, _) \
    inline explicit svec(const T & id0 \
    BOOST_PP_REPEAT_FROM_TO(1, n, AURA_SVEC_ARGS, _) ) : \
      size_(n) { \
      assert(n <= max_size_); \
      BOOST_PP_REPEAT_FROM_TO(0, n, AURA_SVEC_ASSIGN, _) \
    } \
    /**/

  // constructors with 0...AURA_SVEC_MAX_SIZE
  BOOST_PP_REPEAT_FROM_TO(2,  
    BOOST_PP_INC(AURA_SVEC_MAX_SIZE), AURA_SVEC_CTOR, _)

  #undef AURA_SVEC_CTOR
  #undef AURA_SVEC_ASSIGN
  #undef AURA_SVEC_ARGS

  /// operator []
  T & operator [](const int & offset) { 
    return dim_[offset]; 
  }

  /// operator []
  const T & operator [](const int & offset) const { 
    return dim_[offset]; 
  }

  /**
   * @brief add element to vector and increment size
   */
  void push_back(const T & e) {
    assert(size_+1 <= max_size_);
    dim_[size_] = e;
    size_++;
  }

  /**
   * @brief return the size of the small vector 
   */
  inline const std::size_t & size() const { 
    return size_; 
  }

  /**
   * @brief return the maximum size of the small vector
   */
  inline const std::size_t & max_size() const { 
    return max_size_; 
  }

  /**
   * @brief print contents of dimension object
   */
  inline void debug__()
  {
    for(int i=0; i<size_; i++)
    {
      printf("%d: %d\n", i, dim_[i]);
    }
  }

private:
  /// number of Ts in dimension object 
  std::size_t size_;

  /// array containing dimensions 
  std::array<T, max_size_> dim_;
};


template <typename T, std::size_t max_size_>
T product_impl(const svec<T, max_size_> & v, const boost::true_type &) {
  T r = v[0];
  for(std::size_t i=1; i<v.size(); i++) {
    r *= v[i];
  }
  return r;
}

template <typename T, std::size_t max_size_, bool b>
T product_impl(const svec<T, max_size_> & v, const boost::integral_constant<bool, b>&) {
  // FIXME
  assert(false);
}

// calculate the product of all elements of svec (if *= operator exists for T)
template <typename T, std::size_t max_size_>
T product(const svec<T, max_size_> & v) {
  assert(0 < v.size());
  return product_impl(v, boost::has_multiplies_assign<T, T, T>());
}

} // namespace aura

#endif // AURA_DETAIL_SVEC_HPP

