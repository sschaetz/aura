#ifndef AURA_DETAIL_DIM_BASE_HPP
#define AURA_DETAIL_DIM_BASE_HPP

#include <boost/array.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/less_equal.hpp>
#include <aura/config.hpp>

namespace aura 
{

/**
* @brief dim_base dimension base class
*/
template <typename T, std::size_t max_size>
class dim_base 
{

// make sure we have constructors for the specified size
BOOST_MPL_ASSERT((
  boost::mpl::less_equal< 
    boost::mpl::int_<max_size>, 
    boost::mpl::int_<AURA_DIM_BASE_MAX_SIZE> >
));

public:

  /**
   * @brief empty dim constructor
   */
  inline dim_base() : size_(0) { }

  /**
   * @brief construct dim with T
   *
   * There are more constructors that take more Ts, with the maximum
   * being the maximum number of devices supported by the library
   */
  inline dim_base(T i0) : size_(1) { dim_[0] = i0; }


  #define AURA_DIM_BASE_ARGS(z, n, _) , T id ## n
  #define AURA_DIM_BASE_ASSIGN(z, n, _) dim_[n] = id ## n;
  
  #define AURA_DIM_BASE_CTOR(z, n, _) \
    inline dim_base(T id0 \
    BOOST_PP_REPEAT_FROM_TO(1, n, AURA_DIM_BASE_ARGS, _) ) : \
      size_(n) { \
      assert(n <= max_size); \
      BOOST_PP_REPEAT_FROM_TO(0, n, AURA_DIM_BASE_ASSIGN, _) \
    } \
    /**/

  // constructors with 0...AURA_DIM_BASE_MAX_SIZE
  BOOST_PP_REPEAT_FROM_TO(2,  
    BOOST_PP_INC(AURA_DIM_BASE_MAX_SIZE), AURA_DIM_BASE_CTOR, _)

  #undef AURA_DIM_BASE_CTOR
  #undef AURA_DIM_BASE_ASSIGN
  #undef AURA_DIM_BASE_ARGS


  /// operator []
  T & operator [](const int & offset) { return dim_[offset]; }

  /// operator []
  const T & operator [](const int & offset) const
  { return dim_[offset]; }


  /**
   * @brief return the size of the dimension object 
   */
  inline const std::size_t & size() const { return size_; }

  /**
   * @brief set the size of the dimension object 
   */
  inline void set_size_(const std::size_t size) { size_ = size; }

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
  boost::array<T, max_size> dim_;
};

} // namespace aura

#endif // AURA_DETAIL_DIM_BASE_HPP

