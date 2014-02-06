#ifndef AURA_DETAIL_SVEC_HPP
#define AURA_DETAIL_SVEC_HPP

#include <iostream> 
#include <assert.h>
#include <array>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/type_traits/has_multiplies_assign.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/less_equal.hpp>
#include <aura/config.hpp>

namespace aura 
{

/**
* svec small vector class, probably can be replaced by std::array with
* initializer lits, but for now this works 
*/
template <typename T, std::size_t max_size_ = AURA_SVEC_MAX_SIZE>
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

	// FIXME ctor should throw or better not compile if 
	// number of args > max_size

	/**
	 * @brief construct dim with T
	 *
	 * There are more constructors that take more Ts, with the maximum
	 * being the the template argument max_size_ 
	 */
	inline explicit svec(const T & i0) : size_(1) { data_[0] = i0; }

	#define AURA_SVEC_ARGS(z, n, _) , const T & id ## n
	#define AURA_SVEC_ASSIGN(z, n, _) data_[n] = id ## n;

	#define AURA_SVEC_CTOR(z, n, _) \
		inline explicit svec(const T & id0 \
			BOOST_PP_REPEAT_FROM_TO(1, n, AURA_SVEC_ARGS, _) ) : \
			size_(n) \
		{ \
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

	/// copy ctor and assignment can be provided by compiler

	/// operator []
	T & operator [](const int & offset) 
	{ 
		return data_[offset]; 
	}

	/// operator []
	const T & operator [](const int & offset) const 
	{ 
		return data_[offset]; 
	}

	/**
	 * @brief add element to vector and increment size
	 */
	void push_back(const T & e) 
	{
		assert(size_+1 <= max_size_);
		data_[size_] = e;
		size_++;
	}

	/**
	 * @brief return the size of the small vector 
	 */
	inline const std::size_t & size() const 
	{ 
		return size_; 
	}

	/**
	 * @brief return the maximum size of the small vector
	 */
	inline const std::size_t & max_size() const 
	{ 
		return max_size_; 
	}

	/**
	 * @brief clear the small vector
	 */
	inline void clear() {
		size_ = 0;
	}

	/**
	 * @brief print contents of vector
	 */
	inline void debug__()
	{
		for(std::size_t i=0; i<size_; i++)
		{
			printf("%lu: %lu\n", i, data_[i]);
		}
	}

	/**
	 * @brief get data
	 */
	const std::array<T, max_size_> & array() const {
		return data_;
	}

private:
	/// number of Ts in object 
	std::size_t size_;

	/// array containing data 
	std::array<T, max_size_> data_;
};


template <typename T, std::size_t size_>
T product_impl(const std::array<T, size_> & v, std::size_t size, 
		const boost::true_type &) 
{
	T r = v[0];
	for(std::size_t i=1; i<size; i++) {
		r *= v[i];
	}
	return r;
}

template <typename T, std::size_t size_, bool b>
T product_impl(const std::array<T, size_> & v, std::size_t size,
	const boost::integral_constant<bool, b> &) 
{
	// FIXME
	assert(false);
}

/// calculate the product of all elements of svec 
/// (if *= operator exists for T)
template <typename T, std::size_t max_size_>
T product(const svec<T, max_size_> & v) 
{
	assert(0 < v.size());
	return product_impl(v.array(), v.size(), 
			boost::has_multiplies_assign<T, T, T>());
}

/// calculate the product of all elements of std::array 
/// (if *= operator exists for T)
template <typename T, std::size_t size_>
T product(const std::array<T, size_> & v) 
{
	assert(0 < v.size());
	return product_impl(v, v.size(), 
			boost::has_multiplies_assign<T, T, T>());
}

/// write content of svec to sized buffer
template <std::size_t max_size_>
void svec_snprintf(char * c, std::size_t s, 
	const svec<std::size_t, max_size_> & v) 
{
	assert(0 < v.size());
	std::size_t l = 0;
	for(std::size_t i=0; i<v.size(); i++) {
		l = snprintf(c, s, "%lu,", v[i]);
		c+=l;
		s-=l;
	}
	c-=1;
	*c='\0';
}

/// output content of svec to ostream
template <typename T, std::size_t max_size_>
std::ostream& operator << (std::ostream & o, svec<T, max_size_> & a) 
{
	std::size_t i;
	for(i=0; i<a.size()-1; i++) {
		o << a[i] << " ";
	}
	o << a[i];
	return o;
}

} // namespace aura

#endif // AURA_DETAIL_SVEC_HPP

