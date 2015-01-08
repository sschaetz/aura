#ifndef AURA_BACKEND_SHARED_CALC_MESH_BUNDLE_HPP
#define AURA_BACKEND_SHARED_CALC_MESH_BUNDLE_HPP

namespace boost
{
namespace aura {
namespace detail {


/**
 * @brief calculate combination of bundle and mesh, based on v
 *
 * recursively calculates the integer factorization and fills 
 * up an array of mesh and bundle (iterator i), rules are 
 * defined by a maximum size for mesh and bundle (iterator b)
 *
 * the mask m defines if and at what position, the algorithm
 * should restart calculating, taking the previous value
 * into account 
 *
 * FIXME this function and its use is not implemented in a 
 * general way regarding the mask, but it is ok for now
 *
 * @param v is the value that should be factorized
 * @param f is the current factor
 */
inline void calc_mesh_bundle(std::size_t v, std::size_t f, 
		std::array<std::size_t, 4>::iterator i,
		std::array<std::size_t, 4>::const_iterator b,
		std::array<bool, 4>::const_iterator m)
{
	if (f > v) {
		std::cout << *i << " " << v << " " << f << " " << *m << std::endl;
		if (*m) {
		}	
		std::cout << "returning" << std::endl;
		return;
	}
	if (0 == v % f) {
		if (*i*f > *b) {
			++i;
			++b;
			++m;
			// special handling for mask
			if (*m) {
				f *= *(i-1);
				assert(f < *b);
				*i *= f;
				f /= *(i-1);
				std::cout << "special handling for mask" << std::endl;
				calc_mesh_bundle(v/f, f, i, b, m);
				return;
			}
			// if next dimension can hold value
			// the size is invalid
			assert(f < *b);
		}
		// put new factor in
		*i *= f;
		calc_mesh_bundle(v/f, f, i, b, m);
	} else {
		f++;
		calc_mesh_bundle(v, f, i, b, m);
	}
}

} // namespace detail
} // namespace aura 
} // boost

#endif // AURA_BACKEND_SHARED_CALC_MESH_BUNDLE_HPP

