/* 
 * Authored by Dr. Tilman Sumpf
 */

#include <boost/aura/backend.hpp>

	AURA_KERNEL void memset_zero_float(AURA_GLOBAL float* dst_ptr)
{
		dst_ptr[get_mesh_id()] = 0.; // the most simple kernel I wrote yet
}