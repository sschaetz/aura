#ifndef AURA_DEVICE_LOCK_HPP
#define AURA_DEVICE_LOCK_HPP

#include <fstream>
#include <memory>
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>
#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

namespace boost
{
namespace aura 
{

// mwahahahahahahaha 
typedef boost::optional<
		std::pair<
			std::shared_ptr<
				boost::interprocess::scoped_lock<
					boost::interprocess::file_lock>
				>,
			std::shared_ptr<
				boost::interprocess::file_lock
				>
			> 
		> device_lock;

/**
 * I did this because apparently for the scoped lock thingy, the lock
 * itself must exist - if the lock itself goes out of scope it does not
 * seem to work. Now it lives on the heap together with the scoped lock 
 * and is shipped right with it. I hope the destruction of the pair
 * does not foil this evil plan. I made the scoped lock the first
 * member of the pair on purpose.
 */

/// generate a filepath+filename that can be locked 
inline boost::filesystem::path get_lockfile_name(std::size_t ordinal)
{
	boost::filesystem::path p = boost::filesystem::temp_directory_path();
	p /= std::string("AURA_device_lock_") + std::to_string(ordinal);
	return p;
}

/// create a device_lock
inline device_lock create_device_lock(std::size_t ordinal)
{
	// get the file-name
	auto fname = get_lockfile_name(ordinal);

	// create the file if it does not exist
	try {
		std::ofstream fhandle(fname.c_str());	
	} catch (...) {
		return boost::none;
	}

	// give file permissions to everybody
	try {
		boost::filesystem::permissions(fname, 
				boost::filesystem::all_all);
	} catch (...) {
		return boost::none;
	}

	// lock the file
	auto flock = std::make_shared<boost::interprocess::file_lock>(
			fname.c_str());
	try {
		if (!flock->try_lock()) {
			return boost::none;
		}
	} catch (...) {
		return boost::none;
	}
	
	// create the scoped_lock 
	auto sflock = std::make_shared<boost::interprocess::scoped_lock<
					boost::interprocess::file_lock>
				>(*flock);
	// and ship	
	return std::make_pair(sflock, flock);
}


} // namespace aura
} // namespace boost
#endif // AURA_DEVICE_LOCK_HPP

