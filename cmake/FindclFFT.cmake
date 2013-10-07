# - Try to find clFFT 
# Once done, this will define
#
#  clFFT_FOUND - system has clFFT 
#  clFFT_INCLUDE_DIRS - the clFFT include directories
#  clFFT_LIBRARIES - link these to use clFFT 

INCLUDE(LibFindMacros)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(clFFT_PKGCONF clFFT)

# Include dir
FIND_PATH(clFFT_INCLUDE_DIR
  NAMES clFFT.h clAmdFft.h
  PATHS ${clFFT_PKGCONF_INCLUDE_DIRS}
)

# Finally the library itself
FIND_LIBRARY(clFFT_LIBRARY
  NAMES clFFT 
  PATHS ${clFFT_PKGCONF_LIBRARY_DIRS}
)

# Set the include dir variables and the libraries and let libfind_process 
# do the rest.
SET(clFFT_PROCESS_INCLUDES clFFT_INCLUDE_DIR clFFT_INCLUDE_DIRS)
SET(clFFT_PROCESS_LIBS clFFT_LIBRARY clFFT_LIBRARIES)

libfind_process(clFFT)

