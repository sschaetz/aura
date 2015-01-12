# find clFFT libraries

FIND_PACKAGE(PackageHandleStandardArgs)

# Include dir
FIND_PATH(CLFFT_INCLUDE_DIRS
  clFFT.h  
  PATHS "/usr/local/include/clFFT" "/usr/local/clFFT/include"
)

# Finally the library itself
FIND_LIBRARY(CLFFT_LIBRARIES
  NAMES clFFT
  PATHS "/usr/local/clFFT/lib64"
)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(clFFT DEFAULT_MSG 
  CLFFT_LIBRARIES CLFFT_INCLUDE_DIRS
)
MARK_AS_ADVANCED(CLFFT_INCLUDE_DIRS)

