# find clBLAS libraries

FIND_PACKAGE(PackageHandleStandardArgs)

# Include dir
FIND_PATH(CLBLAS_INCLUDE_DIRS
	clBLAS.h  
	PATHS "/usr/local/include/clBLAS" "/usr/local/clBLAS/include"
)

# Finally the library itself
FIND_LIBRARY(CLBLAS_LIBRARIES
	NAMES clBLAS
	PATHS "/usr/local/clBLAS/lib64"
)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(clBLAS DEFAULT_MSG 
	CLBLAS_LIBRARIES CLBLAS_INCLUDE_DIRS
)
MARK_AS_ADVANCED(CLBLAS_INCLUDE_DIRS)

