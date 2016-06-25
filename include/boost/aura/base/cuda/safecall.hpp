#pragma once

#include <sstream>

/// Check if a call returned error and throw exception if it did.
#define AURA_CUDA_SAFE_CALL(call)                                           \
        {                                                                   \
                CUresult err = call;                                        \
                if (err != CUDA_SUCCESS)                                    \
                {                                                           \
                        const char *errstr;                                 \
                        cuGetErrorName(err, &errstr);                       \
                        std::ostringstream os;                              \
                        os << "CUDA error " << err << " " << errstr         \
                           << " file " << __FILE__ << " line " << __LINE__; \
                        throw os.str();                                     \
                }                                                           \
        }                                                                   \
/**/


/// Check for error and throw exception if an error occured.
#define AURA_CUDA_CHECK_ERROR(err)                                          \
        {                                                                   \
                if (err != CUDA_SUCCESS)                                    \
                {                                                           \
                        const char *errstr;                                 \
                        cuGetErrorName(err, &errstr);                       \
                        std::ostringstream os;                              \
                        os << "CUDA error " << err << " " << errstr         \
                           << " file " << __FILE__ << " line " << __LINE__; \
                        throw os.str();                                     \
                }                                                           \
        }                                                                   \
/**/
