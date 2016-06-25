#pragma once

#include <sstream>

#define AURA_OPENCL_SAFE_CALL(call)                                          \
        {                                                                    \
                int err = call;                                              \
                if (err != CL_SUCCESS)                                       \
                {                                                            \
                        std::ostringstream os;                               \
                        os << "OPENCL error " << err << " file " << __FILE__ \
                           << " line " << __LINE__;                          \
                        throw os.str();                                      \
                }                                                            \
        }                                                                    \
/**/


#define AURA_OPENCL_CHECK_ERROR(err)                                         \
        {                                                                    \
                if (err != CL_SUCCESS)                                       \
                {                                                            \
                        std::ostringstream os;                               \
                        os << "OPENCL error " << err << " file " << __FILE__ \
                           << " line " << __LINE__;                          \
                        throw os.str();                                      \
                }                                                            \
        }                                                                    \
/**/
