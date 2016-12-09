#pragma once

#include <sstream>

/// Check if a call returned error and throw exception if it did.
#define AURA_CHECK_INITIALIZED(flag)                                        \
        {                                                                   \
                if (!flag)                                                  \
                {                                                           \
                        std::ostringstream os;                              \
                        os << "Attempt to utilize uninitialized object "    \
                           << " file " << __FILE__ << " line " << __LINE__; \
                        throw os.str();                                     \
                }                                                           \
        }                                                                   \
/**/
