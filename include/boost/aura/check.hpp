#pragma once

#include <stdio.h>

#define AURA_CHECK_ERROR(expr)                                               \
        {                                                                    \
                if (!expr)                                                   \
                {                                                            \
                        printf("AURA error at %s:%d\n", __FILE__, __LINE__); \
                }                                                            \
        }                                                                    \
/**/
