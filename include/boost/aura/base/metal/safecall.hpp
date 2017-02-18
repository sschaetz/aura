#pragma once

#import <Foundation/Foundation.h>

#if ! __has_feature(objc_arc)
#error This file must be compiled with ARC. Either turn on ARC for the project or use -fobjc-arc flag
#endif

#define AURA_METAL_CHECK_ERROR(obj)                                       \
        {                                                                 \
                if (!obj)                                                 \
                {                                                         \
                        [NSException raise:@"METAL error (object NULL)"   \
                                    format:@"%s:%d", __PRETTY_FUNCTION__, \
                                    __LINE__];                            \
                }                                                         \
        }                                                                 \
/**/
