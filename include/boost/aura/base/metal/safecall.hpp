#pragma once

#import <Foundation/Foundation.h>

#define AURA_METAL_CHECK_ERROR(obj) { \
        if (!obj) \
        { \
                [NSException raise:@"METAL error (object NULL)" \
                format:@"%s:%d", __PRETTY_FUNCTION__, __LINE__]; \
        } \
} \
/**/

