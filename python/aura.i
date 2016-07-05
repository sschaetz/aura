%module aura
%{
namespace boost
{
namespace aura
{
#define AURA_BASE_CUDA

#include "boost/aura/environment.hpp"
#include "boost/aura/feed.hpp"
#include "boost/aura/device.hpp"
}
}
%}
namespace boost
{
namespace aura
{
%include "boost/aura/environment.hpp"
%include "boost/aura/feed.hpp"
%include "boost/aura/device.hpp"
}
}
