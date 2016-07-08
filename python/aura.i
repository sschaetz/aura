%module aura
%{
#define SWIG_FILE_WITH_INIT

namespace boost
{
namespace aura
{

#include "../include/boost/aura/base/opencl/device.hpp"
#include "../include/boost/aura/base/opencl/environment.hpp"


%}


%include "../include/boost/aura/base/opencl/device.hpp"
%include "../include/boost/aura/base/opencl/environment.hpp"



