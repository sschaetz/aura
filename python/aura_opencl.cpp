#include <boost/python.hpp>

#include <boost/aura/environment.hpp>

using namespace boost::python;

const char* initialize()
{
        boost::aura::initialize();
        return "yo!";
}

const char* finalize()
{
        boost::aura::finalize();
        return "yo!";
}

BOOST_PYTHON_MODULE(aura_opencl)
{
        using namespace boost::python;
        def("initialize", initialize);
        def("finalize", finalize);
}
