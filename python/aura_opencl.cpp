#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>

#include <boost/aura/environment.hpp>

using namespace boost::python;

void initialize()
{
        boost::aura::initialize();
        return;
}

void finalize()
{
        boost::aura::finalize();
        return;
}

BOOST_PYTHON_MODULE(aura_opencl)
{
        using namespace boost::python;
        def("initialize", raw_function(initialize));
        def("finalize", raw_function(finalize));
}
