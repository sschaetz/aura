#include <boost/aura/environment.hpp>

#include <pybind11/pybind11.h>

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

namespace py = pybind11;

PYBIND11_PLUGIN(pyaura)
{
    py::module m("pyaura", "aura python wrapper");

    m.def("initialize", &initialize, "Function that initializes aura.");
    m.def("finalize", &finalize, "Function that finalizes aura.");

    return m.ptr();
}
