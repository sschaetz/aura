#include <boost/aura/bounds.hpp>
#include <boost/aura/device.hpp>
#include <boost/aura/environment.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_PLUGIN(aura)
{
        using namespace boost::aura;
        py::module m("aura", "aura python wrapper");

        m.def("initialize", &boost::aura::initialize,
                "Function that initializes aura.");
        m.def("finalize", &boost::aura::finalize,
                "Function that finalizes aura.");

        py::class_<device>(m, "device")
                .def(py::init<std::size_t>())
                .def("get_ordinal", &device::get_ordinal)
                .def("activate", &device::activate)
                .def("deactivate", &device::deactivate);

        py::class_<bounds>(m, "bounds")
                .def(py::init<std::vector<int>>())
                .def("size", &bounds::size)
                .def("capacity", &bounds::capacity)
                .def("clear", &bounds::clear)
                .def("debug__", &bounds::debug__);

        return m.ptr();
}
