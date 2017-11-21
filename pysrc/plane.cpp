#include "plane.h"
#include <memory>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

PYBIND11_MAKE_OPAQUE(std::vector<batoid::Ray>);
PYBIND11_MAKE_OPAQUE(std::vector<batoid::Intersection>);

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportPlane(py::module& m) {
        py::class_<Plane, std::shared_ptr<Plane>, Surface>(m, "Plane")
            .def(py::init<>(), "init");
    }
}
