#include "grating.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace batoid {
    void pyExportGrating(py::module& m) {
        py::class_<Grating, std::shared_ptr<Grating>, Surface>(m, "CPPGrating")
            .def(py::init<int>(), "init", "order"_a)
            .def("getN", py::vectorize(&Grating::getN))
            .def("disp_axis",
                [](const Grating& s, size_t xarr, size_t yarr, size_t size, size_t outarr)
                {
                    double tx, ty, tz;
                    double* xptr = reinterpret_cast<double*>(xarr);
                    double* yptr = reinterpret_cast<double*>(yarr);
                    double* outptr = reinterpret_cast<double*>(outarr);
                    for(int i=0; i<size; i++) {
                        s.disp_axis(xptr[i], yptr[i], outptr[i], outptr[i+size], outptr[i+2*size]);
                    }
                }
            )
        ;
    }
    void pyExportSimpleGrating(py::module& m) {
        py::class_<SimpleGrating, std::shared_ptr<SimpleGrating>, Grating, Surface>(m, "CPPSimpleGrating")
            .def(py::init<int,double,double>(), "init", "order"_a, "N"_a, "rot"_a)
        ;
    }
}
