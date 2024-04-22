#include "coating.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;


namespace batoid {
    void pyExportCoating(py::module& m) {
        py::class_<Coating, std::shared_ptr<Coating>>(m, "CPPCoating")
            .def("getCoefs", [](const Coating& coating, double wavelength, double cosIncidenceAngle){
                double reflect, transmit;
                coating.getCoefs(wavelength, cosIncidenceAngle, reflect, transmit);
                return py::make_tuple(reflect, transmit);
            })
            .def("getReflect", &Coating::getReflect)
            .def("getTransmit", &Coating::getTransmit);

        py::class_<SimpleCoating, std::shared_ptr<SimpleCoating>, Coating>(m, "CPPSimpleCoating")
            .def(py::init<double,double>(), "reflectivity"_a, "transmissivity"_a);

        py::class_<TableCoating, std::shared_ptr<TableCoating>, Coating>(m, "CPPTableCoating")
            .def(py::init(
                [](
                    size_t w,
                    size_t r,
                    size_t t,
                    size_t size
                ){
                    return new TableCoating(
                        reinterpret_cast<double*>(w),
                        reinterpret_cast<double*>(r),
                        reinterpret_cast<double*>(t),
                        size
                    );
                }
        ));

    }
}
