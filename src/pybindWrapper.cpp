#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // This is needed for std::vector binding
#include "model.hpp"
#include <cufft.h>
#include <pybind11/numpy.h>
#include <vector>
#include <complex>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(libpyCudaPolygon, m) {
    pybind11::enum_<simControlStruct::modelEnum>(m, "modelEnum")
        .value("normal", simControlStruct::modelEnum::normal)
        .value("abnormal", simControlStruct::modelEnum::abnormal);

    py::class_<Model>(m, "Model")
        .def(py::init<int>())

        // Setters
        .def("setNumVertices", &Model::setNumVertices)
        .def("setPositions", &Model::setPositions)
        .def("setModelEnum", &Model::setModelEnum)

        // Getters
        .def("getNumVertices", &Model::getNumVertices)
        .def("getPositions", &Model::getPositions);
    }
