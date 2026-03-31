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
        .value("edgeOnly", simControlStruct::modelEnum::edgeOnly)
        .value("abnormal", simControlStruct::modelEnum::abnormal);

    py::class_<Model>(m, "Model")

        // initializers

        .def(py::init<int>())
        .def("initializeNeighborCells", &Model::initializeNeighborCells)

        // helpers

        .def("sortKeys", &Model::sortKeys)
        .def("getKeys", &Model::getKeys)

        // setters        

        .def("setNumVertices", &Model::setNumVertices)
        .def("setPositions", &Model::setPositions)
        .def("setForces", &Model::setForces)
        .def("setModelEnum", &Model::setModelEnum)
        .def("setStartIndices", &Model::setStartIndices)
        .def("setMaxEdgeLength", &Model::setMaxEdgeLength)
        .def("setTargetEdgeLengths", &Model::setTargetEdgeLengths)
        .def("setTargetAreas", &Model::setTargetAreas)
        .def("setStiffness", &Model::setStiffness)
        // updaters

        .def("updatePolygonGeometry", &Model::updatePolygonGeometry)
        .def("updatePerimeters", &Model::updatePerimeters)
        .def("updateNeighborCells", &Model::updateNeighborCells)
        .def("updateNeighbors", &Model::updateNeighbors)
        .def("updateValidAndCounts", &Model::updateValidAndCounts)
        .def("updateCompactedIntersections", &Model::updateCompactedIntersections)
        .def("updateOutersections", &Model::updateOutersections)
        .def("updateOverlapArea", &Model::updateOverlapArea)
        .def("updateForceEnergy", &Model::updateForceEnergy)
        .def("updatePositions", &Model::updatePositions)
        .def("updateConstraintForces", &Model::updateConstraintForces)

        // misc
        .def("resetAreas", &Model:: resetAreas)

        // getters

        .def("getNumVertices", &Model::getNumVertices)
        .def("getNumPolygons", &Model::getNumPolygons)
        .def("getShapeId", &Model::getShapeId)
        .def("getPositions", &Model::getPositions)
        .def("getIntersectionsCounter", &Model::getIntersectionsCounter)
        .def("getModelEnum", &Model::getModelEnum)
        .def("getStartIndices", &Model::getStartIndices)
        .def("getMaxEdgeLength", &Model::getMaxEdgeLength)
        .def("getAreas", &Model::getAreas)
        .def("getNeighborCells", &Model::getNeighborCells)
        .def("getNeighborIndices", &Model::getNeighborIndices)
        .def("getIntersections", &Model::getIntersections)
        .def("getNumIntersections", &Model::getNumIntersections)
        .def("getOutersections", &Model::getOutersections)
        .def("getNeighbors", &Model::getNeighbors)
        .def("getNumNeighbors", &Model::getNumNeighbors)
        .def("getBoxCounts", &Model::getBoxCounts)
        .def("getInsideFlag", &Model::getInsideFlag)
        .def("getPerimeters", &Model::getPerimeters)
        .def("getForces", &Model::getForces)
        .def("getTU", &Model::getTU)
        .def("getUT", &Model::getUT)
        .def("getShapeCounts", &Model::getShapeCounts)
        .def("getForces", &Model::getForces)
        .def("getEnergy", &Model::getEnergy)
        .def("getTargetEdgeLengths", &Model::getTargetEdgeLengths)
        .def("getConstraintForces", &Model::getConstraintForces)
        .def("getNorm2", &Model::getNorm2)
        .def("getConstraints", &Model::getConstraints)
        .def("getProjection", &Model::getProjection)
        .def("getMaxUnbalancedForce", &Model::getMaxUnbalancedForce)
        .def("getCOM", &Model::getCOM)
        .def("getOverlapArea", &Model::getOverlapArea);
}
