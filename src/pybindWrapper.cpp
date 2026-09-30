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
    pybind11::enum_<minimizerEnum>(m, "minimizerEnum")
        .value("GD",   minimizerEnum::GD)
        .value("FIRE", minimizerEnum::FIRE);

    pybind11::enum_<simControlStruct::modelEnum>(m, "modelEnum")
        .value("normal",   simControlStruct::modelEnum::normal)
        .value("edgeOnly", simControlStruct::modelEnum::edgeOnly)
        .value("areaOnly", simControlStruct::modelEnum::areaOnly)
        .value("softBody", simControlStruct::modelEnum::softBody)
        .value("hybrid",   simControlStruct::modelEnum::hybrid)
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
        .def("setVertices", &Model::setVertices)
        .def("setForces", &Model::setForces)
        .def("setModelEnum", &Model::setModelEnum)
        .def("setStartIndices", &Model::setStartIndices)
        .def("setMaxEdgeLength", &Model::setMaxEdgeLength)
        .def("setTargetEdgeLengths", &Model::setTargetEdgeLengths)
        .def("setTargetAreas", &Model::setTargetAreas)
        .def("setStiffness", &Model::setStiffness)
        .def("setCompressibility", &Model::setCompressibility)
        .def("getStiffness", &Model::getStiffness)
        .def("getCompressibility", &Model::getCompressibility)
        // updaters

        .def("updatePolygonGeometry", &Model::updatePolygonGeometry)
        .def("projectForce", &Model::projectForce)
        .def("saveTentativeVertices", &Model::saveTentativeVertices)
        .def("getMaxEffectiveForce", &Model::getMaxEffectiveForce)
.def("updateNeighborCells", &Model::updateNeighborCells)
        .def("updateNeighbors", &Model::updateNeighbors)
        .def("updateValidAndCounts", &Model::updateValidAndCounts)
        .def("updateIntersections", &Model::updateIntersections)
        .def("updateOutersections", &Model::updateOutersections)
        .def("updateoverlapAreasGOLD", &Model::updateoverlapAreasGOLD)
        .def("updateoverlapAreas", &Model::updateoverlapAreas)
        .def("updateForceEnergy", &Model::updateForceEnergy)
        .def("updateVertices", &Model::updateVertices)

        // misc
        .def("resetAreas", &Model:: resetAreas)

        // getters

        .def("getNumVertices", &Model::getNumVertices)
        .def("getNumPolygons", &Model::getNumPolygons)
        .def("getShapeId", &Model::getShapeId)
        .def("getVertices", &Model::getVertices)
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
        .def("getConstraints", &Model::getConstraints)
        .def("getTargetEdgeLengths", &Model::getTargetEdgeLengths)
        .def("getTargetAreas", &Model::getTargetAreas)
        .def("getEdgeLengths", &Model::getEdgeLengths)
        .def("getMaxUnbalancedForce", &Model::getMaxUnbalancedForce)
        .def("getCOM", &Model::getCOM)
        .def("getoverlapAreasGOLD", &Model::getoverlapAreasGOLD)
        .def("getoverlapAreas", &Model::getoverlapAreas)
        .def("resetVelocities", &Model::resetVelocities)
        .def("minimizeFIREStep", &Model::minimizeFIREStep)
        .def("minimizeFIRE", &Model::minimizeFIRE);
}
