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
        .value("abnormal", simControlStruct::modelEnum::abnormal)
        .value("rounded",      simControlStruct::modelEnum::rounded)
        .value("areaSquared",  simControlStruct::modelEnum::areaSquared);

    pybind11::enum_<simControlStruct::neighborTypeEnum>(m, "neighborTypeEnum")
        .value("cells", simControlStruct::neighborTypeEnum::cells)
        .value("balls", simControlStruct::neighborTypeEnum::balls);

    py::class_<Model>(m, "Model")

        // initializers

        .def(py::init<int>())
        .def("initializeNeighborCells", &Model::initializeNeighborCells)
        .def("initializeNeighborBall",  &Model::initializeNeighborBall)

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
        .def("setCompressibility", &Model::setCompressibility)
        .def("getStiffness", &Model::getStiffness)
        .def("getCompressibility", &Model::getCompressibility)
        .def("setDelta", &Model::setDelta)
        .def("getDelta", &Model::getDelta)
        // Per-polygon uniform-delta mode (area-preserving rounded).
        .def("enablePerPolygonDelta", &Model::enablePerPolygonDelta)
        .def("disablePerPolygonDelta", &Model::disablePerPolygonDelta)
        .def("isPerPolygonDeltaEnabled", &Model::isPerPolygonDeltaEnabled)
        .def("updatePolyDelta", &Model::updatePolyDelta)
        .def("getPolyDelta", &Model::getPolyDelta)
        .def("getPolyDeltaTargets", &Model::getPolyDeltaTargets)
        .def("setPolyDelta", &Model::setPolyDelta)
        // Phase C scaffolding (sort + RLE).
        .def("runFeaturePairPhaseC_sort", &Model::runFeaturePairPhaseC_sort)
        .def("getLastNumUniqueFA", &Model::getLastNumUniqueFA)
        .def("getSortedCrossingFA", &Model::getSortedCrossingFA)
        .def("getSortedCrossingParamA", &Model::getSortedCrossingParamA)
        .def("getSortedCrossingFB", &Model::getSortedCrossingFB)
        .def("getSortedCrossingX", &Model::getSortedCrossingX)
        .def("getFeatureUniqueFA", &Model::getFeatureUniqueFA)
        .def("getFeatureLengths", &Model::getFeatureLengths)
        .def("runFeaturePairPhaseC_walkArea", &Model::runFeaturePairPhaseC_walkArea)
        .def("runFeaturePairPhaseC_walkAreaPerPolygon", &Model::runFeaturePairPhaseC_walkAreaPerPolygon)
        .def("runFeaturePairPhaseC_walkAreaAndForce", &Model::runFeaturePairPhaseC_walkAreaAndForce)
        .def("runFeaturePairPhaseC_walkAreaAndForceDual", &Model::runFeaturePairPhaseC_walkAreaAndForceDual)
        .def("getPhaseCForce", &Model::getPhaseCForce)
        .def("setSearchFactor", &Model::setSearchFactor)
        .def("getSearchFactor", &Model::getSearchFactor)
        .def("setNeighborType", &Model::setNeighborType)
        .def("getNeighborType", &Model::getNeighborType)
        // updaters

        .def("updatePolygonGeometry", &Model::updatePolygonGeometry)
        .def("projectForce", &Model::projectForce)
        .def("shakeProject", &Model::shakeProject)
        .def("getLastShakeIters", &Model::getLastShakeIters)
        .def("saveTentativePositions", &Model::saveTentativePositions)
        .def("getMaxEffectiveForce", &Model::getMaxEffectiveForce)
.def("updateNeighborCells", &Model::updateNeighborCells)
        .def("updateNeighborBall",  &Model::updateNeighborBall)
        .def("updateNeighbors", &Model::updateNeighbors)
        .def("updateValidAndCounts", &Model::updateValidAndCounts)
        .def("updateCompactedIntersections", &Model::updateCompactedIntersections)
        .def("updateOutersections", &Model::updateOutersections)
        .def("updateOverlapArea", &Model::updateOverlapArea)
        .def("updateForceEnergy", &Model::updateForceEnergy)
        .def("updatePositions", &Model::updatePositions)

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
        .def("getBallNeighbors",     &Model::getBallNeighbors)
        .def("getNumBallNeighbors",  &Model::getNumBallNeighbors)
        .def("getBallMaxNeighbors",  &Model::getBallMaxNeighbors)
        .def("getBallRebuildCount",  &Model::getBallRebuildCount)
        // Phase 2 feature-pair refactor: test-only entry points for the
        // new A->B path; not wired into updateForceEnergy yet.
        .def("runFeaturePairPhaseA",  &Model::runFeaturePairPhaseA)
        .def("getFeaturePairCapacity", &Model::getFeaturePairCapacity)
        .def("runFeaturePairPhaseB",  &Model::runFeaturePairPhaseB)
        .def("getCrossingCapacity",   &Model::getCrossingCapacity)
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
        .def("getPairArea", &Model::getPairArea)
        .def("getConstraints", &Model::getConstraints)
        .def("getTargetEdgeLengths", &Model::getTargetEdgeLengths)
        .def("getTargetAreas", &Model::getTargetAreas)
        .def("getEdgeLengths", &Model::getEdgeLengths)
        .def("getMaxUnbalancedForce", &Model::getMaxUnbalancedForce)
        .def("getCOM", &Model::getCOM)
        .def("getOverlapArea", &Model::getOverlapArea)
        .def("resetVelocities", &Model::resetVelocities)
        .def("minimizeFIREStep", &Model::minimizeFIREStep,
             py::arg("dt"), py::arg("alpha"), py::arg("nPos"),
             py::arg("dtMax") = 0.1, py::arg("alphaStart") = 0.1,
             py::arg("fAlpha") = 0.99, py::arg("fInc") = 1.1, py::arg("fDec") = 0.5,
             py::arg("nMin") = 5, py::arg("shakeIter") = 5,
             py::arg("rollbackRelTol") = 1e-10, py::arg("rollbackAbsTol") = 1e-14)
        .def("minimizeFIRE", &Model::minimizeFIRE,
             py::arg("maxForceThreshold"), py::arg("dtInit"), py::arg("maxSteps"),
             py::arg("dtMax") = 0.1, py::arg("alphaStart") = 0.1,
             py::arg("fAlpha") = 0.99, py::arg("fInc") = 1.1, py::arg("fDec") = 0.5,
             py::arg("nMin") = 5, py::arg("shakeIter") = 5,
             py::arg("rollbackRelTol") = 1e-10, py::arg("rollbackAbsTol") = 1e-14);
}
