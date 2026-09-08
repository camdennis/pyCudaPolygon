#from .pyCudaPolygonLink import libpyCudaPolygon as lpcp
#from . import enums
from pyCudaPolygonLink import libpyCudaPolygon as lpcp
import enums

import numpy as np
from matplotlib import pyplot as plt
import pkgutil
import importlib
import os
import shutil
import glob
import sys
from collections import defaultdict
from tqdm import tqdm
import warnings

# Load the `polygonMixins` package robustly so imports work whether this
# module is loaded as a package or directly as a top-level module.
try:
    polygonMixins = importlib.import_module(__name__ + ".polygonMixins")
except Exception:
    try:
        pkg_root = __name__.split(".")[0]
        polygonMixins = importlib.import_module(pkg_root + ".polygonMixins")
    except Exception:
        # Last resort: try a plain import (works if sys.path is parent of package)
        polygonMixins = importlib.import_module("polygonMixins")

mixins = {}
for loader, moduleName, isPackage in pkgutil.walk_packages(polygonMixins.__path__):
    module = importlib.import_module(polygonMixins.__name__ + "." + moduleName)
    mixins[moduleName] = getattr(module, "Mixin")

class model(lpcp.Model, *mixins.values()):

    # initializers

    def __init__(self, size = 0, seed = None, modelType = "normal", stiffness = 0, compressibility = 0):
        lpcp.Model.__init__(self, size)
        self.setModelEnum(modelType)
        self.setStiffness(stiffness)
        self.setCompressibility(compressibility)
        self._mel = None  # tracks last user-set maxEdgeLength scalar
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
            #lpcp.Model.initializeRandomSeed(self, seed)

    def initializeRandomSeed(self, seed):
        self.rng = np.random.default_rng(seed)
        lpcp.Model.initializeRandomSeed(self, seed)

    def initializeNeighborCells(self):
        lpcp.Model.initializeNeighborCells(self)
        # Restore user-set maxEdgeLength (preserves boxSize); fall back to 0.5
        if self._mel is not None:
            lpcp.Model.setMaxEdgeLength(self, self._mel)
        else:
            self.setMaxEdgeLength()

    def initializeNeighborBall(self):
        lpcp.Model.initializeNeighborBall(self)
        # Ensure maxEdgeLength is set so the first updateNeighborBall has a
        # meaningful ball radius. Mirrors the behavior of initializeNeighborCells.
        if self._mel is not None:
            lpcp.Model.setMaxEdgeLength(self, self._mel)
        else:
            self.setMaxEdgeLength()

    def setNeighborType(self, neighborType):
        if isinstance(neighborType, str):
            if neighborType == "cells":
                lpcp.Model.setNeighborType(self, enums.neighborTypeEnum.cells)
            elif neighborType == "balls":
                lpcp.Model.setNeighborType(self, enums.neighborTypeEnum.balls)
            else:
                raise ValueError(f"Unknown neighborType: {neighborType!r}")
        else:
            lpcp.Model.setNeighborType(self, neighborType)

    def getNeighborType(self):
        return lpcp.Model.getNeighborType(self)

    def setSearchFactor(self, searchFactor):
        lpcp.Model.setSearchFactor(self, searchFactor)

    def getSearchFactor(self):
        return lpcp.Model.getSearchFactor(self)

    def updateNeighborBall(self):
        lpcp.Model.updateNeighborBall(self)
    
    def initForceEnergy(self):
        t = self.getModelEnum()
        if t == "normal":
            self.initializeNeighborCells()
            self.updateNeighborCells()
            self.updateNeighbors()
            self.updateOutersections()

    # setters

    def setStiffness(self, stiffness):
        lpcp.Model.setStiffness(self, stiffness)

    def setDelta(self, delta):
        lpcp.Model.setDelta(self, delta)

    def getDelta(self):
        return lpcp.Model.getDelta(self)

    def setModelEnum(self, modelType):
        if modelType == "abnormal":
            lpcp.Model.setModelEnum(self, enums.modelEnum.abnormal)
        elif modelType == "edgeOnly":
            lpcp.Model.setModelEnum(self, enums.modelEnum.edgeOnly)
        elif modelType == "areaOnly":
            lpcp.Model.setModelEnum(self, enums.modelEnum.areaOnly)
        elif modelType == "softBody":
            lpcp.Model.setModelEnum(self, enums.modelEnum.softBody)
        elif modelType == "normal":
            lpcp.Model.setModelEnum(self, enums.modelEnum.normal)
        elif modelType == "hybrid":
            lpcp.Model.setModelEnum(self, enums.modelEnum.hybrid)
        elif modelType == "rounded":
            lpcp.Model.setModelEnum(self, enums.modelEnum.rounded)
        elif modelType == "areaSquared":
            lpcp.Model.setModelEnum(self, enums.modelEnum.areaSquared)
        else:
            raise Exception("That Model type does not exist")

    def setNumVertices(self, n):
        self.size = n
        lpcp.Model.setNumVertices(self, n)

    def setPositions(self, positions):
        lpcp.Model.setPositions(self, positions)

    def setnArray(self, nArray):
        nArray = nArray.astype(int)
        if nArray.sum() != self.getNumVertices():
            raise ValueError(f"Total vertices from nArray ({nArray.sum()}) "
                            f"does not match model size ({self.getNumVertices()})")
        startIndices = np.concatenate(([0], np.cumsum(nArray)))
        lpcp.Model.setStartIndices(self, startIndices)

    def setAreas(self, targetAreas):
        nArray = self.getnArray()
        self.updatePolygonGeometry()
        areas = self.getAreas()
        positions = self.getPositions().copy().reshape(self.getNumVertices(), 2)

        start = 0
        for i, n in enumerate(nArray):
            s = start
            poly = positions[s:s + n].copy()   # shape (n,2)

            # Unwrap polygon relative to the first vertex to avoid periodic jumps
            ref = poly[0].copy()
            for j in range(1, n):
                d = poly[j] - ref
                # shift to nearest image (puts d in [-0.5,0.5] per coordinate)
                d = d - np.round(d)
                poly[j] = ref + d

            # compute area on unwrapped coordinates (shoelace)
            x = poly[:, 0]
            y = poly[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
            if area <= 0:
                start += n
                continue

            # scale about centroid (unwrapped)
            scale = np.sqrt(targetAreas[i] / area)
            centroid = poly.mean(axis=0)
            poly = (poly - centroid) * scale + centroid

            # re-wrap into [0,1)
            poly = np.mod(poly, 1.0)

            positions[s:s + n] = poly
            start += n

        # write back flattened positions and update areas
        self.setPositions(positions.reshape(self.getNumVertices() * 2))
        self.updatePolygonGeometry()
        areas = self.getAreas()

    def setMonoArea(self, phi = 1):
        # This overrides phi!
        targetArea = phi / self.getNumPolygons()
        self.updatePolygonGeometry()
        areas = self.getAreas()
        # Let's keep phi the same
        n = self.getNumPolygons()
        targetAreas = np.ones(n) * phi / n
        if (np.max(targetAreas) > 1 / 9):
            raise Exception("The phi you have chosen has caused the shapes to be too large and compromised the PBCs")
        self.setAreas(targetAreas)

    def setPhi(self, phi):
        self.updatePolygonGeometry()
        areas = self.getAreas()
        totalArea = np.sum(areas)
        targetAreas = phi * areas / totalArea
        self.setAreas(targetAreas)

    def setMaxEdgeLength(self, maxEdgeLength = None):
        # Default (no arg or explicit None): compute the current max edge length
        # from positions. Pass a number to pin it explicitly (e.g. for a stable
        # ball radius across position changes).
        if maxEdgeLength is None:
            maxEdgeLength = 0
            nArray = self.getnArray()
            numShapes = len(nArray)
            startIndices = self.getStartIndices()
            positions = self.getPositions()
            for i in range(numShapes):
                pos = positions[2 * startIndices[i]:2 * startIndices[i + 1]].reshape(nArray[i], 2)
                diff = np.diff(np.concatenate((pos, [pos[0]])), axis = 0)
                diff += 1.5
                diff %= 1
                diff -= 0.5
                length = np.max(np.sqrt(np.sum(diff**2, axis = 1)))
                maxEdgeLength = np.max([maxEdgeLength, length])
        self._mel = maxEdgeLength
        lpcp.Model.setMaxEdgeLength(self, maxEdgeLength)

    def setBiPerimeters(self, kappa, ratio = 1.4):
        nArray = self.getnArray()
        numPolygons = len(nArray)
        numVertices = self.getNumVertices()
        self.setMaxEdgeLength()
        # Ball-based init: also sets up topology arrays (shapeId/next/prev)
        # that updatePolygonGeometry below needs.
        self.initializeNeighborBall()
        mid = numPolygons // 2
        # The polygon 1 has total perimeter 1
        # Polygon 2 has total perimeter ratio r
        # Polygon 1 has a1 = (1 / kappa)**2
        # Polygon 2 has a2 = (r / kappa)**2
        self.updatePolygonGeometry()
        minArea = np.min(self.getAreas())
        targetAreas = np.ones(numPolygons) * minArea
        targetAreas[:mid] /= ratio**2
        self.setTargetAreas(targetAreas)
        #self.resetAreas()
        targetEdgeLengths = kappa * np.sqrt(targetAreas) / nArray
        self.setTargetEdgeLengths(targetEdgeLengths)
        self.updatePolygonGeometry()

    def setTargetEdgeLengths(self, targetEdgeLengths):
        lpcp.Model.setTargetEdgeLengths(self, targetEdgeLengths)

    def setTargetAreas(self, targetAreas):
        lpcp.Model.setTargetAreas(self, targetAreas)

    def setStiffness(self, stiffness):
        lpcp.Model.setStiffness(self, stiffness)

    def setCompressibility(self, compressibility):
        lpcp.Model.setCompressibility(self, compressibility)

    def getStiffness(self):
        return lpcp.Model.getStiffness(self)

    def getCompressibility(self):
        return lpcp.Model.getCompressibility(self)

    def setPhi(self, phi):
        self.updatePolygonGeometry()
        targetAreas = self.getTargetAreas()
        areaRatio = phi / np.sum(targetAreas)
        lengthRatio = np.sqrt(areaRatio)
        targetAreas *= areaRatio
        targetEdgeLengths = self.getTargetEdgeLengths() * lengthRatio
        self.setTargetAreas(targetAreas)
        self.setTargetEdgeLengths(targetEdgeLengths)
        self.updatePolygonGeometry()
        self.resetAreas()
        self.updatePolygonGeometry()

    # getters

    def getModelEnum(self):
        return lpcp.Model.getModelEnum(self)

    def getRandomSeed(self):
        return lpcp.Model.getRandomSeed(self)

    def getNumVertices(self):
        return int(lpcp.Model.getNumVertices(self))

    def getNumPolygons(self):
        return lpcp.Model.getNumPolygons(self)

    def getShapeId(self):
        return np.array(lpcp.Model.getShapeId(self))

    def getPositions(self):
        return np.array(lpcp.Model.getPositions(self))

    def getAreaPerOverlap(self):
        return np.array(lpcp.Model.getAreaPerOverlap(self))

    def getIntersectionsCounter(self):
        return np.array(lpcp.Model.getIntersectionsCounter(self))

    def getNumNeighbors(self):
        return np.array(lpcp.Model.getNumNeighbors(self))

    def getNeighbors(self):
        v = self.getNumVertices()
        neighbors = np.array(lpcp.Model.getNeighbors(self))
        maxNeighbors = len(neighbors) // v
        neighbors = neighbors.reshape(v, maxNeighbors)
        neigh = dict()
        numNeighbors = self.getNumNeighbors()
        for i, neighbor in enumerate(neighbors):
            allNeighbors = neighbor[:numNeighbors[i]]
            if (len(allNeighbors) == 0):
                continue
            neigh[i] = allNeighbors
        return neigh

    def getInsideFlag(self):
        # v = self.getNumVertices()
        return np.array(lpcp.Model.getInsideFlag(self))

    def getTU(self):
        #v = self.getNumVertices()
        return np.array(lpcp.Model.getTU(self))

    def getUT(self):
        #v = self.getNumVertices()
        return np.array(lpcp.Model.getUT(self))

    def getIntersections(self):
        return np.array(lpcp.Model.getIntersections(self))

    def getKeys(self):
        return np.array(lpcp.Model.getKeys(self))

    def getOutersections(self):
        return np.array(lpcp.Model.getOutersections(self))

    def getNumIntersections(self):
        return lpcp.Model.getNumIntersections(self)

    def getNeighborCells(self):
        return np.array(lpcp.Model.getNeighborCells(self))

    def getBoxCounts(self):
        return np.array(lpcp.Model.getBoxCounts(self))

    def getNeighborIndices(self):
        return np.array(lpcp.Model.getNeighborIndices(self))

    def getStartIndices(self):
        return lpcp.Model.getStartIndices(self)

    def getnArray(self):
        return np.diff(self.getStartIndices())

    def getEnergy(self):
        return lpcp.Model.getEnergy(self)

    def getPairArea(self):
        return np.array(lpcp.Model.getPairArea(self))

    def getForces(self):
        return np.array(lpcp.Model.getForces(self))


    def getCentersOfMass(self):
        positions = self.getPositions()
        nArray = self.getnArray()
        # Loop over n
        startIndices = np.concatenate((np.array([0]), np.cumsum(nArray)))
        csom = []
        for n, s in zip(nArray, startIndices):
            polygonPos = positions[2 * s : 2 * s + 2 * n].reshape(n, 2) + 0
            polygonPos -= positions[2 * s: 2 * s + 2]
            polygonPos += 1.5
            polygonPos %= 1.0
            polygonPos -= 0.5
            csom.append(np.mean(polygonPos, axis = 0) + positions[2 * s : 2 * s + 2])
        return np.concatenate(csom)

    def getShapeCounts(self):
        return np.array(lpcp.Model.getShapeCounts(self))

    def getIntersectionsCounter(self):
        intersectionsCounter = np.array(lpcp.Model.getIntersectionsCounter(self))
        s = int(np.sqrt(len(intersectionsCounter)))
        return intersectionsCounter.reshape(s, s)
    
    def getDf(self, vi, vzi, vj, vzj):
        dj = vzj - vj + 1.5
        di = vzi - vi + 1.5
        dij = vj - vi + 1.5
        dj %= 1
        di %= 1
        dij %= 1
        dj -= 0.5
        di -= 0.5
        dij -= 0.5

        w = dj[0]*di[1] - dj[1]*di[0]
        k = dj[0]*dij[1] - dj[1]*dij[0]
        u = k / w   # kept as requested

        df = np.zeros((2, 8))

        dk = np.zeros((2, 4))
        dk[0, 0] = dj[1]
        dk[0, 2] = -dij[1] - dj[1]
        dk[0, 3] = dij[1]
        dk[1, 0] = -dj[0]
        dk[1, 2] = dj[0] + dij[0]
        dk[1, 3] = -dij[0]

        dw = np.zeros((2, 4))
        dw[0, 0] = dj[1]
        dw[0, 1] = -dj[1]
        dw[0, 2] = -di[1]
        dw[0, 3] = di[1]
        dw[1, 0] = -dj[0]
        dw[1, 1] = dj[0]
        dw[1, 2] = di[0]
        dw[1, 3] = -di[0]

        for alpha in range(2):
            for beta in range(2):
                for p in range(4):
                    du = dk[beta, p] / w - u * dw[beta, p] / w
                    df[alpha, 2 * p + beta] += di[alpha] * du   # += is crucial

                if alpha == beta:
                    df[alpha, beta] += 1 - u
                    df[alpha, 2 + beta] += u

        return df

    def getConstraintViolation(self):
        areas = self.getAreas()
        edgeLengths = self.getEdgeLengths()
        targetAreas = self.getTargetAreas()
        targetEdgeLengths = self.getTargetEdgeLengths()
        shapeId = self.getShapeId()
        rmsAreaViolation = np.sqrt(np.mean((1 - areas / targetAreas)**2))
        nArray = self.getnArray()
        idx = np.repeat(np.arange(len(nArray)), nArray)
        perimeters = np.bincount(idx, weights = edgeLengths)
        targetPerimeters = nArray * targetEdgeLengths
        rmsPerimeterViolation = np.sqrt(np.mean((1 - perimeters / targetPerimeters)**2))
        rmsEdgeViolation = np.sqrt(np.mean((1 - edgeLengths / np.repeat(targetEdgeLengths, nArray))**2))
        return {"rmsAreaViolation":rmsAreaViolation, "rmsEdgeViolation":rmsEdgeViolation, "rmsPerimeterViolation":rmsPerimeterViolation}

    def getConstraints(self):
        return np.array(lpcp.Model.getConstraints(self))

    def getAreas(self):
        # This is MC for now
        return np.array(lpcp.Model.getAreas(self))

    def getTargetEdgeLengths(self):
        return np.array(lpcp.Model.getTargetEdgeLengths(self))

    def getTargetAreas(self):
        return np.array(lpcp.Model.getTargetAreas(self))

    def getEdgeLengths(self):
        return np.array(lpcp.Model.getEdgeLengths(self))

    def getCOM(self):
        return np.array(lpcp.Model.getCOM(self))

    def getPhi(self):
        return np.sum(self.getAreas())

    def getRestEdgeLengths(self):
        return np.array(lpcp.Model.getEdgeLengths(self))

    def getMaxUnbalancedForce(self):
        return lpcp.Model.getMaxUnbalancedForce(self)

    def getMaxTangentialForce(self):
        """Max |F| projected onto the constraint-tangent subspace — the
        physically meaningful convergence indicator under SHAKE constraints.

        Raw `getMaxUnbalancedForce()` returns max |F| in the full 2N-dim space.
        At an *unconstrained* minimum (e.g. φ<φ_jam where no overlap is required)
        both go to zero. At a *constrained* minimum (e.g. jammed φ=1.0 where
        polygons must overlap) raw |F| floors at the constraint-force scale,
        while this projected value → 0.

        Side effect: re-projects the internal force array. Call
        `updateForceEnergy()` afterwards if you need the raw force back."""
        self.updateForceEnergy()
        self.projectForce()
        f_tan = self.getMaxUnbalancedForce()
        self.updateForceEnergy()
        return f_tan

    def getOverlapArea(self):
        return lpcp.Model.getOverlapArea(self)

    def getConstrainedForcePython(self, force):
        # We want to get the constrained force using all
        # of the constraints (one per edge per polygon)
        startIndices = self.getStartIndices()
        nArray = self.getnArray()
        positions = self.getPositions()
        numPolygons = self.getNumPolygons()
        constrainedForce = force.copy()
        for polygon in range(numPolygons):
            start = startIndices[polygon]
            n = nArray[polygon]
            end = start + n
            pos = positions[2 * start : 2 * end]
            x = pos[::2]
            y = pos[1::2]
            # Area gradient row
            ga = np.zeros(2 * n)
            diffx = np.roll(x, -1) - np.roll(x, 1) + 1.5
            diffy = np.roll(y, -1) - np.roll(y, 1) + 1.5
            diffx %= 1
            diffy %= 1
            diffx -= 0.5
            diffy -= 0.5
            ga[::2]  =  diffy / 2
            ga[1::2] = -diffx / 2
            # Edge gradient rows: forward unit vector e[k] = (r_{k+1} - r_k) / |...|
            gk = np.zeros((n + 1, 2 * n))
            tkx = np.roll(x, -1) - x + 1.5
            tky = np.roll(y, -1) - y + 1.5
            tkx %= 1.0
            tky %= 1.0
            tkx -= 0.5
            tky -= 0.5
            norm = np.sqrt(tkx**2 + tky**2)
            tkx /= norm
            tky /= norm
            for edge in range(n):
                kp1 = (edge + 1) % n
                gk[edge][edge * 2]     = -tkx[edge]
                gk[edge][edge * 2 + 1] = -tky[edge]
                gk[edge][kp1 * 2]      =  tkx[edge]
                gk[edge][kp1 * 2 + 1]  =  tky[edge]
            gk[n] = ga
            Q, R = np.linalg.qr(gk.T)
            f = force[2 * start : 2 * end]
            constrainedForce[2 * start : 2 * end] = f - Q @ (Q.T @ f)
        return constrainedForce

    # updaters

    def updateNeighbors(self):
        lpcp.Model.updateNeighbors(self)

    def updateValidAndCounts(self):
        return lpcp.Model.updateValidAndCounts(self)

    def updateCompactedIntersections(self):
        lpcp.Model.updateCompactedIntersections(self)

    def updateOverlapArea(self, pointDensity):
        # This is MC
        lpcp.Model.updateOverlapArea(self, pointDensity)

    def updatePolygonGeometry(self):
        # This is MC
        lpcp.Model.updatePolygonGeometry(self)

    def projectForce(self):
        lpcp.Model.projectForce(self)

    def shakeProject(self, nIter=100, tol=1e-15):
        return lpcp.Model.shakeProject(self, nIter, tol)

    def getLastShakeIters(self):
        return lpcp.Model.getLastShakeIters(self)

    def saveTentativePositions(self):
        lpcp.Model.saveTentativePositions(self)

    def getMaxEffectiveForce(self, dt, minimizerType="GD"):
        mEnum = enums.minimizerEnum.GD if minimizerType == "GD" else enums.minimizerEnum.FIRE
        return lpcp.Model.getMaxEffectiveForce(self, dt, mEnum)

    def updateNeighborCells(self):
        lpcp.Model.updateNeighborCells(self)
        
    def updateForceEnergy(self):
        lpcp.Model.updateForceEnergy(self)

    def updatePositions(self, dt):
        lpcp.Model.updatePositions(self, dt)

    def resetAreas(self):
        lpcp.Model.resetAreas(self)

    # helpers

    def z(self, i):
        startIndices = self.getStartIndices()
        shapeId = self.getShapeId()[i]
        if (i == startIndices[shapeId + 1] - 1):
            return startIndices[shapeId]
        return i + 1

    def zp(self, i):
        startIndices = self.getStartIndices()
        shapeId = self.getShapeId()[i]
        if (i == startIndices[shapeId]):
            return startIndices[shapeId + 1] - 1
        return i - 1

    def unpackIntersections(self):
        intersections = self.getIntersections()
        sj = (intersections >> 48) & 0xFFFF
        si = (intersections >> 32) & 0xFFFF
        i = (intersections >> 16) & 0xFFFF
        j = (intersections) & 0xFFFF
        return np.vstack((sj, si, i, j)).T

    def unpackOutersections(self):
        outersections = self.getOutersections()
        sj = (outersections >> 48) & 0xFFFF
        si = (outersections >> 32) & 0xFFFF
        i = (outersections >> 16) & 0xFFFF
        j = (outersections) & 0xFFFF
        return np.vstack((sj, si, i, j)).T
    
    def sortKeys(self, endBit):
        lpcp.Model.sortKeys(self, endBit)

    # misc

    def minimizeGDStep(self, dt = 1e-3, addedForce = None, dontMove = False, maxDisplacement = 0.05, nIter = 100, tol = 1e-15, minimizerType = "GD"):
        if self.getModelEnum() not in ("edgeOnly", "areaOnly", "softBody"):
            self.updateNeighborCells()
            self.updateNeighbors()
            self.updateOutersections()
        self.updatePolygonGeometry()
        self.updateForceEnergy()
        actualIter = 0
        if addedForce is not None:
            self.setForces(self.getForces() + addedForce)
        if (self.getMaxUnbalancedForce() * dt > maxDisplacement):
            print("here")
            return -1, -1
        if dontMove:
            return self.getEnergy(), actualIter
        if self.getModelEnum() in ("normal", "rounded", "areaSquared"):
            self.projectForce()
        self.updatePositions(dt)
        actualIter = 0
        if self.getModelEnum() in ("normal", "rounded") and nIter > 0:
            self.shakeProject(nIter, tol)
        self.updatePolygonGeometry()
        return self.getEnergy(), 0

    def minimizeGD(self, maxUnbalancedForceThreshold = 1e-14, dt = 1e-3, maxSteps = -1, addedForce = None, progressBar = False, checkpointDir = None, checkpointFreq = 1, overwriteCheckpoint = False, maxDisplacement = 0.5, maxConstraintViolationThreshold = 1e-3, nIter = 100, tol = 1e-15, minimizerType = "GD"):
        if maxSteps == -1 and maxUnbalancedForceThreshold is None:
            raise ValueError("maxSteps=-1 requires maxUnbalancedForceThreshold to be set")
        dontMove = False
        if maxSteps == 0:
            maxSteps = 1
            dontMove = True

        total = maxSteps if maxSteps != -1 else None
        meanIter = 0
        maxIter = 0
        minIter = 1e9
        actualIter = 0
        with tqdm(total = total, desc = "Processing", disable = (not progressBar)) as pbar:
            successes = 0
            prevEnergy = None
            step = 0
            energy = None
            while maxSteps == -1 or step < maxSteps:
                if checkpointDir is not None and (step % checkpointFreq == 0):
                    if not os.path.isdir(checkpointDir):
                        os.makedirs(checkpointDir)
                    self.saveModel(checkpointDir + "/" + str(step), overwrite = overwriteCheckpoint)
                if progressBar:
                    pbar.update(1)
                energy, actualIter = self.minimizeGDStep(addedForce = addedForce, dt = dt, dontMove = dontMove, maxDisplacement = maxDisplacement, nIter = nIter, tol = tol, minimizerType = minimizerType)
                meanIter += actualIter / maxSteps
                maxIter = np.max((actualIter, maxIter))
                minIter = np.min((actualIter, minIter))
                if self.getModelEnum() == "normal":
                    fEff = self.getMaxEffectiveForce(dt, minimizerType)
                else:
                    fEff = self.getMaxUnbalancedForce()
                if maxUnbalancedForceThreshold is not None and fEff <= maxUnbalancedForceThreshold:
                    return energy, 0, np.array([0])
                if (self.getModelEnum() == "normal"  and np.max(self.getConstraintViolation()) > maxConstraintViolationThreshold):
                    return energy, 0, np.array([0])
                if prevEnergy is not None and (energy > prevEnergy or energy == -1):
                    dt = max(1e-32, dt / 2.1)
                    successes = 0
                else:
                    successes += 1
                    prevEnergy = energy
                if successes > 5:
                    successes = 0
                    dt = min(1e4, dt * 1.9)
                step += 1
        if energy is None:
            raise Exception("This is not an appropriate maximum step size. Reset maxSteps.")
        print(minIter, meanIter, maxIter)
        return energy, dt, np.array([minIter, meanIter, maxIter])

    def minimizeFIREStep(self, dt, alpha, nPos, dtMax=0.1, alphaStart=0.1, fAlpha=0.99, fInc=1.1, fDec=0.5, nMin=5, shakeIter=5, rollbackRelTol=1e-10, rollbackAbsTol=1e-14):
        """Single FIRE step. Returns (energy, dt, alpha, nPos)."""
        return lpcp.Model.minimizeFIREStep(self, dt, alpha, nPos, dtMax, alphaStart, fAlpha, fInc, fDec, nMin, shakeIter, rollbackRelTol, rollbackAbsTol)

    def minimizeFIRELoop(self, maxForceThreshold=1e-14, dt=1e-3, maxSteps=100000, dtMax=0.1, alphaStart=0.1, fAlpha=0.99, fInc=1.1, fDec=0.5, nMin=5, shakeIter=5, progressBar=False, rollbackRelTol=1e-10, rollbackAbsTol=1e-14):
        """
        FIRE minimizer generator. Yields (energy, maxForce, dt, step) each step.
        Stops when maxForce <= maxForceThreshold or maxSteps is reached.
        """
        lpcp.Model.resetVelocities(self)
        self.updatePolygonGeometry()
        self.updateForceEnergy()
        if self.getModelEnum() in ("normal", "rounded", "areaSquared"):
            self.projectForce()
        alpha = alphaStart
        nPos = 0
        with tqdm(total=maxSteps, desc="FIRE", disable=(not progressBar)) as pbar:
            for step in range(maxSteps):
                energy, dt, alpha, nPos = self.minimizeFIREStep(dt, alpha, nPos, dtMax, alphaStart, fAlpha, fInc, fDec, nMin, shakeIter, rollbackRelTol, rollbackAbsTol)
                maxForce = self.getMaxUnbalancedForce()
                pbar.update(1)
                yield energy, maxForce, dt, step + 1
                if maxForce <= maxForceThreshold:
                    return

    def minimizeFIRE(self, maxForceThreshold=1e-14, dt=1e-3, maxSteps=100000000, dtMax=0.1, alphaStart=0.1, fAlpha=0.99, fInc=1.1, fDec=0.5, nMin=5, shakeIter=5, progressBar=False, checkpointDir=None, checkpointFreq=1, overwriteCheckpoint=False, stuckEnergyTol=0.0, stuckEnergyCount=100):
        """
        FIRE minimizer. For the 'normal' model call
        updateNeighborCells/updateNeighbors/updateOutersections first.
        maxSteps=0 initializes state without stepping.

        stuckEnergyTol: optional absolute |ΔE| threshold for the "energy
            is stuck" early-stop. Default 0.0 = disabled (recommended).
            Near a true minimum, |ΔE| naturally shrinks proportionally to
            |F|^2·dt; tripping on |ΔE| < 1e-15 used to abort FIRE around
            |F|~1e-9 while convergence was still working to ~1e-15.
        stuckEnergyCount: how many *consecutive* steps below stuckEnergyTol
            trigger the early-stop. The counter resets on any step above tol.

        Returns (energy, dt, steps, [minShake, meanShake, maxShake]).
        """
        lpcp.Model.resetVelocities(self)
        if self.getModelEnum() == "normal":
            self.initForceEnergy()
        self.updatePolygonGeometry()
        self.updateForceEnergy()
        if self.getModelEnum() in ("normal", "rounded", "areaSquared"):
            self.projectForce()
        # One tiny GD step to refresh intersection detection and escape
        # containment configurations (polygon fully inside another) that are
        # invisible to the edge-crossing detector at exact loaded positions.
        counter = 0
        if (self.getModelEnum() == "rounded" or self.getModelEnum() == "normal") and maxSteps > 0:
            self.updatePositions(dt)
            self.updatePolygonGeometry()
            # normal still needs the cell-grid intersection finder; rounded
            # now reads the per-vertex candidate list, which the ball builder
            # populates.
            if self.getModelEnum() == "normal":
                self.updateNeighborCells()
                self.updateNeighbors()
                self.updateOutersections()
            else:
                self.updateNeighborBall()
            self.updateForceEnergy()
            self.projectForce()
            lpcp.Model.resetVelocities(self)
        if maxSteps == 0:
            return self.getEnergy(), dt, 0
        energy, steps = self.getEnergy(), 0
        minLoop = 1e9
        maxLoop = 0
        sumLoop = 0
        prevEnergy = 1e9

        for energy, _, dt, steps in self.minimizeFIRELoop(
                maxForceThreshold=maxForceThreshold, dt=dt, maxSteps=maxSteps,
                dtMax=dtMax, alphaStart=alphaStart, fAlpha=fAlpha,
                fInc=fInc, fDec=fDec, nMin=nMin, shakeIter=shakeIter,
                progressBar=progressBar):
            if checkpointDir is not None and (steps % checkpointFreq == 0):
                if not os.path.isdir(checkpointDir):
                    os.makedirs(checkpointDir)
                self.saveModel(checkpointDir + "/" + str(steps), overwrite=overwriteCheckpoint)
                print(self.getEnergy(), self.getMaxUnbalancedForce())
            curr = self.getLastShakeIters()
            minLoop = np.min((minLoop, curr))
            maxLoop = np.max((maxLoop, curr))
            sumLoop += curr
            if stuckEnergyTol > 0.0:
                if np.abs(energy - prevEnergy) < stuckEnergyTol:
                    counter += 1
                else:
                    counter = 0     # reset on any meaningful step
                if counter > stuckEnergyCount:
                    print(f"Energy stuck: |ΔE| < {stuckEnergyTol:g} for "
                          f"{stuckEnergyCount} consecutive steps")
                    break
            prevEnergy = energy

        if (steps == maxSteps):
            print("This may not have fully minimized")
        nSteps = max(steps, 1)
        return energy, dt, steps, np.array([minLoop, sumLoop / nSteps, maxLoop])

    # ── packing I/O ────────────────────────────────────────────────────────────
    # Single .npz file per packing. Stores everything needed to reconstruct
    # model state exactly: positions, nArray, target arrays, and every scalar
    # parameter (modelEnum, delta, maxEdgeLength, stiffness, compressibility).
    # Derived state (forces, energies, intersections, velocities) is NOT saved
    # — call updatePolygonGeometry + updateForceEnergy after loadModel.

    _SAVE_FORMAT_VERSION = 1

    def saveModel(self, path, overwrite=False, compress=False):
        """Atomically save the packing to a single .npz file.

        Args:
            path: filename (with or without .npz suffix).
            overwrite: replace an existing file.
            compress: use np.savez_compressed (smaller, slower).
        """
        if not path.endswith(".npz"):
            path = path + ".npz"
        if not overwrite and os.path.exists(path):
            raise FileExistsError(
                f"{path} exists; pass overwrite=True to replace it.")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        save_fn = np.savez_compressed if compress else np.savez
        tmp = path + ".tmp.npz"
        save_fn(tmp,
            _format_version  = np.int64(self._SAVE_FORMAT_VERSION),
            modelEnum        = np.asarray(str(self.getModelEnum())),
            positions        = np.asarray(self.getPositions(),         dtype=np.float64),
            nArray           = np.asarray(self.getnArray(),            dtype=np.int64),
            targetAreas      = np.asarray(self.getTargetAreas(),       dtype=np.float64),
            targetEdgeLengths= np.asarray(self.getTargetEdgeLengths(), dtype=np.float64),
            delta            = np.float64(self.getDelta()),
            maxEdgeLength    = np.float64(self.getMaxEdgeLength()),
            stiffness        = np.float64(self.getStiffness()),
            compressibility  = np.float64(self.getCompressibility()),
        )
        os.replace(tmp, path)

    def loadModel(self, path):
        """Load packing state from a .npz file written by saveModel.

        Restores everything needed to fully reproduce a saved configuration.
        Does NOT recompute forces/energies; call updatePolygonGeometry +
        updateForceEnergy afterwards.
        """
        if not path.endswith(".npz") and not os.path.exists(path):
            path = path + ".npz"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No saved model at {path}")

        data = np.load(path, allow_pickle=False)
        version = int(data["_format_version"])
        if version != self._SAVE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported save format version {version} "
                f"(this code understands {self._SAVE_FORMAT_VERSION})")

        positions = np.asarray(data["positions"], dtype=np.float64)
        nArray    = np.asarray(data["nArray"],    dtype=np.int64)

        # Order matters: model type → allocate by vertex count → topology
        # (nArray) → positions → per-polygon targets → scalar parameters →
        # cell structure.
        self.setModelEnum(str(data["modelEnum"]))
        self.setNumVertices(positions.size // 2)
        self.setnArray(nArray)
        self.setPositions(positions)
        self.setTargetAreas(      np.asarray(data["targetAreas"],       dtype=np.float64))
        self.setTargetEdgeLengths(np.asarray(data["targetEdgeLengths"], dtype=np.float64))
        self.setDelta(          float(data["delta"]))
        self.setMaxEdgeLength(  float(data["maxEdgeLength"]))
        self.setStiffness(      float(data["stiffness"]))
        self.setCompressibility(float(data["compressibility"]))
        self.initializeNeighborCells()

    def minimizeSoftBody(self):
        initialModelType = self.getModelEnum()
        self.setStiffness(1)
        self.setCompressibility(1)
        self.updateForceEnergy()
        self.setModelEnum("softBody")
        self.minimizeFIRE(dt = 0.1, maxForceThreshold = 1e-14)
        self.resetAreas()
        self.updatePolygonGeometry()
        self.setModelEnum(initialModelType)

    def makeSubModel(self, sub):
        pos0 = self.getPositions()
        nArray = self.getnArray()
        nArraySub = nArray[sub]
        totalVertices = np.sum(nArraySub)
        positions = np.zeros(totalVertices * 2)
        curr = 0
        startIndices = self.getStartIndices()
        for i in range(len(sub)):
            ind1 = curr * 2
            ind2 = curr * 2 + nArray[sub[i]] * 2
            ind3 = startIndices[sub[i]] * 2
            ind4 = startIndices[sub[i] + 1] * 2
            positions[ind1 : ind2] = pos0[ind3 : ind4]
            curr += nArray[sub[i]]
        nArray = self.getnArray()[sub]
        self.setNumVertices(positions.size // 2)
        self.setPositions(positions)
        self.setnArray(nArray)
