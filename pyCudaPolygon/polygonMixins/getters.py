import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import circulant
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import functools
import threading
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import math

class Mixin():

    # getters

    def getModelEnum(self):
        return self._impl.getModelEnum()

    def getRandomSeed(self):
        return self._impl.getRandomSeed()

    def getNumVertices(self):
        return int(self._impl.getNumVertices())

    def getNumPolygons(self):
        return self._impl.getNumPolygons()

    def getShapeId(self):
        return np.array(self._impl.getShapeId())

    def getVertices(self):
        return np.array(self._impl.getVertices())

    def getAreaPerOverlap(self):
        return np.array(self._impl.getAreaPerOverlap())

    def getIntersectionsCounter(self):
        return np.array(self._impl.getIntersectionsCounter())

    def getNumNeighbors(self):
        return np.array(self._impl.getNumNeighbors())

    def getNeighbors(self):
        v = self.getNumVertices()
        neighbors = np.array(self._impl.getNeighbors())
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
        return np.array(self._impl.getInsideFlag())

    def getTU(self):
        #v = self.getNumVertices()
        return np.array(self._impl.getTU())

    def getUT(self):
        #v = self.getNumVertices()
        return np.array(self._impl.getUT())

    def getIntersections(self):
        return np.array(self._impl.getIntersections())

    def getKeys(self):
        return np.array(self._impl.getKeys())

    def getOutersections(self):
        return np.array(self._impl.getOutersections())

    def getNumIntersections(self):
        return self._impl.getNumIntersections()

    def getNeighborCells(self):
        return np.array(self._impl.getNeighborCells())

    def getBoxCounts(self):
        return np.array(self._impl.getBoxCounts())

    def getNeighborIndices(self):
        return np.array(self._impl.getNeighborIndices())

    def getStartIndices(self):
        return self._impl.getStartIndices()

    def getnArray(self):
        return np.diff(self.getStartIndices())

    def getEnergy(self):
        return self._impl.getEnergy()

    def getForces(self):
        return np.array(self._impl.getForces())


    def getCentersOfMass(self):
        vertices = self.getVertices()
        nArray = self.getnArray()
        # Loop over n
        startIndices = np.concatenate((np.array([0]), np.cumsum(nArray)))
        csom = []
        for n, s in zip(nArray, startIndices):
            polygonPos = vertices[2 * s : 2 * s + 2 * n].reshape(n, 2) + 0
            polygonPos -= vertices[2 * s: 2 * s + 2]
            polygonPos += 1.5
            polygonPos %= 1.0
            polygonPos -= 0.5
            csom.append(np.mean(polygonPos, axis = 0) + vertices[2 * s : 2 * s + 2])
        return np.concatenate(csom)

    def getShapeCounts(self):
        return np.array(self._impl.getShapeCounts())

    def getIntersectionsCounter(self):
        intersectionsCounter = np.array(self._impl.getIntersectionsCounter())
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
        rmsEdgeViolation = np.sqrt(np.mean((1 - edgeLengths / targetEdgeLengths)**2))
        return np.array([rmsAreaViolation, rmsEdgeViolation, rmsPerimeterViolation])

    def getConstraints(self):
        return np.array(self._impl.getConstraints())

    def getAreas(self):
        # This is MC for now
        return np.array(self._impl.getAreas())

    def getTargetEdgeLengths(self):
        return np.array(self._impl.getTargetEdgeLengths())

    def getTargetAreas(self):
        return np.array(self._impl.getTargetAreas())

    def getEdgeLengths(self):
        return np.array(self._impl.getEdgeLengths())

    def getCOM(self):
        return np.array(self._impl.getCOM())

    def getPhi(self):
        return np.sum(self.getAreas())

    def getRestEdgeLengths(self):
        return np.array(self._impl.getEdgeLengths())

    def getMaxUnbalancedForce(self):
        return self._impl.getMaxUnbalancedForce()

    def getoverlapAreasGOLD(self):
        return self._impl.getoverlapAreasGOLD()

    def getoverlapAreas(self):
        return self._impl.getOverlapAreas()

    def getConstrainedForcePython(self, force):
        # We want to get the constrained force using all
        # of the constraints (one per edge per polygon)
        startIndices = self.getStartIndices()
        nArray = self.getnArray()
        vertices = self.getVertices()
        numPolygons = self.getNumPolygons()
        constrainedForce = force.copy()
        for polygon in range(numPolygons):
            start = startIndices[polygon]
            n = nArray[polygon]
            end = start + n
            pos = vertices[2 * start : 2 * end]
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

    def getMaxEdgeLength(self):
        return self._impl.getMaxEdgeLength()