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
from .. import enums

class Mixin():

    # setters

    def setStiffness(self, stiffness):
        self._impl.setStiffness(stiffness)

    def setModelEnum(self, modelType):
        if modelType == "abnormal":
            self._impl.setModelEnum(enums.modelEnum.abnormal)
        elif modelType == "edgeOnly":
            self._impl.setModelEnum(enums.modelEnum.edgeOnly)
        elif modelType == "areaOnly":
            self._impl.setModelEnum(enums.modelEnum.areaOnly)
        elif modelType == "softBody":
            self._impl.setModelEnum(enums.modelEnum.softBody)
        elif modelType == "normal":
            self._impl.setModelEnum(enums.modelEnum.normal)
        elif modelType == "hybrid":
            self._impl.setModelEnum(enums.modelEnum.hybrid)
        else:
            raise Exception("That Model type does not exist")

    def setNumVertices(self, n):
        self.size = n
        self._impl.setNumVertices(n)

    def setVertices(self, vertices):
        self._impl.setVertices(vertices.reshape(np.prod(vertices.shape)))

    def setnArray(self, nArray):
        nArray = nArray.astype(np.int32)
        if nArray.sum() != self.getNumVertices():
            raise ValueError(f"Total vertices from nArray ({nArray.sum()}) "
                            f"does not match model size ({self.getNumVertices()})")
        startIndices = np.concatenate(([0], np.cumsum(nArray)))
        self._impl.setStartIndices(startIndices)

    def setAreas(self, targetAreas):
        nArray = self.getnArray()
        self.updatePolygonGeometry()
        areas = self.getAreas()
        vertices = self.getVertices().copy().reshape(self.getNumVertices(), 2)

        start = 0
        for i, n in enumerate(nArray):
            s = start
            poly = vertices[s:s + n].copy()   # shape (n,2)

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

            vertices[s:s + n] = poly
            start += n

        # write back flattened vertices and update areas
        self.setVertices(vertices.reshape(self.getNumVertices() * 2))
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

    def setMaxEdgeLength(self, maxEdgeLength = 0.5):
        if maxEdgeLength is None:
            maxEdgeLength = 0
            nArray = self.getnArray()
            numShapes = len(nArray)
            startIndices = self.getStartIndices()
            vertices = self.getVertices()
            for i in range(numShapes):
                pos = vertices[2 * startIndices[i]:2 * startIndices[i + 1]].reshape(nArray[i], 2)
                diff = np.diff(np.concatenate((pos, [pos[0]])), axis = 0)
                diff += 1.5
                diff %= 1
                diff -= 0.5
                length = np.max(np.sqrt(np.sum(diff**2, axis = 1)))
                maxEdgeLength = np.max([maxEdgeLength, length])
        self._impl.setMaxEdgeLength(maxEdgeLength)

    def setBiPerimeters(self, kappa, ratio = 1.4):
        nArray = self.getnArray()
        numPolygons = len(nArray)
        numVertices = self.getNumVertices()
        self.setMaxEdgeLength()
        self.initializeNeighborCells()
        self.updateNeighborCells()
        self.updateNeighbors()
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
        targetEdgeLengths = np.repeat(kappa * np.sqrt(targetAreas) / nArray, nArray)
        self.setTargetEdgeLengths(targetEdgeLengths)
        self.updatePolygonGeometry()

    def setTargetEdgeLengths(self, targetEdgeLengths):
        self._impl.setTargetEdgeLengths(targetEdgeLengths)

    def setTargetAreas(self, targetAreas):
        self._impl.setTargetAreas(targetAreas)

    def setStiffness(self, stiffness):
        self._impl.setStiffness(stiffness)

    def setCompressibility(self, compressibility):
        self._impl.setCompressibility(compressibility)

    def getStiffness(self):
        return self._impl.getStiffness()

    def getCompressibility(self):
        return self._impl.getCompressibility()

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

    def setRandomPolygons(self, N, n, kappa):
        # First get the number of polygons
        self.setNumVertices(N * n)
        self.setnArray(np.repeat(n, N).astype(int))
        self.initializeNeighborCells()
        vertices = self.rng.random(N * n * 2).reshape(N, n, 2)
        for i in range(N):
            theta = (np.arctan2((vertices[i, :, 1] - np.mean(vertices[i, :, 1])), (vertices[i, :, 0] - np.mean(vertices[i, :, 0]))) + 2 * np.pi) % (2 * np.pi)
            vertices[i] = vertices[i][np.argsort(theta)]
        # Now we shrink all of the vertices to the bottom left box.
        # We don't really have a specific way of doing this and there's a freedom here
        # Let's keep the perimeter fixed and put a spring on the edges and area
        # To fix the periodic box, scale down by a factor of 4
        vertices /= 8
        nArray = np.array([n] * N)
        # Get the perimeters of each polygon
        diff = np.roll(vertices, 1, axis = 1) - vertices
        edgeLengths = np.sqrt(np.sum(diff**2, axis = 2))
        targetEdgeLengths = np.repeat(np.mean(edgeLengths, axis = 1), n)
        self.setTargetEdgeLengths(targetEdgeLengths)
        self.setTargetAreas((np.sum(targetEdgeLengths.reshape(N, n), axis = 1) / kappa)**2)
        scatter = np.swapaxes(np.repeat(self.rng.random(N * 2), n).reshape(N, 2, n), 1, 2)
        self.setVertices(vertices.reshape(N, n, 2) + scatter)
        # Great, now let's save the model type and make it soft body
        self.setStiffness(1)
        self.setCompressibility(1)
        modelType = self.getModelEnum()
        self.setModelEnum("softBody")
        self.minimizeFIRE()
        self.setModelEnum(modelType)

    def setForces(self, forces):
        self._impl.setForces(forces)