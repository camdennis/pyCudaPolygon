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

    def __init__(self, size = 0, seed = None, modelType = "normal", stiffness = 0):
        lpcp.Model.__init__(self, size)
        self.setModelEnum(modelType)
        self.setStiffness(stiffness)
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

    # setters

    def setStiffness(self, stiffness):
        lpcp.Model.setStiffness(self, stiffness)

    def setModelEnum(self, modelType):
        if modelType == "abnormal":
            lpcp.Model.setModelEnum(self, enums.modelEnum.abnormal)
        elif modelType == "edgeOnly":
            lpcp.Model.setModelEnum(self, enums.modelEnum.edgeOnly)
        elif modelType == "normal":
            lpcp.Model.setModelEnum(self, enums.modelEnum.normal)
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
        lpcp.Model.setMaxEdgeLength(self, maxEdgeLength)

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
        self.resetAreas()
        targetEdgeLengths = kappa * np.sqrt(targetAreas) / nArray
        targetEdgeLengths = np.repeat(targetEdgeLengths, nArray)
        self.setTargetEdgeLengths(targetEdgeLengths)
        self.updatePolygonGeometry()

    def setTargetEdgeLengths(self, targetEdgeLengths):
        lpcp.Model.setTargetEdgeLengths(self, targetEdgeLengths)

    def setTargetAreas(self, targetAreas):
        lpcp.Model.setTargetAreas(self, targetAreas)

    def setStiffness(self, stiffness):
        lpcp.Model.setStiffness(self, stiffness)

    def setPhi(self, phi):
        self.updatePolygonGeometry()
        targetAreas = self.getAreas()
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

    def getForces(self):
        return np.array(lpcp.Model.getForces(self))

    def getConstraints(self):
        return np.array(lpcp.Model.getConstraints(self))

    def getConstraintForces(self):
        return np.array(lpcp.Model.getConstraintForces(self))

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

    def getProj(self):
        return np.array(lpcp.Model.getProjection(self))

    def getConstrainedForceTEST(self, force):
        Q = self.getConstraintMatrixTEST()
        numVertices = self.getNumVertices()
        proj = np.identity(numVertices * 2) - np.dot(Q.T, Q)
        return np.dot(proj, force)

    def MGS(self, constraintMatrix):
        def normalizeBySpecies(v, speciesMap, numSpecies):
            newV = v.copy()
            sp2 = np.zeros(numSpecies)
            for k in range(v.size):
                sp2[speciesMap[k // 2]] += v[k]**2
            for k in range(v.size):
                newV[k] = v[k] / np.sqrt(sp2[speciesMap[k // 2]])
            return newV

        def innerProd(v, u, speciesMap, numSpecies):
            ip = np.zeros(numSpecies)
            for k in range(v.size):
                ip[speciesMap[k // 2]] += v[k] * u[k]
            return ip

        def projection(v, u, ip, speciesMap):
            for k in range(v.size):
                v[k] = v[k] - u[k] * ip[speciesMap[k // 2]]
            return v

        speciesMap = self.getSpeciesMap()
        numSpecies = np.max(speciesMap) + 1
        g = constraintMatrix
        for i in range(0, 4):
            g[i] = normalizeBySpecies(g[i], speciesMap, numSpecies)
            for j in range(i + 1, 4):
                ip = innerProd(g[j], g[i], speciesMap, numSpecies)
                g[j] = projection(g[j], g[i], ip, speciesMap)
        return g

    def getConstraintMatrix(self):
        shapeId = self.getShapeId()
        numVertices = self.getNumVertices()
        gl = np.zeros(2 * numVertices)
        gl2 = np.zeros(2 * numVertices)
        positions = self.getPositions()
        nArray = self.getnArray()
        ap = np.zeros(nArray.size)
        da = np.zeros(2 * numVertices)
        startIndices = self.getStartIndices()
        for j in range(numVertices):
            dp1 = positions[j * 2 : j * 2 + 2] - positions[self.zp(j) * 2 : self.zp(j) * 2 + 2]
            dp2 = positions[self.z(j) * 2 : self.z(j) * 2 + 2] - positions[j * 2 : j * 2 + 2]
            dp3 = positions[self.z(j) * 2 : self.z(j) * 2 + 2] - positions[self.zp(j) * 2 : self.zp(j) * 2 + 2]
            dp1 += 1.5
            dp2 += 1.5
            dp3 += 1.5
            dp1 %= 1
            dp2 %= 1
            dp3 %= 1
            dp1 -= 0.5
            dp2 -= 0.5
            dp3 -= 0.5
            denom1 = np.sqrt(np.sum(dp1**2))
            denom2 = np.sqrt(np.sum(dp2**2))
            gl[j * 2 : j * 2 + 2] = dp1 / denom1 - dp2 / denom2
            gl2[j * 2 : j * 2 + 2] = 2 * (dp1 - dp2)
            da[j * 2 : j * 2 + 2] = (np.arange(2) - 0.5) * dp3[::-1]
            startID = startIndices[shapeId[j]]
            startPoint = positions[2 * startID : 2 * startID + 2]
            dpp1 = positions[2 * self.z(j) : 2 * self.z(j) + 2] - startPoint
            dpp2 = positions[2 * j : 2 * j + 2] - startPoint
            dpp1 += 1.5
            dpp2 += 1.5
            dpp1 %= 1
            dpp2 %= 1
            dpp1 -= 0.5
            dpp2 -= 0.5
            ap[shapeId[j]] += (dpp1 + dpp2)[1] * dp2[0] / 2
        #print("ap = ", -ap)
        ap = np.repeat(ap, nArray)
        startPoints = np.concatenate((np.array([0]), np.cumsum(nArray)))
        lVals = np.zeros(numVertices * 2)
        da2 = np.zeros(2 * numVertices)
        for j in range(numVertices):
            dp3 = positions[self.z(j) * 2 : self.z(j) * 2 + 2] - positions[self.zp(j) * 2 : self.zp(j) * 2 + 2]
            dp3 += 1.5
            dp3 %= 1
            dp3 -= 0.5
            da2[j * 2 : j * 2 +  2] = (np.arange(2) - 0.5) * dp3[::-1] * ap[j]
        # Now we orthonormalize
        dg = np.array(np.vstack((gl, gl2, da, da2)))
        Q, _ = np.linalg.qr(dg.T)
        return Q.T
        return self.MGS(dg)

    def getAreas(self):
        # This is MC for now
        return np.array(lpcp.Model.getAreas(self))

    def getTargetEdgeLengths(self):
        return np.array(lpcp.Model.getTargetEdgeLengths(self))

    def getCOM(self):
        return np.array(lpcp.Model.getCOM(self))

    def getPhi(self):
        return np.sum(self.getAreas())

    def getRestEdgeLengths(self):
        return np.array(lpcp.Model.getEdgeLengths(self))

    def getMaxUnbalancedForce(self):
        return lpcp.Model.getMaxUnbalancedForce(self)

    def getOverlapArea(self):
        return lpcp.Model.getOverlapArea(self)

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

    def updateNeighborCells(self):
        lpcp.Model.updateNeighborCells(self)
        
    def updateForceEnergy(self):
        lpcp.Model.updateForceEnergy(self)

    def updatePositions(self, dt):
        lpcp.Model.updatePositions(self, dt)

    def updateConstraintForces(self):
        lpcp.Model.updateConstraintForces(self)

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

    def getNorm2(self):
        return np.array(lpcp.Model.getNorm2(self))

    def minimizeGDStep(self, a = 0.01, dt = 1e-3, addedForce = None):
        try:
            self.updateNeighborCells()
            self.updateNeighbors()
            self.updateOutersections()
            self.updateForceEnergy()
            self.updateConstraintForces()
            self.updatePolygonGeometry()
            if (self.getMaxUnbalancedForce() > 10):
                print("here")
                return 0 
            self.updatePositions(dt)
            self.updatePolygonGeometry()
            self.resetAreas()
            self.updatePolygonGeometry()
            #overlapArea = self.getEnergy()
            #force = self.getForces()
        except:
            raise Exception("Something went wrong with updating the force and energy")
        #positions = self.getPositions()
        #positions += dt * force
        #positions += 1.0
        #positions %= 1.0
        #self.setPositions(positions)
#        print(self.getEnergy(), self.getMaxUnbalancedForce())
        return self.getEnergy()

    def minimizeGD(self, a = 0.01, dt = 1e-3, maxSteps = 100, addedForce = None, progressBar = False, checkpointDir = None, checkpointFreq = 1, overwriteCheckpoint = False):
        with tqdm(total = maxSteps, desc = "Processing", disable = (not progressBar)) as pbar:
            for step in range(maxSteps):
                if checkpointDir is not None and (step % checkpointFreq == 0):
                    if not os.path.isdir(checkpointDir):
                        os.makedirs(checkpointDir)
                    self.saveModel(checkpointDir + "/" + str(step), overwrite = overwriteCheckpoint)
                if progressBar:
                    pbar.update(1)
                if (self.minimizeGDStep(addedForce = addedForce, dt = dt) == 0):
                    return 0

    def saveModel(self, dirName, overwrite = False):
        if not overwrite and os.path.isdir(dirName):
            raise Exception("Packing exists. Not saving. To save over this file, set kwarg overwrite = True")
        if not os.path.isdir(dirName):
            os.mkdir(dirName)
        # Get stuff to save:
        modelEnum = self.getModelEnum()
        numVertices = self.getNumVertices()
        nArray = np.diff(self.getStartIndices())
        positions = self.getPositions()
        maxEdgeLength = self.getMaxEdgeLength()
        kv = dict()
        kv["modelEnum"] = str(modelEnum)
        kv["numVertices"] = numVertices
        kv["maxEdgeLength"] = maxEdgeLength
        saveFile = dirName + "/scalars.dat"
        with open(saveFile, 'w') as f:
            for k in kv.keys():
                f.write(k + ":\t" + str(kv[k]) + "\n")
        f.close()
        # We can save the nArray and positions back to back in one
        # file since splitting it up is easy with the scalars.dat file
        state = np.concatenate((nArray, positions))
        np.save(dirName + "/state", state)
        
    def loadModel(self, dirName):
        state = np.load(dirName + "/state.npy")
        scalarsFile = dirName + "/scalars.dat"
        with open(scalarsFile, 'r') as f:
            lines = f.readlines()
        f.close()
        modelType = "normal"
        numVertices = 0
        maxEdgeLength = 0.5
        for line in lines:
            k, v = line.split("\n")[0].split("\t")
            if (k == "modelEnum:"):
                modelType = v
            elif (k == "numVertices:"):
                numVertices = int(v)
            elif (k == "maxEdgeLength:"):
                maxEdgeLength = float(v)
        try:
            self.setModelEnum(modelType)
        except Exception:
            print("Warning: model enum not found. Setting to normal")
            self.setModelEnum("normal")
        self.setNumVertices(numVertices)
        nArray = state[:-self.getNumVertices() * 2].astype(int).copy()
        self.setnArray(nArray)
        positions = state[-self.getNumVertices() * 2:].copy()
        self.setPositions(positions)

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
