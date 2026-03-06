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

    def __init__(self, size = 0, seed = None, modelType = "normal"):
        lpcp.Model.__init__(self, size)
        self.setModelEnum(modelType)
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

    def setModelEnum(self, modelType):
        if modelType == "abnormal":
            lpcp.Model.setModelEnum(self, enums.modelEnum.abnormal)
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
        self.updateAreas()
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
        self.updateAreas()
        areas = self.getAreas()

    def setMonoArea(self, phi = 1):
        # This overrides phi!
        targetArea = phi / self.getNumPolygons()
        self.updateAreas()
        areas = self.getAreas()
        # Let's keep phi the same
        n = self.getNumPolygons()
        targetAreas = np.ones(n) * phi / n
        if (np.max(targetAreas) > 1 / 9):
            raise Exception("The phi you have chosen has caused the shapes to be too large and compromised the PBCs")
        self.setAreas(targetAreas)

    def setPhi(self, phi):
        self.updateAreas()
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

    def getContacts(self):
        v = self.getNumVertices()
        contacts = np.array(lpcp.Model.getContacts(self))
        maxNeighbors = len(contacts) // v
        contacts = contacts.reshape(v, maxNeighbors)
        cont = dict()
        numContacts = self.getNumContacts()
        for i, contact in enumerate(contacts):
            allContacts = contact[:numContacts[i]]
            if (len(allContacts) == 0):
                continue
            cont[i] = allContacts
        return cont

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

    def getOverlapArea(self):
        return np.array(lpcp.Model.getOverlapArea(self))

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
        dg, _ = np.linalg.qr(dg.T)
        return dg.T

    def getConstrainedForce(self, force):
        Q = self.getConstraintMatrix()
        numVertices = self.getNumVertices()
        Q2 = Q @ Q.T
        proj = np.identity(numVertices * 2) - np.dot(Q.T, Q)
        return np.dot(proj, force)

    def getAreas(self):
        # This is MC for now
        return np.array(lpcp.Model.getAreas(self))

    def getEdgeLengths(self):
        # TODO: do this with CUDA
        numPolygons = self.getNumPolygons()
        numVertices = self.getNumVertices()
        positions = self.getPositions()
        nArray = self.getnArray()
        startIndices = np.cumsum(np.concatenate(([0], nArray)))
        endIndices = np.roll(startIndices, -1)
        endIndices[-1] = numVertices
        sol = []
        for i in range(numPolygons):
            startIndex = startIndices[i]
            endIndex = endIndices[i]
            for j in range(startIndex, endIndex - 1):
                nextj = j + 1
                deltaVec = positions[j * 2 : (j + 1) * 2] - positions[nextj * 2 : (nextj + 1) * 2] + 1.5
                deltaVec %= 1
                deltaVec -= 0.5
                delta2 = (deltaVec)**2
                delta = np.sqrt(np.sum(delta2))
                sol.append(delta)
            nextj = endIndex - 1
            delta2 = (positions[nextj * 2 : nextj * 2 + 2] - positions[startIndex * 2 : (startIndex + 1) * 2])**2
            delta = np.sqrt(np.sum(delta2))
            sol.append(delta)
        return np.array(sol)

    # updaters

    def updateNeighbors(self, a = 0):
        lpcp.Model.updateNeighbors(self, a)

    def updateContacts(self):
        lpcp.Model.updateContacts(self)

    def updateValidAndCounts(self):
        return lpcp.Model.updateValidAndCounts(self)

    def updateCompactedIntersections(self):
        lpcp.Model.updateCompactedIntersections(self)

    def updateOverlapArea(self, pointDensity):
        # This is MC
        lpcp.Model.updateOverlapArea(self, pointDensity)

    def updateAreas(self):
        # This is MC
        lpcp.Model.updateAreas(self)

    def updateNeighborCells(self):
        lpcp.Model.updateNeighborCells(self)
        
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

    def minimizeGDStep(self, h = None, g12 = None, a = 0, lam = 0, pref = 1, dt = 1e-3, addedForce = None):
        #self.initializeNeighborCells()
        self.updateNeighborCells()
        self.updateNeighbors(a)
        self.updateContacts()
        self.updateOutersections()
        try:
            overlapArea, force = self.functional(h = h, g12 = g12, lam = lam, pref = pref)
        except:
            raise Exception("Something went wrong with the functional")
        if (addedForce is not None):
            force += addedForce
        force = self.getConstrainedForce(force)
        positions = self.getPositions()
        positions += dt * force
        positions += 1.0
        positions %= 1.0
        self.setPositions(positions)
        return overlapArea

    def minimizeGD(self, h = None, g12 = None, lam = 0, pref = 1, dt = 1e-3, maxSteps = 100, addedForce = None, progressBar = False, checkpointDir = None, checkpointFreq = 1):
        self.initializeNeighborCells()
        with tqdm(total = maxSteps, desc="Processing", disable=(not progressBar)) as pbar:
            for step in range(maxSteps):
                if checkpointDir is not None and (step % checkpointFreq == 0):
                    if not os.path.isdir(checkpointDir):
                        os.makedirs(checkpointDir)
                    self.saveModel(checkpointDir + "/" + str(step))
                if progressBar:
                    pbar.update(1)
                if (self.minimizeGDStep(h = h, g12 = g12, lam = lam, pref = pref, addedForce = addedForce, dt = dt) == 0):
                    return 0

    def functionalExterior(self, h, g12 = None, lam = 0, pref = 1):
        numVertices = self.getNumVertices()
        intersections = self.getIntersections()
        newTU = self.getTU()
        newUT = self.getUT()
        if (len(intersections) == 0):
            return 0, np.zeros(numVertices * 2)
        outersections = self.getOutersections()
        positions = self.getPositions()
        x = positions[::2]
        y = positions[1::2]
        sol = 0
        shapeId = self.getShapeId()
        nArray = self.getnArray()
        startIndices = self.getStartIndices()
        grad = np.zeros(numVertices * 2)
        for index in range(len(outersections)):
            # s(j), s(i), i, j
            startID1 = startIndices[(intersections[index] >> 32) & 0xFFFF]
            startID2 = startIndices[(intersections[index] >> 48) & 0xFFFF]
            startID = np.min([startID1, startID2])
            startPoint = positions[2 * startID : 2 * startID + 2]
            i = ((intersections[index] >> 16) & 0xFFFF) + startID1
            j = ((intersections[index]) & 0xFFFF) + startID2
            k = ((outersections[index] >> 16) & 0xFFFF) + startID2
            l = ((outersections[index]) & 0xFFFF) + startID1
            zi = self.z(i)
            zj = self.z(j)
            zk = self.z(k)
            zl = self.z(l)
            pi = positions[2 * i: 2 * i + 2]
            pj = positions[2 * j: 2 * j + 2]
            pk = positions[2 * k: 2 * k + 2]
            pl = positions[2 * l: 2 * l + 2]
            pzi = positions[2 * zi: 2 * zi + 2]
            pzj = positions[2 * zj: 2 * zj + 2]
            pzk = positions[2 * zk: 2 * zk + 2]
            pzl = positions[2 * zl: 2 * zl + 2]
            r1 = pzi - pi + 1.5
            r2 = pzk - pk + 1.5
            r1 %= 1
            r1 -= 0.5
            r2 %= 1
            r2 -= 0.5
            t1 = newTU[2 * index + 1]
            fij = (pi + t1 * r1 + 1) % 1
            t2 = newUT[2 * index + 1]
            fkl = (pk + t2 * r2 + 1) % 1
            if (i == l):
                sol += h(fij, fkl, startPoint, lam, pref)
                if (g12 is not None):
                    dfij = self.getDf(pi, pzi, pj, pzj)  # i<->j
                    dfki = self.getDf(pk, pzk, pi, pzi)  # k<->i
                    g1, g2 = g12(fij, fkl, startPoint, lam, pref)
                    for alpha in range(2):
                        # i and zi contributions
                        grad[i * 2 + alpha]  += np.dot(g1, dfij[:, 0 + alpha]) + np.dot(g2, dfki[:, 4 + alpha])
                        grad[zi * 2 + alpha] += np.dot(g1, dfij[:, 2 + alpha]) + np.dot(g2, dfki[:, 6 + alpha])
                        
                        # j and zj contributions
                        grad[j * 2 + alpha]  += np.dot(g1, dfij[:, 4 + alpha])
                        grad[zj * 2 + alpha] += np.dot(g1, dfij[:, 6 + alpha])
                        
                        # k and zk contributions
                        grad[k * 2 + alpha]  += np.dot(g2, dfki[:, 0 + alpha])
                        grad[zk * 2 + alpha] += np.dot(g2, dfki[:, 2 + alpha])
            else:
                sol += h(fij, positions[self.z(i) * 2: self.z(i) * 2 + 2], startPoint, lam, pref)
                sol += h(positions[l * 2: l * 2 + 2], fkl, startPoint, lam, pref)
                if (g12 is not None):
                    g1, g2 = g12(fij, pzi, startPoint, lam, pref)
                    dfij = self.getDf(pi, pzi, pj, pzj)  # i<->j
                    dfkl = self.getDf(pk, pzk, pl, pzl)  # k<->i
                    for alpha in range(2):
                        grad[i * 2 + alpha] += np.dot(g1, dfij[:, 0 + alpha])
                        grad[j * 2 + alpha] += np.dot(g1, dfij[:, 4 + alpha])
                        grad[zi * 2 + alpha] += np.dot(g1, dfij[:, 2 + alpha]) + g2[alpha]
                        grad[zj * 2 + alpha] += np.dot(g1, dfij[:, 6 + alpha])
                    g1, g2 = g12(pl, fkl, startPoint, lam, pref)
                    for alpha in range(2):
                        grad[l * 2 + alpha] += g1[alpha] + np.dot(g2, dfkl[:, 4 + alpha])
                        grad[k * 2 + alpha] += np.dot(g2, dfkl[:, 0 + alpha])
                        grad[zl * 2 + alpha] += np.dot(g2, dfkl[:, 6 + alpha])
                        grad[zk * 2 + alpha] += np.dot(g2, dfkl[:, 2 + alpha])
        return sol, -grad

    def functionalInterior(self, h, g12 = None, lam = 0, pref = 1):
        numVertices = self.getNumVertices()
        outersections = self.getOutersections()
        shapeId = self.getShapeId()
        intersections = self.getIntersections()
        numIntersections = len(intersections)
        nArray = self.getnArray()
        startIndices = self.getStartIndices()
        positions = self.getPositions()
        sol = 0
        grad = np.zeros(numVertices * 2)
        mask = (1 << 48) - 1
        for m in range(numVertices):                
            s = shapeId[m]
            n = nArray[s]
            start = 0
            end = numIntersections
            lb = (s << 32)
            ub = ((s + 1) << 32)
            while (end > start):
                mid = (end + start) // 2
                if (intersections[mid] & mask < lb): 
                    start = mid + 1
                else:
                    end = mid
            index = start 
            while (index < numIntersections and intersections[index] & mask < ub):
                i = ((intersections[index] >> 16) & 0xFFFF) + startIndices[s]
                l = ((outersections[index]) & 0xFFFF) + startIndices[s]
                sj = (intersections[index] >> 48) & 0xFFFF
                startID = startIndices[s]
                startPointID = min(startID, startIndices[sj])
                # The l + n - i works fine, but the mDist doesn't when m and i aren't
                # in the same shape. 
                mDist = (m + n - i) % n
                lDist = (l + n - i) % n
                if (mDist == 0 or mDist == lDist):
                    index += 1
                    continue
                if (mDist < lDist):
                    # This intersection does matter
                    startPoint = positions[2 * startPointID : 2 * startPointID + 2]
                    nextIndex = self.z(m)
                    p1 = positions[m * 2: m * 2 + 2]
                    p2 = positions[nextIndex * 2: nextIndex * 2 + 2]
                    sol += h(p1, p2, startPoint, lam, pref)
                    if (not (g12 is None)):
                        g1, g2 = g12(p1, p2, startPoint, lam, pref)
                        grad[m * 2] += g1[0]
                        grad[m * 2 + 1] += g1[1]
                        grad[nextIndex * 2] += g2[0]
                        grad[nextIndex * 2 + 1] += g2[1]
                index += 1
        return sol, -grad

    def functional(self, h, g12 = None, lam = 0, pref = 1):
        eef = self.functionalExterior(h, g12 = g12, lam = lam, pref = pref)
        try:
            exterior, exteriorForce = eef
        except TypeError:
            print("eef = ", eef)
        interior, interiorForce = self.functionalInterior(h, g12 = g12, lam = lam, pref = pref)
        #return interior, interiorForce
        if (lam == 0 and exterior + interior < 0):
            raise Exception("negative energy found!")
        #return interior, interiorForce
        return exterior + interior, exteriorForce + interiorForce

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
        self.setnArray(state[:-self.getNumVertices() * 2].astype(int))
        self.setPositions(state[-self.getNumVertices() * 2:])

