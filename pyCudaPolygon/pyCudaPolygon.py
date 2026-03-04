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
    def __init__(self, 
                 size = 0,
                 seed = None,
                 modelType = "normal",
    ):
        lpcp.Model.__init__(self, size)
        self.setModelEnum(modelType)
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
#            lpcp.Model.initializeRandomSeed(self, seed)
    
    def setModelEnum(self, modelType):
        if modelType == "abnormal":
            lpcp.Model.setModelEnum(self, enums.modelEnum.abnormal)
        elif modelType == "normal":
            lpcp.Model.setModelEnum(self, enums.modelEnum.normal)
        else:
            raise Exception("That Model type does not exist")
            
    def getModelEnum(self):
        return lpcp.Model.getModelEnum(self)

    def getRandomSeed(self):
        return lpcp.Model.getRandomSeed(self)

    def setNumVertices(self, n):
        self.size = n
        lpcp.Model.setNumVertices(self, n)

    def getNumVertices(self):
        return int(lpcp.Model.getNumVertices(self))

    def getNumPolygons(self):
        return lpcp.Model.getNumPolygons(self)

    def initializeRandomSeed(self, seed):
        self.rng = np.random.default_rng(seed)
        lpcp.Model.initializeRandomSeed(self, seed)

    def setPositions(self, positions):
        lpcp.Model.setPositions(self, positions)

    def getShapeId(self):
        return np.array(lpcp.Model.getShapeId(self))

    def getPositions(self):
        return np.array(lpcp.Model.getPositions(self))

    def markGroupBoundaries(self):
        lpcp.Model.markGroupBoundaries(self)

    def getGroupStart(self):
        return np.array(lpcp.Model.getGroupStart(self))

    def getGroupLength(self):
        return np.array(lpcp.Model.getGroupLength(self))

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
#        v = self.getNumVertices()
        return np.array(lpcp.Model.getInsideFlag(self))
#        maxNeighbors = len(inside) // v
#        inside = inside.reshape(v, maxNeighbors)
#        insideDict = dict()
#        numNeighbors = self.getNumNeighbors()
#        for i, el in enumerate(inside):
#            allEl = el[:numNeighbors[i]]
#            if (len(allEl) == 0):
#                continue
#            insideDict[i] = allEl
#        return insideDict

    def getTU(self):
#        v = self.getNumVertices()
        return np.array(lpcp.Model.getTU(self))

    def getUT(self):
#        v = self.getNumVertices()
        return np.array(lpcp.Model.getUT(self))

    def updateNeighbors(self, a = 0):
        lpcp.Model.updateNeighbors(self, a)

    def updateContacts(self):
        lpcp.Model.updateContacts(self)

    def markValidAndCounts(self):
        return lpcp.Model.markValidAndCounts(self)

    def writeCompacted(self):
        lpcp.Model.writeCompacted(self)

    def sortKeys(self, endBit):
        lpcp.Model.sortKeys(self, endBit)

    def getIntersections(self):
        return np.array(lpcp.Model.getIntersections(self))

    def getKeys(self):
        return np.array(lpcp.Model.getKeys(self))

    def getOutersections(self):
        return np.array(lpcp.Model.getOutersections(self))

    def getNumIntersections(self):
        return lpcp.Model.getNumIntersections(self)

    def updateOverlapArea(self, pointDensity):
        # This is MC
        lpcp.Model.updateOverlapArea(self, pointDensity)

    def setnArray(self, nArray):
        nArray = nArray.astype(int)
        # we don't actually set the nArray, we set the startIndices:
        startIndices = np.concatenate(([0], np.cumsum(nArray)))
        lpcp.Model.setStartIndices(self, startIndices)

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

    def updateAreas(self):
        # This is MC
        lpcp.Model.updateAreas(self)

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

    def getAreaOfPos(self, pos):
        # This is pythonic for now
        x = pos[::2]
        y = pos[1::2]
        x = np.concatenate((x, [x[0]]))
        y = np.concatenate((y, [y[0]]))
        a = np.dot(x[:-1], y[1:])
        a -= np.dot(x[1:], y[:-1])
        a /= 2
        a = np.abs(a)
        return a

    def initializeNeighborCells(self):
        lpcp.Model.initializeNeighborCells(self)

    def updateNeighborCells(self):
        lpcp.Model.updateNeighborCells(self)

    def getNeighborCells(self):
        return np.array(lpcp.Model.getNeighborCells(self))

    def getBoxCounts(self):
        return np.array(lpcp.Model.getBoxCounts(self))

    def getNeighborIndices(self):
        return np.array(lpcp.Model.getNeighborIndices(self))

    def generateNaivePolygon(self, n):
        def convex_hull(points):
            pts = points[np.lexsort((points[:,1], points[:,0]))]
            def half(pts):
                h = []
                for p in pts:
                    while len(h) >= 2:
                        r, q = h[-2], h[-1]
                        if (q[0]-r[0])*(p[1]-r[1]) - (q[1]-r[1])*(p[0]-r[0]) <= 0:
                            h.pop()
                        else:
                            break
                    h.append(tuple(p))
                return h
            lower = half(pts)
            upper = half(pts[::-1])
            hull = np.array(lower[:-1] + upper[:-1])
            return hull
        pts = self.rng.random(max(8, n*3) * 2).reshape(max(8, n*3), 2) - 0.5           # oversample
        hull = convex_hull(pts)
        if len(hull) < n:
            # subdivide hull edges to reach n
            extra = []
            i = 0
            while len(hull) + len(extra) < n:
                a = hull[i % len(hull)]
                b = hull[(i+1) % len(hull)]
                t = self.rng.random()
                extra.append(a*(1-t) + b*t)
                i += 1
            hull = np.vstack([hull, np.array(extra)])
        hull = hull[:n]       # trim if needed
        return hull.reshape(n*2)

    def getStartIndices(self):
        return lpcp.Model.getStartIndices(self)

    def getnArray(self):
        return np.diff(self.getStartIndices())
        
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
        for line in lines:
            k, v = line.split("\n")[0].split("\t")
            if (k == "modelEnum"):
                self.setModelEnum(v)
            elif (k == "numVertices"):
                self.setNumVertices(v)
            elif (k == "maxEdgeLength"):
                self.setMaxEdgeLength(v)
        self.setnArray(state[:-self.getNumVertices() * 2].astype(int))
        self.setPositions(state[-self.getNumVertices() * 2:])

    def getForces(self):
        return np.array(lpcp.Model.getForces(self))

    def setRandomPolygons(self):
        # Generates random polygons with size similar to box size (so they're huge)
        pos = []
        for n in self.getnArray():
            pg = self.generateNaivePolygon(n) / 2
            # Make the polygons smaller so they don't have image problems
            # perturb the polygons so they're centered randomly
            pg[::2] += self.rng.random()
            pg[1::2] += self.rng.random()
            pg += 1
            pg %= 1
            pos.append(pg)
        pos = np.concatenate(pos)
        # Now they all lie in a much smaller box
        self.setPositions(pos)

    def setECPolygons(self, n):
        # EC stands for equally coordinated
        # Here n is the number of nodes per particle
        N = self.getNumVertices()
        if (N % n != 0):
            raise Exception("Total vertices and number of vertices per polygon are incompatible")
        self.setnArray(np.ones(N // n) * n)

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

    def whichShape(self, edge):
        startIndices = self.getStartIndices()
        shapeId = np.argmin(abs(startIndices - edge))
        if edge > startIndices[shapeId]:
            shapeId -= 1
        return shapeId

    def isRealNeighbor(self, e1, e2):
        startArray = self.getStartIndices()
        s1 = self.whichShape(e1)
        s2 = self.whichShape(e2)
        positions = self.getPositions()
        a1x = positions[e1 * 2]
        a1y = positions[e1 * 2 + 1]
        b1x = positions[e2 * 2]
        b1y = positions[e2 * 2 + 1]
        if s1 < self.whichShape(e1 + 1):
            v2 = startArray[s1]
        else:
            v2 = e1 + 1
        a2x = positions[v2 * 2]
        a2y = positions[v2 * 2 + 1]
        if s2 < self.whichShape(e2 + 1):
            v2 = startArray[s2]
        else:
            v2 = e2 + 1
        b2x = positions[v2 * 2]
        b2y = positions[v2 * 2 + 1]
        r = np.array([a2x - a1x, a2y - a1y])
        s = np.array([b2x - b1x, b2y - b1y])
        g = np.array([b1x - a1x, b1y - a1y])
        denom = np.linalg.det(r, s)
        u = np.linalg.det([g, r]) / denom
        t = np.linalg.det([g, s]) / denom
        if (u < 0 or u > 1 or t < 0 or t > 1):
            return False
        return True

    # We haven't found the neighbors properly yet. We still need to 
    # implement that, but it's pretty easy so we can put it off for now.
    # Let's assume we have all the edge intersections in "neighbors."
    # We want to create a list that contains the exits
    # and a list that contains the points that enter. This will
    # help to find the players.

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

    def getIntersectionsPY(self):
        def pack(numbers):
            result = (numbers[3] & 0xFFFF) | \
                    ((numbers[2] & 0xFFFF) << 16) | \
                    ((numbers[1] & 0xFFFF) << 32) | \
                    ((numbers[0] & 0xFFFF) << 48)
            return result

        def unpack(val):
            """Unpack a 64-bit integer back into four 16-bit numbers."""
            nums = np.zeros(4)
            nums[3] = val & 0xFFFF
            nums[2] = (val >> 16) & 0xFFFF
            nums[1] = (val >> 32) & 0xFFFF
            nums[0] = (val >> 48) & 0xFFFF
            return nums  

        def flipPack(val):
            a = (val >> 48) & 0xFFFF
            b = (val >> 32) & 0xFFFF
            return (a & 0xFFFF) | ((b & 0xFFFF) << 16)
        
        numVertices = self.getNumVertices()
        contacts = np.array(lpcp.Model.getContacts(self))
        maxNeighbors = len(contacts) // numVertices
        numContacts = self.getNumContacts()
        insideFlag = self.getInsideFlag()
        shapeIds = self.getShapeId()
        numShapes = self.getNumPolygons()
        intersections = []
        t, u = self.getTU().reshape(2, numVertices * maxNeighbors)
        newTU = []
        numEach = np.zeros(numShapes, dtype = int)
        for n1 in range(numVertices):
            for i in range(numContacts[n1]):
                s1 = shapeIds[n1]
                n2 = contacts[n1 * maxNeighbors + i]
                s2 = shapeIds[n2]
                tVal = t[n1 * maxNeighbors + i]
                uVal = u[n1 * maxNeighbors + i]
                if insideFlag[n1 * maxNeighbors + i]:
                    intersection = pack([s2, s1, n1, n2])
                    numEach[s2] += 1
                    newTU.append(uVal)
                    newTU.append(tVal)
                else:
                    intersection = pack([s1, s2, n2, n1])
                    numEach[s1] += 1
                    newTU.append(tVal)
                    newTU.append(uVal)
                intersections.append(intersection)
        ends = np.cumsum(numEach)
        starts = np.concatenate(([0], ends[:-1]))
        intersections = np.array(intersections)
        numIntersections = intersections.size
        newTU = np.array(newTU).reshape(numIntersections, 2)
        # Now we want to sort these!
        args = np.argsort(intersections)
        intersections = intersections[args]
        newTU = newTU[args].reshape(numIntersections * 2)
        # Now we want to find the start and end indices
        bizarroStarts = np.zeros(np.uint64(numVertices)  << np.uint64(16), dtype = int)
        bizarroEnds = np.zeros(np.uint64(numVertices)  << np.uint64(16), dtype = int)
        if (len(intersections) == 0):
            return intersections, newTU, starts, ends, bizarroStarts, bizarroEnds
        packedShape = intersections.astype(np.uint64) >> np.uint64(32)
        shapeStartFlag = np.concatenate(([True], np.diff(packedShape).astype(bool)))
        bizarroIntersection = flipPack(intersections[0])
        bizarroStarts[bizarroIntersection] = 0
        for i in range(1, len(intersections)):
            if shapeStartFlag[i]:
                bizarroIntersection = flipPack(intersections[i])
                # This intersection, if you search it should start here
                bizarroStarts[bizarroIntersection] = i
                # The intersection before this, if you search it should end before this
                bizarroIntersectionM1 = flipPack(intersections[i - 1])
                bizarroEnds[bizarroIntersectionM1] = i - 1
        # The last one ends as well
        bizarroIntersection = flipPack(intersections[-1])
        bizarroEnds[bizarroIntersection] = len(intersections) - 1
        return intersections, newTU, starts, ends, bizarroStarts, bizarroEnds
    
    def getPlayersPY(self):
        intersections = self.getIntersections()
        self.markGroupBoundaries()
        starts = self.getGroupStart()
        ends = starts + self.getGroupLength() - 1
        n = self.getnArray()
        numIntersections = len(intersections)
        players = np.empty(numIntersections, dtype=int)
        newTU = self.getTU()
        newTU = np.array(newTU).reshape(numIntersections, 2).T.reshape(2 * numIntersections)


        for i in range(numIntersections):
            this_i = intersections[i] & 0xFFFF
            vertex = (intersections[i] >> 48) & 0xFFFF
            ni = n[vertex]

            # Slice of neighbors
            start = starts[i]
            end = ends[i]

            tVal = newTU[2 * i]

            bestDist = None
            bestU = np.inf
            bestIdx = None
            fallbackU = np.inf
            fallbackIdx = None
            for k in range(start, end + 1):
                j = (intersections[k] >> 16) & 0xFFFF
                uVal = newTU[2 * k + 1]

                # Cyclic distance
                d = (j - this_i + ni) % ni

                # Conditional self-pairing: allow only if current u < candidate u
                if j == this_i and tVal >= uVal:
                    continue

                # New minimum distance found
                if bestDist is None or d < bestDist:
                    bestDist = d
                    bestU = np.inf
                    bestIdx = None
                    fallbackU = np.inf
                    fallbackIdx = None

                if d == bestDist:
                    # Prefer candidates with u >= tVal
                    if uVal >= tVal and uVal < bestU:
                        bestU = uVal
                        bestIdx = k
                    # Keep fallback for candidates violating TU
                    if uVal < fallbackU:
                        fallbackU = uVal
                        fallbackIdx = k

            # Choose the best candidate or fallback if none satisfy TU
            players[i] = bestIdx if bestIdx is not None else fallbackIdx

        return players

    def getOverlapArea(self):
        return np.array(lpcp.Model.getOverlapArea(self))

    def getShapeCounts(self):
        return np.array(lpcp.Model.getShapeCounts(self))

    def getIntersectionsCounter(self):
        intersectionsCounter = np.array(lpcp.Model.getIntersectionsCounter(self))
        s = int(np.sqrt(len(intersectionsCounter)))
        return intersectionsCounter.reshape(s, s)

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
#        return interior, interiorForce
        if (lam == 0 and exterior + interior < 0):
            raise Exception("negative energy found!")
#        return interior, interiorForce
        return exterior + interior, exteriorForce + interiorForce
    
    def minimizeGDStep(self, h = None, g12 = None, a = 0, lam = 0, pref = 1, dt = 1e-3, addedForce = None):
        self.initializeNeighborCells()
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
        self.setPositions(positions)
        return overlapArea

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

    def minimizeGD(self, h = None, g12 = None, lam = 0, pref = 1, dt = 1e-3, maxSteps = 100, addedForce = None, progressBar = False, checkpointDir = None, checkpointFreq = 1):
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
#        print("ap = ", -ap)
        ap = np.repeat(ap, nArray)
        startPoints = np.concatenate((np.array([0]), np.cumsum(nArray)))
        lVals = np.zeros(numVertices * 2)
#        for p in range(self.getNumPolygons()):
#            for point in range(startPoints[p], startPoints[p + 1]):
#                nextPoint = self.z(point)
#                dl = positions[nextPoint * 2 : nextPoint * 2 + 2] - positions[point * 2 : point * 2 + 2]
#                dl += 1.5
#                dl %= 1
#                dl -= 0.5
#                lVals[point] = np.sqrt(np.sum(dl**2))
#        print("lMean = ", np.mean(lVals))
#        print("lVar = ", np.var(lVals))
#        print("aMean = ", np.mean(ap))
#        print("aVar = ", np.var(ap))
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

    

