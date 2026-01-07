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
#        maxNeighbors = len(tu) // v // 2
#        t = tu[:v * maxNeighbors].reshape(v, maxNeighbors)
#        u = tu[v * maxNeighbors:].reshape(v, maxNeighbors)
#        tDict = dict()
#        uDict = dict()
#        numNeighbors = self.getNumNeighbors()
#        for i in range(len(t)):
#            allEl = t[i][:numNeighbors[i]]
#            if (len(allEl) == 0):
#                continue
#           tDict[i] = allEl
#           allEl = u[i][:numNeighbors[i]]
#           if (len(allEl) == 0):
#               continue
#            uDict[i] = allEl
#        return tDict, uDict

    def updateNeighbors(self, a):
        lpcp.Model.updateNeighbors(self, a)

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

    def draw(self, numbering = False):

        def fixPXPY(px, py):
            minX = min(px)
            maxX = max(px)
            minY = min(py)
            maxY = max(py)
            px += 1.5 - minX
            py += 1.5 - minY
            px %= 1
            py %= 1
            px += minX - 0.5
            py += minY - 0.5
            return px, py

        pos = self.getPositions()
        start = 0
        fig, ax = plt.subplots()

        nArray = self.getnArray()
        cmap = plt.get_cmap('tab20')  # choose any colormap you prefer

        for poly_idx, n in enumerate(nArray):
            color = cmap(poly_idx % cmap.N)
            px = pos[start:start + 2 * n][::2]
            py = pos[start:start + 2 * n][1::2]
            px = np.concatenate((px, [px[0]]))
            py = np.concatenate((py, [py[0]]))
            px, py = fixPXPY(px, py)

            for i in range(3):
                for j in range(3):
                    ax.plot(px + i - 1, py + j - 1,
                            '-o', markersize=3,
                            color=color,
                            markerfacecolor=color,
                            markeredgecolor=color)

            if (numbering):
                for k in range(len(px) - 1):
                    textX = (px[k] + 1) % 1
                    textY = (py[k] + 1) % 1
                    ax.text(textX, textY, str(start // 2 + k),
                            fontsize = 8, color = 'k', ha = 'left', va = 'bottom')

            start += 2 * n

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

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
#        if (np.sum(abs(areas - targetAreas)) >= 1e-9):
#            raise Exception("One of the areas was not set correctly. This may be due to the area of the original shape being too small for the boundary conditions")
        
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
        shapeId = np.argmin(abs(startIndices - edge))
        if edge > startIndices[shapeId]:
            shapeId -= 1
        return shapeId

    def isRealNeighbor(self, e1, e2):
        startArray = self.getStartArray()
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

    def getIntersections(self):
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
        
        # Here we just update the neighbors. In practice I would like to implement
        # Something to check the neighbors for intersection and then save that
        self.updateNeighbors(0.0)
        numVertices = self.getNumVertices()
        neighbors = np.array(lpcp.Model.getNeighbors(self))
        maxNeighbors = len(neighbors) // numVertices
        numNeighbors = self.getNumNeighbors()
        insideFlag = self.getInsideFlag()
        shapeIds = self.getShapeId()
        numShapes = self.getNumPolygons()
        intersections = []
        t, u = self.getTU().reshape(2, numVertices * maxNeighbors)
        newTU = []
        for n1 in range(numVertices):
            for i in range(numNeighbors[n1]):
                s1 = shapeIds[n1]
                n2 = neighbors[n1 * maxNeighbors + i]
                s2 = shapeIds[n2]
                tVal = t[n1 * maxNeighbors + i]
                uVal = u[n1 * maxNeighbors + i]
                if insideFlag[n1 * maxNeighbors + i]:
                    intersection = pack([s2, s1, n1, n2])
                    newTU.append(uVal)
                    newTU.append(tVal)
                else:
                    intersection = pack([s1, s2, n2, n1])
                    newTU.append(tVal)
                    newTU.append(uVal)
                intersections.append(intersection)
        intersections = np.array(intersections)
        numIntersections = intersections.size
        newTU = np.array(newTU).reshape(numIntersections, 2)
        # Now we want to sort these!
        args = np.argsort(intersections)
        intersections = intersections[args]
        newTU = newTU[args].reshape(numIntersections * 2)
        # Now we want to find the start and end indices
        starts = np.zeros(np.uint64(numVertices)  << np.uint64(16), dtype = int)
        ends = np.zeros(np.uint64(numVertices)  << np.uint64(16), dtype = int)
        packedShape = intersections.astype(np.uint64) >> np.uint64(32)
        shapeStartFlag = np.concatenate(([True], np.diff(packedShape).astype(bool)))
        bizarroIntersection = flipPack(intersections[0])
        starts[bizarroIntersection] = 0
        for i in range(1, len(intersections)):
            if shapeStartFlag[i]:
                bizarroIntersection = flipPack(intersections[i])
                # This intersection, if you search it should start here
                starts[bizarroIntersection] = i
                # The intersection before this, if you search it should end before this
                bizarroIntersectionM1 = flipPack(intersections[i - 1])
                ends[bizarroIntersectionM1] = i - 1
        # The last one ends as well
        bizarroIntersection = flipPack(intersections[-1])
        ends[bizarroIntersection] = len(intersections) - 1
        return np.array(intersections), newTU, starts, ends
    
    def getPlayers(self):

        def pack(numbers):
            result = (numbers[3] & 0xFFFF) | \
                    ((numbers[2] & 0xFFFF) << 16) | \
                    ((numbers[1] & 0xFFFF) << 32) | \
                    ((numbers[0] & 0xFFFF) << 48)
            return result

        def flipPack(val, flip = True):
            a = (val >> 48) & 0xFFFF
            b = (val >> 32) & 0xFFFF
            if not flip:
                (a, b) = (b, a)
            return (a & 0xFFFF) | ((b & 0xFFFF) << 16)

        def flipPack2(val, flip = True):
            a = (val >> 16) & 0xFFFF
            b = val & 0xFFFF
            if not flip:
                (a, b) = (b, a)
            return (a & 0xFFFF) | ((b & 0xFFFF) << 16)

        def findMinCyclic(a, i, ni, l, r):
            def distance(j):
                return (j - i + ni) % ni
#            l, r = 0, len(a) - 1
            while l < r:
                mid = (l + r) // 2
                if distance((a[mid] >> 16) & 0xFFFF) < distance((a[r] >> 16) & 0xFFFF):
                    r = mid
                elif distance((a[mid] >> 16) & 0xFFFF) > distance((a[r] >> 16) & 0xFFFF):
                    l = mid + 1
                else:
                    # a[mid] == a[r], discard one duplicate
                    r -= 1
            return (a[l] >> 16) & 0xFFFF

        def leftmost(a, val, l0, r0):
            res = None
            l, r = l0, r0
            while l <= r:
                mid = (l + r) // 2
                if (a[mid] >> 16) & 0xFFFF < val:
                    l = mid + 1
                else:
                    if (a[mid] >> 16) & 0xFFFF == val:
                        res = mid
                    r = mid - 1
            if res is None:
                return None, False
            # Check if there is another occurrence to the right
            only = (res == r0) or ((a[res + 1] >> 16) & 0xFFFF != val)
            return res, only

        def rightmost(a, val, l, r):
            res = None
            while l <= r:
                mid = (l + r) // 2
                if (a[mid] >> 16) & 0xFFFF  > val:
                    r = mid - 1
                else:
                    if (a[mid] >> 16) & 0xFFFF  == val:
                        res = mid
                    l = mid + 1
            return res

        intersections, newTU, starts, ends = self.getIntersections()
        numVertices = self.getNumVertices()
        numNeighbors = self.getNumNeighbors()
        n = self.getnArray()
        r = -1
        numIntersections = len(intersections)
        players = np.zeros(numIntersections, dtype = int)
        for i in range(len(intersections)):
            intersectionId = flipPack(intersections[i], flip = False)
            start = starts[intersectionId]
            end = ends[intersectionId]
            # You want to do a binary search to find the index with the minimum distance
            # Relative to i
            ni = n[(intersections[i] >> 48) & 0xFFFF]
            thisIntersectionsFirstElement = (intersections[i] & 0xFFFF)
            minVal = findMinCyclic(intersections, thisIntersectionsFirstElement, ni, start, end)
            l, only = leftmost(intersections, minVal, start, end)
#            print(minVal, l, only)
            uMin = np.inf
            uMinArg = l
            if not only:
                r = rightmost(intersections, minVal, l, end)
                # You'll need t
                tVal = newTU[2 * i]
#                print(l, r)
                for k in range(l, r + 1):
                    # Is this the right u value?
                    u = newTU[2 * k + 1]
                    if u < t:
                        continue
                    if u < uMin:
                        uMin = u
                        uMinArg = k
            players[i] = uMinArg
        return players

    def getOverlapArea(self):
        return np.array(lpcp.Model.getOverlapArea(self))

    def getOverlapAreaPY(self):
        def h(pt1, pt2):
            # Get dh
            dy = pt2[1] - pt1[1] + 1.5
            dx = pt2[0] - pt1[0] + 1.5
            dy %= 1
            dx %= 1
            dy -= 0.5
            dx -= 0.5
            return (2 * pt1[0] + dx) * dy

        intersections, newTU, starts, ends = self.getIntersections()
        players = self.getPlayers()
        positions = self.getPositions()
        x = positions[::2]
        y = positions[1::2]
        allF = np.zeros((2 * len(players), 2))
        sol = 0
        shapeId = self.getShapeId()
        nArray = self.getnArray()
        for index in range(len(players)):
            # s(i), s(j), j, i
            i = intersections[index] & 0xFFFF
            j = (intersections[index] >> 16) & 0xFFFF
            k = intersections[players[index]] & 0xFFFF
            l = (intersections[players[index]] >> 16) & 0xFFFF
            pi = positions[2 * i: 2 * i + 2]
            zi = self.z(i)
            pzi = positions[2 * zi: 2 * zi + 2]
            pk = positions[2 * k: 2 * k + 2]
            zk = self.z(k)
            pzk = positions[2 * zk: 2 * zk + 2]
            r1 = pzi - pi + 1.5
            r2 = pzk - pk + 1.5
            r1 %= 1
            r1 -= 0.5
            r2 %= 1
            r2 -= 0.5
            t1 = newTU[2 * index]
            fij = (pi + t1 * r1 + 1) % 1
            t2 = newTU[2 * players[index]]
            fkl = (pk + t2 * r2 + 1) % 1
            if (i == l):
                sol += h(fij, fkl)
                continue
            else:
                sol += h(fij, positions[self.z(i) * 2: self.z(i) * 2 + 2])
                sol += h(positions[l * 2: l * 2 + 2], fkl)
            n = nArray[shapeId[i]]
            dist = (l - i + n) % n
            if dist < 2:
                continue
            currIndex = self.z(i)
            endIndex = self.z(l)
            while (currIndex != endIndex):
                nextIndex = self.z(currIndex)
                sol += h(positions[currIndex * 2 : currIndex * 2 + 2],
                    positions[nextIndex * 2: nextIndex * 2 + 2])
                currIndex = nextIndex
            sol += h(positions[currIndex * 2 : currIndex * 2 + 2],
                    positions[l * 2: l * 2 + 2])
        return sol / 2


