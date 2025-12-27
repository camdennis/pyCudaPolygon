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
        return lpcp.Model.getNumVertices(self)

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
        v = self.getNumVertices()
        inside = np.array(lpcp.Model.getInsideFlag(self))
        maxNeighbors = len(inside) // v
        inside = inside.reshape(v, maxNeighbors)
        insideDict = dict()
        numNeighbors = self.getNumNeighbors()
        for i, el in enumerate(inside):
            allEl = el[:numNeighbors[i]]
            if (len(allEl) == 0):
                continue
            insideDict[i] = allEl
        return insideDict

    def getTU(self):
        v = self.getNumVertices()
        tu = np.array(lpcp.Model.getTU(self))
        maxNeighbors = len(tu) // v // 2
        t = tu[:v * maxNeighbors].reshape(v, maxNeighbors)
        u = tu[v * maxNeighbors:].reshape(v, maxNeighbors)
        tDict = dict()
        uDict = dict()
        numNeighbors = self.getNumNeighbors()
        for i in range(len(t)):
            allEl = t[i][:numNeighbors[i]]
            if (len(allEl) == 0):
                continue
            tDict[i] = allEl
            allEl = u[i][:numNeighbors[i]]
            if (len(allEl) == 0):
                continue
            uDict[i] = allEl
        return tDict, uDict

    def updateNeighbors(self, a):
        lpcp.Model.updateNeighbors(self, a)

    def updateOverlapArea(self, pointDensity):
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
        lpcp.Model.updateAreas(self)

    def getAreas(self):
        return np.array(lpcp.Model.getAreas(self))

    def getAreaOfPos(self, pos):
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

    def generatePolygon(self, n):
        # Use the polygon generator from the mixins package.
        from .mixins.polygon_utils import generate_polygon as _generate_polygon
        return _generate_polygon(self.rng, n)

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
            pg = self.generatePolygon(n) / 2
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
    # help to find the players. However, we need to figure out how
    # to get forces from the players. What information do we need?

    def z(self, i):
        startIndices = self.getStartIndices()
        shapeId = self.getShapeId()[i]
        if (i == startIndices[shapeId + 1] - 1):
            return startIndices[shapeId]
        return i + 1

    def getIntersectionsAndOutersections(self):
        self.updateNeighbors(0.0)
        numVertices = self.getNumVertices()
        intersections = defaultdict(list)
        outersections = defaultdict(list)
        t = defaultdict(list)
        u = defaultdict(list)
        insideFlagDict = self.getInsideFlag()
        tDict, uDict = self.getTU()
        shapeIds = self.getShapeId()
        numShapes = self.getNumPolygons()
        for key, value in self.getNeighbors().items():
            for i, v in enumerate(value):
                sh = np.min([shapeIds[v], shapeIds[key]]) * numShapes + np.max([shapeIds[v], shapeIds[key]])
                if not insideFlagDict[key][i]:
                    intersections[sh].append(key * numVertices + v)
                    t[sh].append(tDict[key][i])
                    outersections[sh].append(v * numVertices + key)
                    u[sh].append(uDict[key][i])
                else:
                    outersections[sh].append(key * numVertices + v)
                    t[sh].append(uDict[key][i])
                    intersections[sh].append(v * numVertices + key)
                    u[sh].append(tDict[key][i])
        return intersections, t, outersections, u
    
    def getPlayers(self):
        intersections, ts, outersections, us = self.getIntersectionsAndOutersections()
        numVertices = self.getNumVertices()
        players = []
        playerLengths = []
        nArray = self.getnArray()
        shapeIds = self.getShapeId()
        for key, values in intersections.items():
            for iteration, v in enumerate(values):
                i = v // numVertices
                j = v % numVertices
                ks = np.array(outersections[key]) // numVertices
                ls = np.array(outersections[key]) % numVertices
                # So i intersects j
                # Does i outsect anything?
                args = np.argwhere(ks == i).T[0]
                if len(args) == 0:
                    # Find where the thing exits
                    minDist = 1e9
                    n = nArray[shapeIds[i]]
                    nextIndex = -1
                    for index, k in enumerate(ks):
                        if (n + k - i) % n < minDist:
                            nextIndex = index
                            minDist = (n + k - i) % n
                    players.append(i)
                    players.append(j)
                    players.append(ks[nextIndex])
                    players.append(ls[nextIndex])
                    playerLengths.append(minDist + 2)
                else:
                    # Here we need to find the appropriate next index
                    # by finding the next t/u
                    t = ts[key][iteration]
                    bestIndex = -1
                    smallestU = 1e9
                    for arg in args:
                        u = us[key][arg]
                        if (u < t):
                            continue
                        if (u < smallestU):
                            bestIndex = arg
                            smallestU = u
                    players.append(i)
                    players.append(j)
                    players.append(ks[bestIndex])
                    players.append(ls[bestIndex])
                    playerLengths.append((n + ks[bestIndex] - i) % n + 2)
        return np.array(players).reshape(len(players) // 4, 4), np.cumsum(np.concatenate((np.array([0]), np.array(playerLengths))))