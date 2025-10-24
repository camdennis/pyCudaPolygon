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
#from scipy.special import gamma as gammaFunction
#import scipy.fftpack as sp

class model(lpcp.Model):
    def __init__(self, 
                 size = 0,
                 seed = None,
                 modelType = "normal",
    ):
        lpcp.Model.__init__(self, size)
        self.setModelEnum(modelType)
    
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

    def initializeRandomSeed(self, seed = None):
        if seed is None:
            seed = np.random.randint(2**31)
        lpcp.Model.initializeRandomSeed(self, seed)

    def setPositions(self, positions):
        lpcp.Model.setPositions(self, positions)

    def getPositions(self):
        return np.array(lpcp.Model.getPositions(self))

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

    def updateNeighbors(self, a):
        lpcp.Model.updateNeighbors(self, a)

    def setnArray(self, nArray):
        nArray = nArray.astype(int)
        # we don't actually set the nArray, we set the startIndices:
        startIndices = np.concatenate(([0], np.cumsum(nArray)))
        lpcp.Model.setStartIndices(self, startIndices)

    def setMaxEdgeLength(self, maxEdgeLength):
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
        pts = np.random.rand(max(8, n*3), 2) - 0.5           # oversample
        hull = convex_hull(pts)
        if len(hull) < n:
            # subdivide hull edges to reach n
            extra = []
            i = 0
            while len(hull) + len(extra) < n:
                a = hull[i % len(hull)]
                b = hull[(i+1) % len(hull)]
                t = np.random.rand()
                extra.append(a*(1-t) + b*t)
                i += 1
            hull = np.vstack([hull, np.array(extra)])
        hull = hull[:n]       # trim if needed
        return hull.reshape(n*2)

    '''
    def generatePolygons(self, nArray, areaArray):
        nArray = nArray.astype(int)
        totalN = np.sum(nArray).astype(int)
        self.setNumVertices(totalN)
        self.size = totalN
        polygonPos = []
        self.setnArray(nArray)
        for i in range(len(nArray)):
            n = nArray[i]
            a = areaArray[i]
            pos = self.generatePolygon(n) / 50
            pos[::2] += np.random.rand()
            pos[1::2] += np.random.rand()
            polygonPos.append(pos)
        polygonPos = np.concatenate(polygonPos)
        pos += 1
        pos %= 1
        self.setPositions(polygonPos)
        self.updateAreas()
        areas = self.getAreas()
        scaling = []
        for i in range(len(nArray)):
            scaling.append(np.sqrt(np.repeat(areaArray[i], nArray[i] * 2) / areas[i]))
        scaling = np.concatenate(scaling)
        self.setPositions(polygonPos * scaling)
    '''
    def draw(self, numbering = True):

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

        for n in self.getnArray():
            px = pos[start:start + 2 * n][::2]
            py = pos[start:start + 2 * n][1::2]
            px = np.concatenate((px, [px[0]]))
            py = np.concatenate((py, [py[0]]))
            px, py = fixPXPY(px, py)

            for i in range(3):
                for j in range(3):
                    ax.plot(px + i - 1, py + j - 1, '-o', markersize=3)

            # Label each vertex (except the repeated closing point)
            for k in range(len(px) - 1):
                textX = (px[k] + 1) % 1
                textY = (py[k] + 1) % 1
                ax.text(textX, textY, str(start//2 + k),
                        fontsize=8, color='k', ha='left', va='bottom')

            start += 2 * n

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect(1)
#        plt.show()

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
            pg[::2] += np.random.rand()
            pg[1::2] += np.random.rand()
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