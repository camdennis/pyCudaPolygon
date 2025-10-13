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

    def updateNeighbors(self, a):
        lpcp.Model.updateNeighbors(self, a)

    def setnArray(self, nArray):
        # we don't actually set the nArray, we set the startIndices:
        startIndices = np.concatenate(([0], np.cumsum(nArray)))
        lpcp.Model.setStartIndices(self, startIndices)

    def setMaxEdgeLength(self, maxEdgeLength):
        lpcp.Model.setMaxEdgeLength(self, maxEdgeLength)

    def updateAreas(self):
        lpcp.Model.updateAreas(self)

    def getArea(self):
        return np.array(lpcp.Model.getArea(self))

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
        angles = 2 * np.pi * np.sort(np.random.rand(n))
        radius = np.random.rand(n) * 0.2
        pos = np.vstack((radius * np.cos(angles), radius * np.sin(angles))).T.reshape(n * 2)
        return pos

    def generatePolygons(self, nArray, areaArray):
        totalN = np.sum(nArray).astype(int)
        self.setNumVertices(totalN)
        self.size = totalN
        polygonPos = []
        self.setnArray(nArray)
        for i in range(len(nArray)):
            n = nArray[i]
            a = areaArray[i]
            pos = self.generatePolygon(n)
            area = self.getAreaOfPos(pos)
#            multiplier = a / area
#            pos *= multiplier
            pos[::2] += np.random.rand()
            pos[1::2] += np.random.rand()
#            pos %= 1
            polygonPos.append(pos)
            area = self.getAreaOfPos(pos)
        polygonPos = np.concatenate(polygonPos)
        self.setPositions(polygonPos)

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
                ax.text(px[k], py[k], str(start//2 + k),
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
    
    def areaTesting(self):
        positions = self.getPositions()
        startIndices = self.getStartIndices()
        numShapes = len(startIndices) - 1
        areas = np.zeros(numShapes)
        for idx in range(numShapes):
            start = startIndices[idx]
            end = startIndices[idx + 1]
            startY = positions[2 * start + 1]
            for i in range(start, end - 1):
                dx = (positions[2 * i] - positions[2 * i + 2] + 0.5) % 1 - 0.5
                dy1 = (positions[2 * i + 1] - startY + 0.5) % 1
                dy2 = (positions[2 * i + 3] - startY + 0.5) % 1
                areas[idx] += dx * (dy1 + dy2 - 1.0 + 2.0 * startY) / 2.0
            dx = (positions[2 * end - 2] - positions[2 * start] + 0.5) % 1 - 0.5
            dy1 = (positions[2 * end - 1] - startY + 0.5) % 1 - 0.5
            areaBit = dx * (dy1 + 2.0 * startY) / 2.0
            areas[idx] += areaBit
        return areas
    
    def saveModel(self, dirName, overwrite = False):
        if not overwrite and os.path.isdir(dirName):
            raise Exception("Packing exists. Not saving. To save over this file, set kwarg overwrite = True")
        if not os.path.isdir(dirName):
            os.mkdir(fileName)
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