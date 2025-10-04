#from .pyCudaPolygonLink import libpyCudaPolygon as lpcp
#from . import enums
import libpyCudaPolygon as lpcp
import enums
#import libpyCudaElasto as lpcp
#import calcKernel
#import enums

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

    def updateNeighbors(self):
        lpcp.Model.updateNeighbors(self)

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
            print(area)
#            multiplier = a / area
#            pos *= multiplier
            pos[::2] += np.random.rand()
            pos[1::2] += np.random.rand()
#            pos %= 1
            polygonPos.append(pos)
            area = self.getAreaOfPos(pos)
        polygonPos = np.concatenate(polygonPos)
        self.setPositions(polygonPos)

    def draw(self, nList, gridSize):

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
        for n in nList:
            px = pos[start:start + 2 * n][::2]
            py = pos[start:start + 2 * n][1::2]
            px = np.concatenate((px, [px[0]]))
            py = np.concatenate((py, [py[0]]))
            px, py = fixPXPY(px, py)
            for i in range(3):
                for j in range(3):
                    plt.plot(px + i - 1, py + j - 1)
            start += 2 * n
        for i in range(gridSize):
            plt.plot([0, 1], [i / gridSize, i / gridSize], color = 'b')
            plt.plot([i / gridSize, i / gridSize], [0, 1], color = 'b')
        plt.gca().set_xlim([0, 1])
        plt.gca().set_ylim([0, 1])
        plt.gca().set_aspect(1)
        plt.show()

    def getStartIndices(self):
        return lpcp.Model.getStartIndices(self)

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

if __name__ == "__main__":
    m = model(size = 10, seed = 1)
    m.setModelEnum("abnormal")
    n = int(sys.argv[1])
    nArray = np.ones(2, dtype = int) * 5
    areaArray = np.ones(2) * 0.05
    m.generatePolygons(nArray, areaArray)
    m.setPositions((m.getPositions() + 1) % 1)
    m.updateAreas()
    print(m.areaTesting())
    print(m.getAreas())
    m.setMaxEdgeLength(0.2)
    m.initializeNeighborCells()
    m.updateNeighborCells()
    print(m.getNeighborCells())
    print(m.getBoxCounts())
    print(m.getNeighborIndices())
    m.draw(nArray, 5)

