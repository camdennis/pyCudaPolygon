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
from matplotlib import pyplot as plt
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

    def getArea(self, pos):
        x = pos[::2]
        y = pos[1::2]
        x = np.concatenate((x, [x[0]]))
        y = np.concatenate((y, [y[0]]))
        a = np.dot(x[:-1], y[1:])
        a -= np.dot(x[1:], y[:-1])
        a /= 2
        a = np.abs(a)
        return a

    def generatePolygon(self, n):
        angles = 2 * np.pi * np.sort(np.random.rand(n))
        radius = np.random.rand(n)
        pos = np.vstack((radius * np.cos(angles), radius * np.sin(angles))).T.reshape(n * 2)
        return pos

    def generatePolygons(self, nArray, areaArray):
        totalN = np.sum(nArray).astype(int)
        self.setNumVertices(totalN)
        self.size = totalN
        polygonPos = []
        for i in range(len(nArray)):
            n = nArray[i]
            a = areaArray[i]
            pos = self.generatePolygon(n)
            area = self.getArea(pos)
            print(area)
            multiplier = a / area
            pos *= multiplier
            pos[::2] += np.random.rand()
            pos[1::2] += np.random.rand()
#            pos %= 1
            polygonPos.append(pos)
        polygonPos = np.concatenate(polygonPos)
        self.setPositions(polygonPos)

    def draw(self, nList):
        pos = self.getPositions()
        start = 0
        for n in nList:
            px = pos[start:start + 2 * n][::2]
            py = pos[start:start + 2 * n][1::2]
            px = np.concatenate((px, [px[0]]))
            py = np.concatenate((py, [py[0]]))
            plt.plot(px, py)
            start += 2 * n
        plt.gca().set_xlim([0, 1])
        plt.gca().set_ylim([0, 1])
        plt.show()

'''
    def initialize(self, initialStrain = 0.01, meanSoftness = -1.660):
        size = self.getGridSize()
        seed = self.getRandomSeed()
        np.random.seed(seed + 1)
        strains = np.random.normal(0, initialStrain, size * size * 2)
        softness = np.random.normal(meanSoftness, 2, size * size)
        self.setStrainX(strains[:size * size])
        self.setStrainY(strains[size * size:])
        self.setSoftness(softness)
        self.initializeIndexMatrix()
        self.initializeMovingAverageTargetSoftness(meanSoftness)
        self.updateMeanSMatrix()
        self.updateInverseSPKernel()
        rows = np.repeat(np.arange(size), size)
        cols = np.tile(np.arange(size), size)
        self.setRearrangerIDs(rows, cols)
        self.updateYieldStrainPx()
        self.updateYieldStrain()
'''

if __name__ == "__main__":
    m = model(size = 10, seed = None)
    m.setModelEnum("abnormal")
    n = int(sys.argv[1])
    nArray = np.ones(2, dtype = int) * 5
    areaArray = np.ones(2) * 0.05
    m.generatePolygons(nArray, areaArray)
    m.draw(nArray)

