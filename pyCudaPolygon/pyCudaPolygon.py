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

    def initializeRandomSeed(self, seed = None):
        if seed is None:
            seed = np.random.randint(2**31)
        lpcp.Model.initializeRandomSeed(self, seed)

    def setPositions(self, positions):
        size = self.getNumVertices()
        lpcp.Model.setPositions(self, positions)

    def getPositions(self):
        size = self.getNumVertices()
        return np.array(lpcp.Model.getPositions(self))

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
    m = model(size = 64, seed = None)
    m.setModelEnum("abnormal")
    n = int(sys.argv[1])

