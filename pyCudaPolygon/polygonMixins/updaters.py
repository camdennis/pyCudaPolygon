import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import circulant
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import functools
import threading
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import math
from .. import enums

class Mixin():

    # updaters

    def updateNeighbors(self):
        self._impl.updateNeighbors()

    def updateValidAndCounts(self):
        return self._impl.updateValidAndCounts()

    def updateIntersections(self):
        return self._impl.updateIntersections()

    def updateOverlapAreaGOLD(self, pointDensity):
        # This is a slow estimation
        self._impl.updateOverlapAreaGOLD(pointDensity)

    def updateOverlapArea(self):
        self._impl.updateOverlapArea()

    def updatePolygonGeometry(self):
        self._impl.updatePolygonGeometry()

    def projectForce(self):
        self._impl.projectForce()

    def saveTentativeVertices(self):
        self._impl.saveTentativeVertices()

    def getMaxEffectiveForce(self, dt, minimizerType="GD"):
        mEnum = enums.minimizerEnum.GD if minimizerType == "GD" else enums.minimizerEnum.FIRE
        return self._impl.getMaxEffectiveForce(dt, mEnum)

    def updateNeighborCells(self):
        self._impl.updateNeighborCells()
        
    def updateForceEnergy(self):
        self._impl.updateForceEnergy()

    def updateVertices(self, dt):
        self._impl.updateVertices(dt)

    def resetAreas(self):
        self._impl.resetAreas()

    def updateOutersections(self):
        self._impl.updateOutersections()