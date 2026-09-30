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
import os

class Mixin():
    
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
        self._impl.sortKeys(endBit)

    # misc

    def minimizeGDStep(self, dt = 1e-3, addedForce = None, dontMove = False, maxDisplacement = 0.05, nIter = 100, tol = 1e-15, minimizerType = "GD"):
        if self.getModelEnum() not in ("edgeOnly", "areaOnly", "softBody"):
            self.updateNeighborCells()
            self.updateNeighbors()
            self.updateOutersections()
        self.updatePolygonGeometry()
        self.updateForceEnergy()
        actualIter = 0
        if addedForce is not None:
            self.setForces(self.getForces() + addedForce)
        if (self.getMaxUnbalancedForce() * dt > maxDisplacement):
            print("here")
            return -1, -1
        if dontMove:
            return self.getEnergy(), actualIter
        if (self.getModelEnum() == "normal"):
            self.projectForce()
        self.updateVertices(dt)
        actualIter = 0
        # TODO(M6): constraint projection went here, gated on nIter > 0.
        self.updatePolygonGeometry()
        return self.getEnergy(), 0

    def minimizeGD(self, maxUnbalancedForceThreshold = 1e-14, dt = 1e-3, maxSteps = -1, addedForce = None, progressBar = False, checkpointDir = None, checkpointFreq = 1, overwriteCheckpoint = False, maxDisplacement = 0.5, maxConstraintViolationThreshold = 1e-3, nIter = 100, tol = 1e-15, minimizerType = "GD"):
        if maxSteps == -1 and maxUnbalancedForceThreshold is None:
            raise ValueError("maxSteps=-1 requires maxUnbalancedForceThreshold to be set")
        dontMove = False
        if maxSteps == 0:
            maxSteps = 1
            dontMove = True

        total = maxSteps if maxSteps != -1 else None
        meanIter = 0
        maxIter = 0
        minIter = 1e9
        actualIter = 0
        with tqdm(total = total, desc = "Processing", disable = (not progressBar)) as pbar:
            successes = 0
            prevEnergy = None
            step = 0
            energy = None
            while maxSteps == -1 or step < maxSteps:
                if checkpointDir is not None and (step % checkpointFreq == 0):
                    if not os.path.isdir(checkpointDir):
                        os.makedirs(checkpointDir)
                    self.saveModel(checkpointDir + "/" + str(step), overwrite = overwriteCheckpoint)
                if progressBar:
                    pbar.update(1)
                energy, actualIter = self.minimizeGDStep(addedForce = addedForce, dt = dt, dontMove = dontMove, maxDisplacement = maxDisplacement, nIter = nIter, tol = tol, minimizerType = minimizerType)
                meanIter += actualIter / maxSteps
                maxIter = np.max((actualIter, maxIter))
                minIter = np.min((actualIter, minIter))
                if self.getModelEnum() == "normal":
                    fEff = self.getMaxEffectiveForce(dt, minimizerType)
                else:
                    fEff = self.getMaxUnbalancedForce()
                if maxUnbalancedForceThreshold is not None and fEff <= maxUnbalancedForceThreshold:
                    return energy, 0, np.array([0])
                if (self.getModelEnum() == "normal"  and np.max(self.getConstraintViolation()) > maxConstraintViolationThreshold):
                    return energy, 0, np.array([0])
                if prevEnergy is not None and (energy > prevEnergy or energy == -1):
                    dt = max(1e-32, dt / 2.1)
                    successes = 0
                else:
                    successes += 1
                    prevEnergy = energy
                if successes > 5:
                    successes = 0
                    dt = min(1e4, dt * 1.9)
                step += 1
        if energy is None:
            raise Exception("This is not an appropriate maximum step size. Reset maxSteps.")
        print(minIter, meanIter, maxIter)
        return energy, dt, np.array([minIter, meanIter, maxIter])

    def minimizeFIREStep(self, dt, alpha, nPos, dtMax=0.1, alphaStart=0.1, fAlpha=0.99, fInc=1.1, fDec=0.5, nMin=5):
        """Single FIRE step. Returns (energy, dt, alpha, nPos)."""
        return self._impl.minimizeFIREStep(dt, alpha, nPos, dtMax, alphaStart, fAlpha, fInc, fDec, nMin)

    def minimizeFIRELoop(self, maxForceThreshold=1e-14, dt=1e-3, maxSteps=100000, dtMax=0.1, alphaStart=0.1, fAlpha=0.99, fInc=1.1, fDec=0.5, nMin=5, progressBar=False):
        """
        FIRE minimizer generator. Yields (energy, maxForce, dt, step) each step.
        Stops when maxForce <= maxForceThreshold or maxSteps is reached.
        """
        self._impl.resetVelocities()
        self.updatePolygonGeometry()
        self.updateForceEnergy()
        if self.getModelEnum() == "normal":
            self.projectForce()
        alpha = alphaStart
        nPos = 0
        with tqdm(total=maxSteps, desc="FIRE", disable=(not progressBar)) as pbar:
            for step in range(maxSteps):
                energy, dt, alpha, nPos = self.minimizeFIREStep(dt, alpha, nPos, dtMax, alphaStart, fAlpha, fInc, fDec, nMin)
                maxForce = self.getMaxUnbalancedForce()
                pbar.update(1)
                yield energy, maxForce, dt, step + 1
                if maxForce <= maxForceThreshold:
                    return

    def minimizeFIRE(self, maxForceThreshold=1e-14, dt=1e-3, maxSteps=100000000, dtMax=0.1, alphaStart=0.1, fAlpha=0.99, fInc=1.1, fDec=0.5, nMin=5, progressBar=False, checkpointDir=None, checkpointFreq=1, overwriteCheckpoint=False):
        """
        FIRE minimizer. For the 'normal' model call
        updateNeighborCells/updateNeighbors/updateOutersections first.
        maxSteps=0 initializes state without stepping.
        Returns (energy, dt, steps).
        """
        self._impl.resetVelocities()
        if self.getModelEnum() == "normal":
            self.initForceEnergy()
        self.updatePolygonGeometry()
        self.updateForceEnergy()
        if self.getModelEnum() == "normal":
            self.projectForce()
        # One tiny GD step to refresh intersection detection and escape
        # containment configurations (polygon fully inside another) that are
        # invisible to the edge-crossing detector at exact loaded vertices.
        counter = 0
        if self.getModelEnum() == "normal" and maxSteps > 0:
            self.updateVertices(dt)
            self.updatePolygonGeometry()
            self.updateNeighborCells()
            self.updateNeighbors()
            self.updateOutersections()
            self.updateForceEnergy()
            self.projectForce()
            self._impl.resetVelocities()
        if maxSteps == 0:
            return self.getEnergy(), dt, 0
        energy, steps = self.getEnergy(), 0
        prevEnergy = 1e9

        for energy, _, dt, steps in self.minimizeFIRELoop(
                maxForceThreshold=maxForceThreshold, dt=dt, maxSteps=maxSteps,
                dtMax=dtMax, alphaStart=alphaStart, fAlpha=fAlpha,
                fInc=fInc, fDec=fDec, nMin=nMin,
                progressBar=progressBar):
            if checkpointDir is not None and (steps % checkpointFreq == 0):
                if not os.path.isdir(checkpointDir):
                    os.makedirs(checkpointDir)
                self.saveModel(checkpointDir + "/" + str(steps), overwrite=overwriteCheckpoint)
                print(self.getEnergy(), self.getMaxUnbalancedForce())
            # TODO(M6): constraint-projection iteration statistics were collected
            # here and returned as [min, mean, max]. Restore when SHAKE returns.
            if (np.abs(energy - prevEnergy) < 1e-15):
                counter += 1
            if (counter > 5):
                break
            prevEnergy = energy
            
        if (steps == maxSteps):
            print("This may not have fully minimized")
        return energy, dt, steps

    def saveModel(self, dirName, overwrite = False):
        if not overwrite and os.path.isdir(dirName):
            raise Exception("Packing exists. Not saving. To save over this file, set kwarg overwrite = True")
        if not os.path.isdir(dirName):
            os.mkdir(dirName)
        # Get stuff to save:
        modelEnum = self.getModelEnum()
        numVertices = self.getNumVertices()
        nArray = np.diff(self.getStartIndices())
        vertices = self.getVertices()
        maxEdgeLength = self.getMaxEdgeLength()
        kv = dict()
        kv["modelEnum"] = str(modelEnum)
        kv["numVertices"] = numVertices
        kv["maxEdgeLength"] = maxEdgeLength
        kv["stiffness"] = self.getStiffness()
        kv["compressibility"] = self.getCompressibility()
        saveFile = dirName + "/scalars.dat"
        with open(saveFile, 'w') as f:
            for k in kv.keys():
                f.write(k + ":\t" + str(kv[k]) + "\n")
        f.close()
        # We can save the nArray and vertices back to back in one
        # file since splitting it up is easy with the scalars.dat file
        state = np.concatenate((nArray, vertices))
        np.save(dirName + "/state", state)
        # Save per-polygon arrays needed to fully restore model state
        np.savez(dirName + "/arrays",
                 targetAreas=self.getTargetAreas(),
                 targetEdgeLengths=self.getTargetEdgeLengths())

    def minimizeSoftBody(self):
        initialModelType = self.getModelEnum()
        self.setStiffness(1)
        self.setCompressibility(1)
        self.updateForceEnergy()
        self.setModelEnum("softBody")
        self.minimizeFIRE(dt = 0.1, maxForceThreshold = 1e-14)
        self.resetAreas()
        self.updatePolygonGeometry()
        self.setModelEnum(initialModelType)

    def loadModel(self, dirName):
        state = np.load(dirName + "/state.npy")
        scalarsFile = dirName + "/scalars.dat"
        with open(scalarsFile, 'r') as f:
            lines = f.readlines()
        f.close()
        modelType = "normal"
        numVertices = 0
        maxEdgeLength = 0.5
        stiffness = 0.0
        compressibility = 0.0
        for line in lines:
            k, v = line.split("\n")[0].split("\t")
            if (k == "modelEnum:"):
                modelType = v
            elif (k == "numVertices:"):
                numVertices = int(v)
            elif (k == "maxEdgeLength:"):
                maxEdgeLength = float(v)
            elif (k == "stiffness:"):
                stiffness = float(v)
            elif (k == "compressibility:"):
                compressibility = float(v)
        try:
            self.setModelEnum(modelType)
        except Exception:
            print("Warning: model enum not found. Setting to normal")
            self.setModelEnum("normal")
        self.setNumVertices(numVertices)
        nArray = state[:-self.getNumVertices() * 2].astype(int).copy()
        self.setnArray(nArray)
        vertices = state[-self.getNumVertices() * 2:].copy()
        self.setVertices(vertices)
        self.setMaxEdgeLength(maxEdgeLength)
        self.setStiffness(stiffness)
        self.setCompressibility(compressibility)
        self.initializeNeighborCells()
        # Restore per-polygon constraint arrays if saved with new format
        arrays_path = dirName + "/arrays.npz"
        if os.path.exists(arrays_path):
            arrays = np.load(arrays_path)
            self.setTargetAreas(arrays['targetAreas'])
            self.setTargetEdgeLengths(arrays['targetEdgeLengths'])

    def makeSubModel(self, sub):
        pos0 = self.getVertices()
        nArray = self.getnArray()
        nArraySub = nArray[sub]
        totalVertices = np.sum(nArraySub)
        vertices = np.zeros(totalVertices * 2)
        curr = 0
        startIndices = self.getStartIndices()
        for i in range(len(sub)):
            ind1 = curr * 2
            ind2 = curr * 2 + nArray[sub[i]] * 2
            ind3 = startIndices[sub[i]] * 2
            ind4 = startIndices[sub[i] + 1] * 2
            vertices[ind1 : ind2] = pos0[ind3 : ind4]
            curr += nArray[sub[i]]
        nArray = self.getnArray()[sub]
        self.setNumVertices(vertices.size // 2)
        self.setVertices(vertices)
        self.setnArray(nArray)
