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

def energy(phi, phi0, phif0):
    phiFinal = 2 * np.pi - np.sum(phi)
    return (np.sum((phi - phi0)**2) + (phiFinal - phif0)**2) / 2

def grad(phi, phi0, phif0):
    grad0 = phi - phi0
    grad0 += - (2 * np.pi - np.sum(phi) - phif0)
    return grad0

def utilArea(vertices):
    return (np.sum(vertices[:-1, 0] * vertices[1:, 1] - vertices[1:, 0] * vertices[:-1, 1])) / 2

def constraints(phi, n, targetArea):
    theta = np.zeros(n)
    theta[0] = 0.0
    for i in range(1, n):
        theta[i] = theta[i - 1] + phi[i - 1]
    
    # Edge vectors
    e = np.column_stack((np.cos(theta), np.sin(theta)))
    
    # Closure
    closure = np.sum(e, axis = 0)
    
    # Area
    vertices = np.zeros((n + 1, 2))
    vertices[0] = [0, 0]
    for i in range(n):
        vertices[i + 1] = vertices[i] + e[i]

    area = utilArea(vertices)
    return np.array([closure[0], closure[1], area / targetArea - 1])

class Mixin():
    def __init__(self, rng = None):
        # random number generator fallback
        self.rng = rng if rng is not None else np.random.default_rng()

    def generateRandomPolygon(self, n, kappa, packingFraction):
        targetArea = n**2 / kappa**2
        phi0 = self.rng.dirichlet(np.ones(n)) * 2 * np.pi
        phif0 = phi0[-1]
        phi0 = phi0[:-1]
        
        bounds = [(0, np.pi)] * (n - 1)
        ineqCons = {'type': 'ineq',
                    'fun': lambda phi: np.array([np.sum(phi) - np.pi,
                                                2 * np.pi - np.sum(phi)])}
        eqCons = {'type': 'eq',
                'fun': lambda phi: constraints(phi, n, targetArea)}
        result = minimize(energy, phi0, args = (phi0, phif0), 
            method = 'SLSQP',
            jac = grad,
            bounds = bounds,
            constraints = [eqCons, ineqCons])
        if not result.success:
            return None
        newPhi = result.x
        theta = np.zeros(n)
        theta[0] = 0.0
        for i in range(1, n):
            theta[i] = theta[i-1] + newPhi[i-1]
        
        # Edge vectors
        e = np.column_stack((np.cos(theta), np.sin(theta)))
        
        # Build vertices cumulatively
        vertices = np.zeros((n+1, 2))
        vertices[0] = [0.0, 0.0]
        for i in range(n):
            vertices[i+1] = vertices[i] + e[i]
        
        # Optional: enforce closure (last vertex = first) to fix numerical drift
        vertices[-1] = vertices[0]
        return vertices

    def generateRandomPolygons(self, numPolygons, n, kappa, phi, maxTries = 10):
        def rotate(vertices, angle):
            cx, cy = np.mean(vertices, axis = 0)
            for i in range(len(vertices)):
                xNew = (vertices[i][0] - cx) * np.cos(angle) - (vertices[i][1] - cy) * np.sin(angle) + cx
                yNew = (vertices[i][0] - cx) * np.sin(angle) + (vertices[i][1] - cy) * np.cos(angle) + cy
                vertices[i] = [xNew, yNew]
            return vertices              

        self.setnArray(np.ones(numPolygons) * n)
        numPolygons = self.getNumPolygons()
        numVertices = self.getNumVertices()
        positions = np.zeros((numVertices, 2))
        l = np.ones(n)
        targetArea = phi / numPolygons
        i = 0
        tries = 0
        while i < numPolygons:
            if tries > maxTries * numPolygons:
                raise Exception("We've tried. These constraints just aren't working out")
            vertices = self.generateRandomPolygon(n, kappa, phi)
            if vertices is None:
                tries += 1
                continue
            targetArea = phi / numPolygons
            vertices *= np.sqrt(phi / numPolygons / utilArea(vertices))
            pos = vertices[:-1]
            pos = rotate(pos, self.rng.random() * 2 * np.pi)
            # Do a random translation
            pos[:, 0] += self.rng.random()
            pos[:, 1] += self.rng.random()
            positions[i * n: (i + 1) * n] = pos
            i += 1

        self.setPositions(positions.reshape(numVertices * 2))