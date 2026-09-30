if __package__:
    # Imported as part of the pyCudaPolygon package: load the extension that
    # CMake copies into this directory, so the package always uses its own build.
    from . import libpyCudaPolygon as lpcp
    from . import enums
else:
    # Imported as a top-level module, with this directory on sys.path.
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
from tqdm import tqdm
import warnings

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

class model(*mixins.values()):

    # initializers

    def __init__(self, size = 0, seed = None, modelType = "softBody", stiffness = 0, compressibility = 0):
        self._impl = lpcp.Model(size)
        self.setModelEnum(modelType)
        self.setStiffness(stiffness)
        self.setCompressibility(compressibility)
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def initializeNeighborCells(self):
        # Just in case, set the maximum edge lengths to be large
        self.setMaxEdgeLength()
        self._impl.initializeNeighborCells()
    
    def initForceEnergy(self):
        t = self.getModelEnum()
        if t == "normal":
            self.initializeNeighborCells()
            self.updateNeighborCells()
            self.updateNeighbors()
            self.updateOutersections()
