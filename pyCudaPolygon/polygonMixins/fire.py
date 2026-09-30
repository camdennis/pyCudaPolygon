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

    def minimizeFIRE(
        self, 
        maxForceThreshold = 1e-12, 
        maxSteps = 2**31 - 1, 
        dtInit = 0.001, 
        dtMax = 0.1,
        alphaStart = 0.1,
        fAlpha = 0.99,
        fInc = 1.1,
        fDec = 0.5,
        nMin = 5
        ):
        self.initForceEnergy()
        self.updateForceEnergy()
        return self._impl.minimizeFIRE(maxForceThreshold, dtInit, maxSteps, dtMax, alphaStart, fAlpha, fInc, fDec, nMin)

