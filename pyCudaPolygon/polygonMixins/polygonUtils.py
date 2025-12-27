import numpy as np
from matplotlib import pyplot as plt

class Mixin():
    def meanPerimeter(x, y):
        dx = np.roll(x, -1) - x
        dy = np.roll(y, -1) - y
        return np.mean(np.sqrt(dx**2 + dy**2))

    def meanPerimeterSquared(x, y):
        dx = np.roll(x, -1) - x
        dy = np.roll(y, -1) - y
        return np.mean(dx**2 + dy**2)

    def area(x, y):
        n = x.size
        xC = np.concatenate((x, np.array([x[0]])))
        yC = np.concatenate((y, np.array([y[0]])))
        dx = xC[:-1] - xC[1:]
        sy = yC[1:] + yC[:-1]
        return np.sum(dx * sy) / 2

    def getShapeIndex(x, y):
        n = x.size
        return n * meanPerimeter(x, y) / np.sqrt(area(x, y))

    def g1(x, y, kappa):
        n = x.size
        return kappa * np.sqrt(area(x, y)) - n * meanPerimeter(x, y)

    def g2(x, y):
        n = x.size
        xC = np.concatenate((x, np.array([x[0]])))
        yC = np.concatenate((y, np.array([y[0]])))
        dx = xC[1:] - xC[:-1]
        dy = yC[1:] - yC[:-1]
        meanPerimeter = np.sum(np.sqrt(dx**2 + dy**2)) / n
        return np.sum(dx**2 + dy**2) / n - meanPerimeter**2

    def L(x, y, vx, vy, kappa, lambda1, lambda2):
        return np.sum((x - vx)**2 + (y - vy)**2 - lambda1 * g1(vx, vy, kappa) - lambda2 * g2(vx, vy))

    def dAx(x, y):
        nxty = np.roll(y, -1)
        prvy = np.roll(y,  1)
        return 0.5 * (nxty - prvy)

    def dAy(x, y):
        nxtx = np.roll(x, -1)
        prvx = np.roll(x,  1)
        return -0.5 * (nxtx - prvx)

    def dPx(x, y):
        n = x.size
        nxtx = np.roll(x, -1)
        prvx = np.roll(x,  1)
        nxty = np.roll(y, -1)
        prvy = np.roll(y,  1)

        dxn = x - nxtx
        dyn = y - nxty
        dxp = x - prvx
        dyp = y - prvy

        ln = np.sqrt(dxn**2 + dyn**2)
        lp = np.sqrt(dxp**2 + dyp**2)

        # numerical safety
        eps = 1e-12
        ln = np.maximum(ln, eps) / n
        lp = np.maximum(lp, eps) / n
        return dxn/ln + dxp/lp

    def dPy(x, y):
        n = x.size
        nxtx = np.roll(x, -1)
        prvx = np.roll(x,  1)
        nxty = np.roll(y, -1)
        prvy = np.roll(y,  1)

        dxn = x - nxtx
        dyn = y - nxty
        dxp = x - prvx
        dyp = y - prvy

        ln = np.sqrt(dxn**2 + dyn**2)
        lp = np.sqrt(dxp**2 + dyp**2)

        # numerical safety
        eps = 1e-12
        ln = np.maximum(ln, eps) / n
        lp = np.maximum(lp, eps) / n
        return dyn/ln + dyp/lp

    def dP2x(x, y):
        n = x.size
        nxtx = np.roll(x, -1)
        prvx = np.roll(x,  1)
        return 2 * (2 * x - nxtx - prvx) / n

    def dP2y(x, y):
        n = x.size
        nxty = np.roll(y, -1)
        prvy = np.roll(y,  1)
        return 2 * (2 * y - nxty - prvy) / n

    def dg1x(x, y, kappa):
        n = x.size
        A = area(x, y)
        diffAx = dAx(x, y)
        diffPx = dPx(x, y)
        pref = kappa / (2 * np.sqrt(A))
        return pref * diffAx - n * diffPx

    def dg1y(x, y, kappa):
        n = x.size
        A = area(x, y)
        diffAy = dAy(x, y)
        diffPy = dPy(x, y)
        pref = kappa / (2 * np.sqrt(A))
        return pref * diffAy - n * diffPy

    def dg2x(x, y):
        meanP = meanPerimeter(x, y)
        diffPx = dPx(x, y)
        diffP2x = dP2x(x, y)
        return diffP2x - 2 * meanP * diffPx

    def dg2y(x, y):
        meanP = meanPerimeter(x, y)
        diffPy = dPy(x, y)
        diffP2y = dP2y(x, y)
        return diffP2y - 2 * meanP * diffPy

    def dLx(x, y, vx, vy, kappa, lambda1, lambda2):
        return 2 * (vx - x) - lambda1 * dg1x(vx, vy, kappa) - lambda2 * dg2x(vx, vy)

    def dLy(x, y, vx, vy, kappa, lambda1, lambda2):
        return 2 * (vy - y) - lambda1 * dg1y(vx, vy, kappa) - lambda2 * dg2y(vx, vy)

    def dLl1(x, y, vx, vy, kappa, lambda1, lambda2):
        return -g1(vx, vy, kappa)

    def dLl2(x, y, vx, vy, kappa, lambda1, lambda2):
        return - g2(vx, vy)

    def ddAxy(x, y):
        n = x.size
        # Hessian of raw area
        H = np.zeros((n, n))
        for i in range(n):
            previ = (i + n - 1) % n
            nexti = (i + 1) % n
            H[i, nexti] += 1
            H[i, previ] -= 1
        return H / 2

    def ddPxx(x, y):
        n = x.size
        H = np.zeros((n, n))
        for i in range(n):
            previ = (i + n - 1) % n
            nexti = (i + 1) % n
            li = np.sqrt((x[i] - x[nexti])**2 + (y[i] - y[nexti])**2)
            lzpi = np.sqrt((x[previ] - x[i])**2 + (y[previ] - y[i])**2)
            H[i][i] += 1 / li
            H[i][nexti] -= 1 / li
            H[i][previ] -= 1 / lzpi
            H[i][i] += 1 / lzpi
            H[i][i] -= (x[i] - x[nexti]) / li**3
            H[i][nexti] += (x[i] - x[nexti]) / li**3
            H[i][previ] += (x[previ] - x[i]) / lzpi**3
            H[i][i] -= (x[previ] - x[i]) / lzpi**3
        return H / n

    def ddPyy(x, y):
        n = x.size
        H = np.zeros((n, n))
        for i in range(n):
            previ = (i + n - 1) % n
            nexti = (i + 1) % n
            li = np.sqrt((x[i] - x[nexti])**2 + (y[i] - y[nexti])**2)
            lzpi = np.sqrt((x[previ] - x[i])**2 + (y[previ] - y[i])**2)
            H[i][i] += 1 / li
            H[i][nexti] -= 1 / li
            H[i][previ] -= 1 / lzpi
            H[i][i] += 1 / lzpi
            H[i][i] -= (y[i] - y[nexti]) / li**3
            H[i][nexti] += (y[i] - y[nexti]) / li**3
            H[i][previ] += (y[previ] - y[i]) / lzpi**3
            H[i][i] -= (y[previ] - y[i]) / lzpi**3
        return H / n

    def ddPxy(x, y):
        n = x.size
        H = np.zeros((n, n))
        for i in range(n):
            previ = (i + n - 1) % n
            nexti = (i + 1) % n
            li = np.sqrt((x[i] - x[nexti])**2 + (y[i] - y[nexti])**2)
            lzpi = np.sqrt((x[previ] - x[i])**2 + (y[previ] - y[i])**2)
            H[i][i] -= (x[i] - x[nexti]) * (y[i] - y[nexti]) / li**3
            H[i][nexti] += (x[i] - x[nexti]) * (y[i] - y[nexti]) / li**3
            H[i][previ] += (x[previ] - x[i]) * (y[previ] - y[i]) / lzpi**3
            H[i][i] -= (x[previ] - x[i]) * (y[previ] - y[i]) / lzpi**3
        return H / n

    def ddP2xx(x, y):
        n = x.size
        H = np.zeros((n, n))
        for i in range(n):
            nexti = (i + 1) % n
            previ = (i + n - 1) % n
            H[i][i] += 2
            H[i][nexti] -= 1
            H[i][previ] -= 1
        return H * 2 / n

    def ddP2yy(x, y):
        n = x.size
        H = np.zeros((n, n))
        for i in range(n):
            nexti = (i + 1) % n
            previ = (i + n - 1) % n
            H[i][i] += 2
            H[i][nexti] -= 1
            H[i][previ] -= 1
        return H * 2 / n

    def ddg1xx(x, y, kappa):
        n = x.size
        return - n * ddPxx(x, y)

    def ddg1yy(x, y, kappa):
        n = x.size
        return - n * ddPyy(x, y)

    def ddg1xy(x, y, kappa):
        n = x.size
        A = area(x, y)
        pref = kappa / (2 * np.sqrt(A))
        return pref * ddAxy(x, y)

    def ddg2xx(x, y, kappa):
        meanP = meanPerimeter(x, y)
        diffPx = dPx(x, y)
        return ddP2xx(x, y) - 2 * diffPx * diffPx - 2 * meanP * ddPxx

    def ddg2yy(x, y, kappa):
        meanP = meanPerimeter(x, y)
        diffPy = dPy(x, y)
        return ddP2yy(x, y) - 2 * diffPy * diffPy - 2 * meanP * ddPyy

    def ddg2xy(x, y, kappa):
        meanP = meanPerimeter(x, y)
        diffPx = dPx(x, y)
        diffPy = dPy(x, y)
        return -2 * meanP * ddPxy(x, y) - 2 * diffPx * diffPy

    def generateRandomPolygon(self, n, kappa, numSteps = 10, tol = 1e-10):
        pts = np.random.rand(n, 2) - 0.5
        center = pts.mean(axis = 0)
        angles = np.arctan2(pts[:,1] - center[1],
                            pts[:,0] - center[0])
        x, y = pts[np.argsort(angles)].T
        vx = x + 0
        vy = y + 0
        # Actually, let's make a square and perturb it a bit
        x = np.array([-0.5, 0.5, 0.5, -0.5])
        y = np.array([-0.5, -0.5, 0.5, 0.5])
        vx = x + np.random.rand(4) / 100
        vy = y + np.random.rand(4) / 100
        l1 = np.random.rand() - 0.5
        l2 = np.random.rand() - 0.5
