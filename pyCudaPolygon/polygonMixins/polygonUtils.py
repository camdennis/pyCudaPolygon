import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import circulant

class Mixin():
    def __init__(self, rng = None):
        # random number generator fallback
        self.rng = rng if rng is not None else np.random.default_rng()
        
    def getRandomConvexPolygon(self, l):
        l = np.asarray(l, dtype=float)
        n = l.size

        # Polygon inequality
        if np.max(l) > np.sum(l) - np.max(l):
            raise ValueError("Side lengths violate polygon inequality.")

        # Random turning angles summing to 2π
        turns = self.rng.dirichlet(np.ones(n)) * 2 * np.pi
        turns = np.pi - turns
        return turns[1:]

    def getPhi(self, theta):
        phi = np.zeros(theta.size + 1)
        for i in range(1, theta.size + 1):
            phi[i] = phi[i - 1] + theta[i - 1]
        phi = np.arange(theta.size + 1) * np.pi - phi
        return phi

    def getVertices(self, theta, l):
        phi = self.getPhi(theta)
        n = theta.size + 1
        v = np.zeros((n + 1, 2))
        for i in range(1, n + 1):
            v[i] = v[i - 1] + l[i - 1] * np.array([np.cos(phi[i - 1]), np.sin(phi[i - 1])])
        return v

    def area(self, theta, l):
        n = l.size
        ct = np.concatenate(([0], np.cumsum(theta)))
        sol = 0
        for m in range(1, n):
            for k in range(1, m):
                sol += l[m - 1] * l[k - 1] * (-1)**(m - k + 1) * np.sin(ct[m - 1] - ct[k - 1])
        return sol / 2

    def getShapeIndex(self, theta, l):
        A = self.area(theta, l)
        return np.sum(l) / np.sqrt(A)

    def dArea(self, theta, l):
        n = l.size
        ct = np.concatenate(([0], np.cumsum(theta)))
        sol = np.zeros(n - 1)
        for a in range(1, n):
            for m in range(a + 1, n):
                for k in range(1, a + 1):
                    sol[a - 1] += l[m - 1] * l[k - 1] * (-1)**(m - k + 1) * np.cos(ct[m - 1] - ct[k - 1])
        return sol / 2

    def ddArea(self, theta, l):
        n = l.size
        ct = np.concatenate(([0], np.cumsum(theta)))
        sol = np.zeros((n - 1, n - 1))
        for a in range(1, n):
            for b in range(1, n):
                cp = a
                cm = b
                if (cm > cp):
                    (cp, cm) = (cm, cp)
                for m in range(cp + 1, n):
                    for k in range(1, cm + 1):
                        sol[a - 1][b - 1] += l[m - 1] * l[k - 1] * (-1)**(m - k + 1) * np.sin(ct[m - 1] - ct[k - 1])
        return -sol / 2

    def vn(self, theta, l):
        phi = self.getPhi(theta)
        return np.dot(l, np.array([np.cos(phi), np.sin(phi)]).T)

    def dvn(self, theta, l):
        vertices = self.getVertices(theta, l)
        vf = vertices[-1]
        dv = vf - vertices[1:-1]
        mat = np.array([[0, 1], [-1, 0]])
        return np.dot(mat, dv.T).T

    def ddvn(self, theta, l):
        n = l.size
        vertices = self.getVertices(theta, l)
        vf = vertices[-1]
        dv = vf - vertices[1:-1]
        sol = np.zeros((n - 1, n - 1, 2))
        for a in range(n - 1):
            for b in range(n - 1):
                sol[a][b] = -vf + vertices[1 + np.max([a, b])]
        return sol

    def L(self, theta, l, theta0, a0, la, lam):
        perimeter = np.sum(l)
        vf = self.vn(theta, l)
        A = self.area(theta, l)
        return np.sum((theta - theta0)**2) - la * (A - a0) / perimeter**2 - np.dot(lam, vf) / perimeter

    def dL(self, theta, l, theta0, a0, la, lam):
        perimeter = np.sum(l)
        vf = self.vn(theta, l)
        A = self.area(theta, l)
        dLTheta = 2 * (theta - theta0) - la * self.dArea(theta, l) / perimeter**2 - np.dot(lam, self.dvn(theta, l).T).T / perimeter
        dLLa = -(A - a0) / perimeter**2
        dLLam = -vf / perimeter
        return np.concatenate((dLTheta, [dLLa], dLLam))

    def ddL(self, theta, l, theta0, a0, la, lam):
        perimeter = np.sum(l)
        n = l.size
        vf = self.vn(theta, l)
        A = self.area(theta, l)
        dA = self.dArea(theta, l)
        ddA = self.ddArea(theta, l)
        dvf = self.dvn(theta, l)
        ddvf = self.ddvn(theta, l)
        # Here's all the pieces

        dLThetaTheta = 2 * np.identity(n - 1) - la * ddA / perimeter**2 - np.tensordot(ddvf, lam, axes=([2],[0])) / perimeter

        dLThetaLa = -dA / perimeter**2
        dLThetaLam = -dvf / perimeter

        sol = np.zeros((n + 2, n + 2))
        sol[:n - 1, :n - 1] = dLThetaTheta
        sol[:n - 1, n - 1: n] = np.array([dLThetaLa]).T
        sol[n - 1: n, :n - 1] = [dLThetaLa]
        sol[:n - 1, n:] = dLThetaLam
        sol[n:, :n - 1] = dLThetaLam.T
        return sol

    def takeStep(self, theta, l, theta0, a0, la, lam, maxSteps=100, tol=1e-10):
        n = l.size
        counts = 0
        for _ in range(maxSteps):
            f = self.dL(theta, l, theta0, a0, la, lam)
            Df = self.ddL(theta, l, theta0, a0, la, lam)
            try:
                s = np.linalg.solve(Df, -f)
            except np.linalg.LinAlgError:
                # singular Jacobian — stop iteration
                raise Exception("Singular Jacobian")
            # backtracking line search on step length
            a = 1.0
            fNorm = np.linalg.norm(f)
            while a > 1e-8:
                thetaNew = theta + a * s[:n - 1]
                laNew = la + a * s[n - 1]
                lamNew = lam + a * s[n:]
                fNew = self.dL(thetaNew, l, theta0, a0, laNew, lamNew)
                if np.linalg.norm(fNew) < fNorm:
                    break
                a *= 0.5

            counts += 1
#            print(self.vn(thetaNew, l), self.area(thetaNew, l))
            theta, la, lam = thetaNew, laNew, lamNew
            yield theta, la, lam

    def generateRandomPolygon(self, l, kappa, theta0 = None, numSteps = 10, tol = 1e-10):
        if theta0 is None:
            theta0 = self.getRandomConvexPolygon(l)
        theta = theta0.copy()
        a0 = (np.sum(l) / kappa)**2
        la = self.rng.random()
        lam = self.rng.random(2)
        aStart = self.area(theta, l)
        i = 0
        for theta, la, lam in self.takeStep(theta, l, theta0, a0, la, lam, maxSteps = numSteps, tol = tol):
            if np.abs(self.getShapeIndex(theta, l) - kappa) < tol and np.max(np.abs(self.vn(theta, l))) < tol:
                break
            i += 1
        if i == numSteps:
            print("Warning: It seems that this did not converge after " + str(i) + " steps")
        return theta0, theta
