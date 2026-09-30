"""Ground-truth estimators for the contact energy.
"""

import numpy as np
from matplotlib import pyplot as plt

def boundaryDistance(point, loop):
    """
    Distance from `point` to the boundary of `loop`, or 0.0 if the point is
    outside `loop`.

    Args:
        point: (2,) array.
        loop:  (n, 2) array, CCW.

    Returns:
        float. Zero when the point lies outside `loop`; otherwise the distance
        to the nearest point of the boundary.

    Example:
        loop = unit square [(0,0),(1,0),(1,1),(0,1)]
        boundaryDistance([0.25, 0.5],  loop) -> 0.25
        boundaryDistance([0.5,  0.5],  loop) -> 0.5
        boundaryDistance([1.5,  0.5],  loop) -> 0.0
    """
    bd = np.inf
    n = len(loop)
    # Check if it is inside:
    if (not inside(point, loop)):
        return 0.0
    for i in range(n):
        edge = loop[(i + 1) % n] - loop[i]
        l = np.linalg.norm(edge)
        normal = np.array([edge[1], -edge[0]]) / l
        # Check if you project onto the loop
        t = np.dot(point - loop[i], edge) / l / l
        # clamp t and rebuild
        if (t > 1):
            dist = np.linalg.norm(loop[(i + 1) % n] - point)
        elif (t < 0):
            dist = np.linalg.norm(loop[i] - point)
        else:
            dist = np.dot(normal, loop[i] - point)
            if (dist < 0):
                continue
        bd = min(bd, dist)
    if (bd == np.inf):
        return 0.0
    return bd

def inside(point, loop):
    n = len(loop)
    insideFlag = False
    for i in range(n):
        inext = (i + n + 1) % n
        if (((loop[i, 1] > point[1]) != (loop[inext, 1] > point[1])) and 
            (point[0] < (loop[inext, 0] - loop[i, 0]) * (point[1] - loop[i, 1]) / (loop[inext, 1] - loop[i, 1]) + loop[i, 0])):
            insideFlag = not insideFlag
    return insideFlag

def edgeEnergy(loopA, i, loopB, k = 1.0, w = 3, samples = 1_000_000, rng = None):
    """
    Monte-Carlo estimate of the energy contributed by edge `i` of `loopA`
    against the interior of `loopB`.

        E_i = |e_i| * (k/w) * mean over t ~ U(0,1) of d_B(x(t))^w
        x(t) = v_i + t * e_i

    Sample t uniformly on [0, 1]. Evaluate d_B by brute force. Where x(t) lies
    outside `loopB`, the integrand is zero.

    Returns:
        (energy, standardError, insideFraction)
    """
    n = len(loopA)
    vi = loopA[i]
    ei = loopA[(i + 1) % n] - vi
    li = np.sqrt(np.dot(ei, ei))
    sol = 0
    sol2 = 0
    count = 0
    if rng is None:
        tValues = np.random.rand(samples)
    else:
        tValues = rng.random(samples)
    for t in tValues:
        point = vi + ei * t
        bd = boundaryDistance(point, loopB)**w
        if (bd == 0):
            count += 1
            continue
        sol += bd
        sol2 += bd**2
    sol /= samples
    sol2 /= samples
    count /= samples
    factor = li * k / w
    sol *= factor
    sol2 *= factor * factor
    return sol, np.sqrt(sol2 - sol * sol) / np.sqrt(samples), 1 - count

def pairEnergy(loopA, loopB, k = 1.0, w = 3, samples = 1_000_000, rng = None):
    """
    Monte-Carlo estimate of the one-directional energy: the whole boundary of
    `loopA` against the interior of `loopB`.

        E(dA -> B) = sum over edges i of edgeEnergy(loopA, i, loopB)

    Returns:
        (energy, standardError, insideFraction (np array))

    Example 1 -- self-check, exact closed form available:
        B = unit square         [(0,0),(1,0),(1,1),(0,1)]
        A = the square          [(0.25,0.25),(0.75,0.25),(0.75,0.75),(0.25,0.75)]
        k = 1, w = 3

            E(dA -> B) = 1/96 = 0.010416666666...
    """
    sol = 0
    sterr = 0
    insideFraction = []
    for i in range(len(loopA)):
        m, s, c = edgeEnergy(loopA, i, loopB, k = k, w = w, samples = samples, rng = rng)
        insideFraction.append(c)
        # The standard error is the rms
        sterr += s**2
        sol += m
    return sol, np.sqrt(sterr), np.array(insideFraction)

def symmetricPairEnergy(P, Q):
    return 0.5 * (pairEnergy(P, Q) + pairEnergy(Q, P))
