"""diagnostic: are the candidate lists ordered identically between cells
and balls paths? Order matters because the rounded kernel caps unique
shapes at maxProc = polygonSize."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp

N, n, seed = 32, 16, 7
kappa, rho = 3.7, 0.01

def build(neighborType):
    m = pcp.model(size=N*n, seed=seed)
    m.setModelEnum("rounded")
    m.generateRandomPolygons(N, n)
    m.setBiPerimeters(kappa)
    m.setDelta(rho)
    m.setStiffness(1.0)
    m.setCompressibility(1.0)
    m.setNeighborType(neighborType)
    m.setSearchFactor(3.0)
    m.initializeNeighborCells()
    m.updateNeighborCells()
    if neighborType == "balls":
        m.initializeNeighborBall()
        m.updateNeighborBall()
    return m

m_c = build("cells")
m_b = build("balls")
bm_c = m_c.getBallMaxNeighbors()
bm_b = m_b.getBallMaxNeighbors()
ball_c = np.asarray(m_c.getBallNeighbors(), dtype=np.int64).reshape(-1, bm_c)
ball_b = np.asarray(m_b.getBallNeighbors(), dtype=np.int64).reshape(-1, bm_b)
num_c = np.asarray(m_c.getNumBallNeighbors(), dtype=np.int64)
num_b = np.asarray(m_b.getNumBallNeighbors(), dtype=np.int64)
shapeId = np.asarray(m_c.getShapeId(), dtype=np.int64)

# For each vertex, compute the order in which unique SHAPES are first encountered.
def first_encounter_shapes(ball, num, i):
    seen = set()
    order = []
    for k in range(num[i]):
        s = shapeId[ball[i, k]]
        if s in seen:
            continue
        seen.add(s)
        order.append(int(s))
    return order

n_diff = 0
first_diff_v = -1
for i in range(N*n):
    a = first_encounter_shapes(ball_c, num_c, i)
    b = first_encounter_shapes(ball_b, num_b, i)
    if a != b:
        n_diff += 1
        if first_diff_v < 0:
            first_diff_v = i

print(f"vertices where first-encounter SHAPE order differs: {n_diff} / {N*n}")
if first_diff_v >= 0:
    a = first_encounter_shapes(ball_c, num_c, first_diff_v)
    b = first_encounter_shapes(ball_b, num_b, first_diff_v)
    print(f"first divergent vertex {first_diff_v}:")
    print(f"  cells order: {a[:20]}{' ...' if len(a)>20 else ''}")
    print(f"  balls order: {b[:20]}{' ...' if len(b)>20 else ''}")

# Also check raw candidate-vertex order
n_raw = 0
for i in range(N*n):
    if not np.array_equal(ball_c[i, :num_c[i]], ball_b[i, :num_b[i]]):
        n_raw += 1
print(f"vertices where raw candidate vertex order differs: {n_raw} / {N*n}")
