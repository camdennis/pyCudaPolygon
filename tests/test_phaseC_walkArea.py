"""test_phaseC_walkArea.py — per-feature inside chord-area walk.

After Phase B/C-sort, each A-feature has a contiguous slice of crossings in
paramA order. The walk kernel produces, per A-feature, the sum of chord-areas
over "inside" sub-segments: pairs of consecutive same-shapeB crossings at
even/odd rank, anchored at startIndices[min(sA, sB)].

This test replicates the same logic on the host and asserts agreement to
floating-point tolerance.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp


def wrap_box(d):
    # Box convention: positions in [0,1)^2, periodic with period 1. wrap2 maps
    # offsets into (-0.5, 0.5].
    return d - np.round(d)


def chord_area_host(p1, p2, anchor):
    # Mirrors integrate::chordArea: wrap2 each leg, then 1/2 (r1+r2) x (r2-r1).
    r1 = wrap_box(p1 - anchor)
    r2 = wrap_box(p2 - anchor)
    d  = wrap_box(r2 - r1)
    return 0.5 * ((r1[0] + r2[0]) * d[1] - (r1[1] + r2[1]) * d[0])


def host_walk_area(uniqueFA, lengths, fA_stream, fB_stream, X_flat,
                    shapeId, startIndices, positions):
    """Reproduce phaseCWalkAreaPerFeatureKernel on the host."""
    starts = np.zeros(len(uniqueFA), dtype=np.int64)
    starts[1:] = np.cumsum(lengths[:-1])
    out = np.zeros(len(uniqueFA), dtype=np.float64)
    for fIdx, (fA, n, s) in enumerate(zip(uniqueFA, lengths, starts)):
        vA = int(fA) >> 2
        sA = int(shapeId[vA])
        chordSum = 0.0
        for k in range(1, int(n)):
            fB_k = fB_stream[s + k]
            vBk  = int(fB_k) >> 2
            sB_k = int(shapeId[vBk])
            # Count prior same-shape crossings, remember most recent.
            rank = 0
            prev = -1
            for j in range(k):
                fB_j = fB_stream[s + j]
                sB_j = int(shapeId[int(fB_j) >> 2])
                if sB_j == sB_k:
                    rank += 1
                    prev = j
            if (rank & 1) == 1 and prev >= 0:
                sRef = sA if sA < sB_k else sB_k
                anchor_idx = int(startIndices[sRef])
                anchor = positions[anchor_idx]
                P0 = X_flat[s + prev]
                P1 = X_flat[s + k]
                chordSum += chord_area_host(P0, P1, anchor)
        out[fIdx] = chordSum
    return out


def build(N=12, n=14, seed=5, kappa=3.7, rho=0.05):
    m = pcp.model(size=N * n, seed=seed)
    m.setModelEnum("rounded")
    m.generateRandomPolygons(N, n)
    m.setBiPerimeters(kappa)
    m.setDelta(rho)
    m.setStiffness(1.0)
    m.setCompressibility(1.0)
    m.initializeNeighborBall()
    m.updateForceEnergy()
    m.updateNeighborBall()
    return m


def check(m, label, fs=2):
    print(f"\n=== {label} (fs={fs}) ===")
    nPairs = m.runFeaturePairPhaseA(fs)
    nCross = m.runFeaturePairPhaseB(m.getDelta())
    nFA    = m.runFeaturePairPhaseC_sort()
    print(f"  pairs / crossings / unique fA: {nPairs} / {nCross} / {nFA}")
    if nCross == 0 or nFA == 0:
        print("  no crossings, skipping")
        return

    device_area = np.array(m.runFeaturePairPhaseC_walkArea())

    uniqueFA = np.array(m.getFeatureUniqueFA(), dtype=np.uint32)
    lengths  = np.array(m.getFeatureLengths(),  dtype=np.uint32)
    fA_stream = np.array(m.getSortedCrossingFA(), dtype=np.uint32)
    fB_stream = np.array(m.getSortedCrossingFB(), dtype=np.uint32)
    X_flat    = np.array(m.getSortedCrossingX(), dtype=np.float64).reshape(-1, 2)
    shapeId   = np.array(m.getShapeId(), dtype=np.int64)
    startIndices = np.array(m.getStartIndices(), dtype=np.int64)
    positions = np.array(m.getPositions(), dtype=np.float64).reshape(-1, 2)

    host_area = host_walk_area(uniqueFA, lengths, fA_stream, fB_stream, X_flat,
                                shapeId, startIndices, positions)

    diff = np.abs(device_area - host_area)
    scale = np.maximum(np.abs(host_area), 1e-30)
    rel = diff / scale
    print(f"  device sum: {device_area.sum():.6e}    host sum: {host_area.sum():.6e}")
    print(f"  per-feature max abs diff = {diff.max():.3e}")
    print(f"  per-feature max rel diff = {rel.max():.3e}")
    assert diff.max() < 1e-10, f"per-feature mismatch: max abs {diff.max():.3e}"
    nz = np.count_nonzero(device_area)
    print(f"  A-features with nonzero chord-area contribution: {nz} / {nFA}")
    print(f"  OK")


m = build()
check(m, "rounded N=12 n=14 fs=2", fs=2)
check(m, "rounded N=12 n=14 fs=1", fs=1)

m2 = build(N=20, n=12, seed=11, rho=0.06)
check(m2, "rounded N=20 n=12 fs=2", fs=2)

print("\nPhase C walk-area device vs host: all matches.")
