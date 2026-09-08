"""test_phaseC_sort.py — sort + run-length-encode of Phase B output.

After Phase B emits an unordered flat list of crossings, Phase C's first job
is to sort them by (fA, paramA) and run-length-encode the fA stream so each
A-feature's slice is contiguous. The actual per-feature walk lands in a
later installment.

Checks:
  1. After sort, the fA stream is non-decreasing.
  2. Within each fA-slice, paramA is non-decreasing.
  3. RLE: unique fAs are strictly increasing.
  4. RLE: lengths sum to the total crossing count.
  5. RLE: lengths agree with raw slice lengths.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp


def build(N=16, n=16, seed=7, kappa=3.7, rho=0.05):
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


def check_phaseC(m, label, fs=2):
    print(f"\n=== {label} (fs={fs}) ===")
    nPairs = m.runFeaturePairPhaseA(fs)
    nCross = m.runFeaturePairPhaseB(m.getDelta())
    nFA    = m.runFeaturePairPhaseC_sort()
    print(f"  pairs / crossings / unique fA: {nPairs} / {nCross} / {nFA}")

    if nCross == 0:
        print("  no crossings, nothing to check")
        return

    fA     = np.array(m.getSortedCrossingFA(),     dtype=np.uint32)
    paramA = np.array(m.getSortedCrossingParamA(), dtype=np.float32)
    uFA    = np.array(m.getFeatureUniqueFA(),      dtype=np.uint32)
    lens   = np.array(m.getFeatureLengths(),       dtype=np.uint32)

    # 1) fA non-decreasing.
    df = np.diff(fA.astype(np.int64))
    assert (df >= 0).all(), f"fA stream not sorted: min diff = {df.min()}"

    # 2) within each fA-slice, paramA non-decreasing.
    offset = 0
    for k in range(nFA):
        L = int(lens[k])
        slc = paramA[offset:offset + L]
        ds = np.diff(slc)
        assert (ds >= -1e-7).all(), (
            f"paramA not sorted in slice fA={int(uFA[k])} (offset={offset}, L={L}): "
            f"min ds = {ds.min():.3e}")
        assert (fA[offset:offset + L] == uFA[k]).all(), (
            f"fA slice mismatch: expected all == {int(uFA[k])} at offset {offset}")
        offset += L
    assert offset == nCross, f"slice lengths sum {offset} != nCross {nCross}"

    # 3) unique fAs strictly increasing.
    dq = np.diff(uFA.astype(np.int64))
    assert (dq > 0).all(), f"unique fA stream not strictly increasing: min = {dq.min()}"

    # 4) lengths positive.
    assert (lens > 0).all(), f"some run length is zero or negative: min = {lens.min()}"

    print(f"  fA range:     [{fA.min()}, {fA.max()}]")
    print(f"  paramA range: [{paramA.min():.4f}, {paramA.max():.4f}]")
    print(f"  per-feature length: min={lens.min()}, max={lens.max()}, mean={lens.mean():.2f}")
    print(f"  OK")


m = build()
check_phaseC(m, "rounded N=16 n=16 fs=2", fs=2)
check_phaseC(m, "rounded N=16 n=16 fs=1", fs=1)

# Also try a denser config to stress.
m2 = build(N=24, n=20, seed=11, rho=0.06)
check_phaseC(m2, "rounded N=24 n=20 fs=2", fs=2)

print("\nPhase C sort + RLE checks passed.")
