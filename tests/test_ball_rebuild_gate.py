"""test_ball_rebuild_gate.py — verify the Verlet rebuild gate inside
updateNeighborBall(). Observes Model::ballRebuildCount directly so the test
works regardless of whether the ball list is sparse or all-to-all.

Protocol (repeated for softBody and rounded):
  1. Build initial list (count goes 0 -> 1).
  2. Sub-skin shift -> count stays at 1 (gate suppressed rebuild).
  3. Over-skin shift -> count goes to 2 (gate forced rebuild).
  4. Many accumulating sub-skin shifts -> count increments only when motion
     since the last rebuild crosses skin/2. Also verifies that ball-list
     contents always match a brute-force ground truth at each step (so the
     gate is never *too* lazy)."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp


def wrap2(d):
    return d - np.round(d)


def ground_truth_counts(pos, shapeId, radius):
    N = pos.shape[0]
    counts = np.zeros(N, dtype=np.int64)
    for i in range(N):
        d = wrap2(pos - pos[i])
        r2 = (d * d).sum(axis=1)
        mask = (np.arange(N) != i) & (shapeId != shapeId[i]) & (r2 <= radius * radius)
        counts[i] = mask.sum()
    return counts


def run_protocol(modelType):
    print(f"\n========== modelType = {modelType} ==========")
    N, n, seed = 8, 6, 7
    kappa, rho = 3.7, 0.001
    m = pcp.model(size=N*n, seed=seed)
    m.setModelEnum(modelType)
    m.generateRandomPolygons(N, n)
    m.setBiPerimeters(kappa)
    m.setDelta(rho)
    m.setStiffness(1.0)
    m.setCompressibility(1.0)
    m.setNeighborType("balls")
    m.initializeNeighborCells()
    m.updateNeighborCells()
    m.initializeNeighborBall()

    assert m.getBallRebuildCount() == 0, "no rebuilds before first updateNeighborBall"

    m.updateNeighborBall()                           # build #1
    n_after_build1 = m.getBallRebuildCount()
    print(f"  after first updateNeighborBall: rebuildCount = {n_after_build1}")
    assert n_after_build1 == 1, "first call should perform a rebuild"

    mel = m.getMaxEdgeLength()
    polygonSize = m.getNumVertices() // m.getNumPolygons()
    R    = m.getSearchFactor() * mel
    # Match the C++ trueCutoff: max(2*L, 2*polygon_diameter).
    trueCutoff = max(2.0 * mel, 2.0 * polygonSize * mel / 3.14159265358979323846)
    skin = R - trueCutoff
    print(f"  maxEdgeLength={mel:.4e}  R={R:.4e}  skin={skin:.4e}")
    assert skin > 0, "test assumes positive skin"

    pos0  = np.asarray(m.getPositions(), dtype=np.float64).reshape(-1, 2).copy()
    shape = np.asarray(m.getShapeId(), dtype=np.int64)
    cnt0  = np.asarray(m.getNumBallNeighbors(), dtype=np.int64).copy()
    gt0   = ground_truth_counts(pos0, shape, R)
    assert np.array_equal(cnt0, gt0), \
        f"initial ball list != ground truth\n device={cnt0}\n ground={gt0}"

    # --- sub-skin shift: gate should suppress rebuild ---
    eps = 0.1 * skin / 2.0
    pos1 = pos0.copy()
    pos1[0, 0] += eps
    m.setPositions(pos1.flatten())
    m.updateNeighborBall()
    n_after_small = m.getBallRebuildCount()
    print(f"  tiny shift eps={eps:.3e} (skin/2={skin/2:.3e}): rebuildCount = {n_after_small}")
    assert n_after_small == n_after_build1, \
        f"gate FAILED: rebuild fired for sub-skin shift (count {n_after_build1} -> {n_after_small})"

    # --- over-skin shift: gate must rebuild ---
    # Cap shift below 0.5 so the periodic minimum-image wrap doesn't fold the
    # shift back to a small displacement (the gate uses MIC-wrapped distance).
    big = min(3.0 * skin, 0.45)
    pos2 = pos1.copy()
    pos2[0, 0] += big
    m.setPositions(pos2.flatten())
    m.updateNeighborBall()
    n_after_big = m.getBallRebuildCount()
    print(f"  large shift big={big:.3e}: rebuildCount = {n_after_big}")
    assert n_after_big == n_after_small + 1, \
        f"gate FAILED: large shift did not force rebuild (count stayed at {n_after_big})"
    cnt2 = np.asarray(m.getNumBallNeighbors(), dtype=np.int64)
    gt2  = ground_truth_counts(pos2, shape, R)
    assert np.array_equal(cnt2, gt2), \
        f"rebuilt counts != ground truth\n device={cnt2}\n ground={gt2}"

    # --- accumulating sub-skin shifts: rebuild must fire before list goes stale ---
    pos_acc = pos2.copy()
    step    = 0.3 * skin / 2.0
    n_steps = 60
    rebuilds_at_start = m.getBallRebuildCount()
    for k in range(1, n_steps + 1):
        pos_acc[0, 0] += step
        m.setPositions(pos_acc.flatten())
        m.updateNeighborBall()
        cnt_now = np.asarray(m.getNumBallNeighbors(), dtype=np.int64)
        gt_now  = ground_truth_counts(pos_acc, shape, R)
        if not np.array_equal(cnt_now, gt_now):
            raise AssertionError(
                f"step {k}: stale counts -- gate is too lazy. device={cnt_now}  ground={gt_now}")
    n_extra_rebuilds = m.getBallRebuildCount() - rebuilds_at_start
    # Each rebuild starts fresh; the gate fires on the *first* step whose
    # cumulative displacement crosses skin/2. So one rebuild cycle holds
    # floor(skin/2 / step) sub-skin steps + 1 crossing step.
    steps_per_cycle = int(np.floor((skin / 2.0) / step)) + 1
    expected = n_steps // steps_per_cycle
    print(f"  {n_steps} sub-skin steps of {step:.3e}: triggered {n_extra_rebuilds} rebuilds "
          f"(expected ~ {expected}, steps_per_cycle = {steps_per_cycle})")
    assert abs(n_extra_rebuilds - expected) <= 1, \
        f"rebuild count {n_extra_rebuilds} too far from expected {expected}"

    print(f"  ==> rebuild gate OK for {modelType}")


run_protocol("softBody")
run_protocol("rounded")

print("\nREBUILD GATE OK for softBody and rounded.")
