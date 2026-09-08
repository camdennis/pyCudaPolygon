"""test_polyDeltaMode.py — sanity-check the per-polygon uniform-delta mode.

Each polygon gets its own radius delta(s); targets are frozen at enable-time
to dA*(s) = delta_initial^2 * S_initial(s), where S(s) = sum_k [tan(phi_k/2)
- phi_k/2]. Per-step updates restore delta(s) = sqrt(dA*(s) / S(s)).

Checks:
  1. After enable, the closed-form delta(s) equals the global scalar delta
     (because targets were just set from that scalar).
  2. The targets are positive and finite.
  3. After perturbing positions and re-updating, dA(s) = delta(s)^2 * S(s)
     matches the original target within floating-point tolerance.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp


def wrap(d):
    return d - np.round(d)


def compute_S(positions, start, n):
    S = 0.0
    pts = positions.reshape(-1, 2)[start:start + n]
    for k in range(n):
        kp = (k - 1) % n
        kn = (k + 1) % n
        u = wrap(pts[k] - pts[kp])
        v = wrap(pts[kn] - pts[k])
        ul = np.linalg.norm(u)
        vl = np.linalg.norm(v)
        if ul < 1e-14 or vl < 1e-14:
            continue
        cr = (u[0] * v[1] - u[1] * v[0]) / (ul * vl)
        dt = (u[0] * v[0] + u[1] * v[1]) / (ul * vl)
        phi = np.arctan2(cr, dt)
        if phi <= 1e-14 or phi >= np.pi - 1e-14:
            continue
        S += np.tan(0.5 * phi) - 0.5 * phi
    return S


def build(N=8, n=12, seed=3, kappa=3.7, rho=0.05):
    m = pcp.model(size=N * n, seed=seed)
    m.setModelEnum("rounded")
    m.generateRandomPolygons(N, n)
    m.setBiPerimeters(kappa)
    m.setDelta(rho)
    m.setStiffness(1.0)
    m.setCompressibility(1.0)
    m.initializeNeighborBall()
    m.updateForceEnergy()
    return m


print("=== per-polygon uniform-delta mode test ===")
m = build()
print(f"numPolygons = {m.getNumPolygons()}, scalar delta = {m.getDelta():.6f}")

assert not m.isPerPolygonDeltaEnabled(), "should be off before enable"

m.enablePerPolygonDelta()
assert m.isPerPolygonDeltaEnabled(), "should be on after enable"

polyDelta = np.array(m.getPolyDelta())
targets   = np.array(m.getPolyDeltaTargets())
scalar    = m.getDelta()

print(f"\nimmediately after enable (delta(s) should equal scalar = {scalar}):")
print(f"  polyDelta min/max = {polyDelta.min():.6e} / {polyDelta.max():.6e}")
print(f"  targets  min/max = {targets.min():.6e} / {targets.max():.6e}")

assert np.allclose(polyDelta, scalar, rtol=1e-12, atol=1e-14), \
    f"delta(s) should equal scalar at enable time; got max abs dev {np.abs(polyDelta - scalar).max():.3e}"
assert np.all(targets > 0), "targets should be positive"
assert np.all(np.isfinite(targets)), "targets should be finite"

# Verify targets against a host-side recomputation: dA*(s) = scalar^2 * S(s).
positions = np.array(m.getPositions())
startIndices = np.array(m.getStartIndices())
N = m.getNumPolygons()
S_host = np.zeros(N)
for s in range(N):
    start = startIndices[s]
    n     = startIndices[s + 1] - start
    S_host[s] = compute_S(positions, start, n)

expected_targets = scalar * scalar * S_host
rel = np.abs(targets - expected_targets) / np.maximum(np.abs(expected_targets), 1e-30)
print(f"\nhost vs device target rel err: max = {rel.max():.3e}")
assert rel.max() < 1e-10, f"target mismatch: max rel err {rel.max():.3e}"

# Perturb positions slightly, re-update, and check the area invariant holds.
pos_new = positions.copy()
rng = np.random.default_rng(1)
pos_new += 1e-3 * rng.standard_normal(pos_new.shape)
m.setPositions(pos_new.tolist())
m.updatePolyDelta()

polyDelta_new = np.array(m.getPolyDelta())
S_new = np.zeros(N)
for s in range(N):
    start = startIndices[s]
    n     = startIndices[s + 1] - start
    S_new[s] = compute_S(pos_new, start, n)

dA_new = polyDelta_new * polyDelta_new * S_new
rel_invariant = np.abs(dA_new - targets) / np.maximum(np.abs(targets), 1e-30)
print(f"\narea-invariant after perturb: max rel err = {rel_invariant.max():.3e}")
print(f"  polyDelta min/max (post) = {polyDelta_new.min():.6e} / {polyDelta_new.max():.6e}")

assert rel_invariant.max() < 1e-10, \
    f"dA(s) drifted from target: max rel err {rel_invariant.max():.3e}"

m.disablePerPolygonDelta()
assert not m.isPerPolygonDeltaEnabled()

print("\nper-polygon delta mode OK.")
