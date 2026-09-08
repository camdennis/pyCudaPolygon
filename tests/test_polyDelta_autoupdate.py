"""test_polyDelta_autoupdate.py — verify that updateForceEnergy refreshes
polyDelta_d when per-polygon mode is enabled (no manual updatePolyDelta
required).

Procedure:
  1. Enable per-poly mode. At freeze, polyDelta(s) = scalar for every s.
  2. Perturb positions and call updateForceEnergy (NOT updatePolyDelta).
  3. polyDelta should reflect the new positions via the closed form.
  4. As a control: with per-poly mode DISABLED, updateForceEnergy should NOT
     touch polyDelta_d.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
import numpy as np
import pyCudaPolygon as pcp


def build(delta=0.04):
    m = pcp.model(size=8 * 12, seed=4)
    m.setModelEnum("rounded")
    m.generateRandomPolygons(8, 12)
    m.setBiPerimeters(3.0)
    m.setDelta(delta)
    m.setStiffness(1.0)
    m.setCompressibility(1.0)
    m.initializeNeighborBall()
    m.updateForceEnergy()
    return m


DELTA = 0.04
m = build(delta=DELTA)
m.enablePerPolygonDelta()
P = m.getNumPolygons()

# At freeze, polyDelta = scalar everywhere.
poly_d0 = np.array(m.getPolyDelta())
assert np.allclose(poly_d0, DELTA, rtol=1e-12), \
    f"polyDelta at freeze should equal scalar; got max dev {np.abs(poly_d0 - DELTA).max():.3e}"
print(f"after freeze: polyDelta = {DELTA} everywhere (OK)")

# Tiny perturbation that varies S(s) modestly across polygons.
rng = np.random.default_rng(7)
pos0 = np.array(m.getPositions())
m.setPositions((pos0 + 1e-4 * rng.standard_normal(pos0.shape)).tolist())

# Call updateForceEnergy WITHOUT manually calling updatePolyDelta.
m.updateForceEnergy()

poly_d1 = np.array(m.getPolyDelta())
spread = poly_d1.max() - poly_d1.min()
print(f"after perturb + updateForceEnergy: polyDelta range "
      f"[{poly_d1.min():.6e}, {poly_d1.max():.6e}]  spread={spread:.3e}")
assert spread > 1e-9, "polyDelta should differ from frozen value after perturb"

# Reset positions; updateForceEnergy should restore polyDelta to ~scalar.
m.setPositions(pos0.tolist())
m.updateForceEnergy()
poly_d2 = np.array(m.getPolyDelta())
print(f"after restore + updateForceEnergy: polyDelta range "
      f"[{poly_d2.min():.6e}, {poly_d2.max():.6e}]")
assert np.allclose(poly_d2, DELTA, rtol=1e-10), \
    f"polyDelta should return to scalar after position restore; max dev {np.abs(poly_d2 - DELTA).max():.3e}"

# Control: with per-poly mode disabled, updateForceEnergy should NOT touch
# polyDelta_d (we still fill it with scalar inside the force-kernel method,
# but the closed-form should not run here).
m.disablePerPolygonDelta()
m.setPositions((pos0 + 1e-4 * rng.standard_normal(pos0.shape)).tolist())
m.updateForceEnergy()
poly_d3 = np.array(m.getPolyDelta())
print(f"per-poly DISABLED + perturb + updateForceEnergy: polyDelta = scalar? "
      f"max dev {np.abs(poly_d3 - DELTA).max():.3e}")
# polyDelta_d should still be the last-stored values from the per-poly era,
# OR the scalar if the force-kernel method re-filled it. Either way it should
# NOT have been recomputed by the closed form (which would diverge from scalar).
# Since we're not running the force kernel here, polyDelta is whatever was
# left over. The key check: closed-form was NOT applied.
assert np.allclose(poly_d3, DELTA, rtol=1e-10), \
    "with per-poly mode disabled, polyDelta should not change via closed-form"
print("control passes: closed-form gated by usePerPolygonDelta")

print("\nAutomatic polyDelta refresh inside updateForceEnergy works.")
