"""
test_rounded.py
Tests for the 'rounded' model enum in pyCudaPolygon.

Tests:
  1. Energy matches the Python reference (smooth_polygon.intersection_area)
  2. Translation invariance: total force on all vertices sums to zero
  3. Finite-difference gradient check: CUDA force ≈ -dE/d(vertex positions)
  4. Continuity: energy varies smoothly as two polygons approach each other
  5. Zero overlap: energy is 0 when polygons don't overlap

Run with:
  python test_rounded.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../pyCudaPolygon')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pyPolygon')))

import numpy as np
import math
import pyCudaPolygon as pcp

# --- helpers to build a pyCudaPolygon model from two smoothed polygons ----------

def _build_model(polys_verts, delta, box=1.0):
    """
    polys_verts: list of (n,2) arrays, backbone vertices in [0,box)
    Returns a configured model with rounded enum and delta set.
    """

    all_verts = np.vstack(polys_verts)
    # Normalize to [0,1) box
    all_verts = all_verts / box

    nv = len(all_verts)
    sizes = [len(v) for v in polys_verts]
    start_indices = [0]
    for s in sizes:
        start_indices.append(start_indices[-1] + s)

    m = pcp.model(nv)
    m.setStartIndices(start_indices)
    m.setPositions(all_verts.flatten().tolist())
    m.setModelEnum("rounded")
    m.setDelta(delta / box)   # delta also normalized
    m.setMaxEdgeLength(0.5)   # must be set before initializeNeighborCells
    m.initializeNeighborCells()
    m.updateNeighborCells()
    m.updateNeighbors()
    m.updatePolygonGeometry()
    return m


def _get_energy_forces(polys_verts, delta, box=1.0):
    """Run one force/energy step and return (energy_raw, forces array)."""
    m = _build_model(polys_verts, delta, box)
    m.updateForceEnergy()
    E = m.getEnergy()
    F = np.array(m.getForces())
    pos = np.array(m.getPositions()) * box   # back to real units
    return E, F, pos


def regular_polygon(n, R, cx=0.0, cy=0.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([cx + R*np.cos(angles), cy + R*np.sin(angles)])


# ============================================================
# Test 1: energy matches Python reference
# ============================================================
def test_energy_vs_python(tol=5e-3):
    """
    CUDA energy should match smooth_polygon.intersection_area to within tol
    (relative error).  We use a generous tolerance since the CUDA kernel
    approximates the crossing-point gradient but computes energy exactly.
    """
    try:
        from smooth_polygon import SmoothedPolygon, intersection_area
    except ImportError:
        print("SKIP test_energy_vs_python: smooth_polygon not found")
        return True

    n = 6
    R = 0.15
    delta = 0.02
    box = 1.0

    # Two hexagons, slightly overlapping
    cx1, cy1 = 0.5, 0.5
    cx2, cy2 = 0.5 + 0.22, 0.5
    v1 = regular_polygon(n, R, cx1, cy1)
    v2 = regular_polygon(n, R, cx2, cy2)

    # Python reference
    A = SmoothedPolygon(v1, delta)
    B = SmoothedPolygon(v2, delta)
    ref = intersection_area(A, B)

    # CUDA
    E_cuda, _, _ = _get_energy_forces([v1, v2], delta, box)

    if ref < 1e-12:
        ok = E_cuda < 1e-8
        print(f"test_energy_vs_python: ref={ref:.3e} cuda={E_cuda:.3e}  "
              + ("PASS" if ok else "FAIL"))
        return ok

    rel_err = abs(E_cuda - ref) / ref
    ok = rel_err < tol
    print(f"test_energy_vs_python: ref={ref:.6e}  cuda={E_cuda:.6e}  "
          f"rel_err={rel_err:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================
# Test 2: translation invariance (net force = 0)
# ============================================================
def test_translation_invariance(tol=1e-10):
    """
    Sum of all forces on vertices of polygon A should cancel with polygon B.
    Total net force on all vertices = 0 (Newton's 3rd law).
    """
    n = 5
    R = 0.12
    delta = 0.015
    v1 = regular_polygon(n, R, 0.5, 0.5)
    v2 = regular_polygon(n, R, 0.5 + 0.18, 0.5)

    E, F, _ = _get_energy_forces([v1, v2], delta)
    F2d = F.reshape(-1, 2)
    net = F2d.sum(axis=0)
    norm_net = np.linalg.norm(net)
    norm_F   = np.linalg.norm(F2d)

    ok = norm_net < tol * (norm_F + 1.0)
    print(f"test_translation_invariance: |net_force|={norm_net:.2e}  "
          f"|force|={norm_F:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================
# Test 3: finite-difference gradient check
# ============================================================
def test_gradient_fd(eps=1e-5, rtol=0.05, atol=1e-8):
    """
    CUDA force on vertex k should equal -(dE/d pos_k) via central differences.
    We check a subset of vertices to keep runtime reasonable.
    Tolerance is generous (5%) because crossing-point gradient is approximate.
    """
    n = 5
    R = 0.12
    delta = 0.015
    box = 1.0

    v1 = regular_polygon(n, R, 0.5, 0.5)
    v2 = regular_polygon(n, R, 0.5 + 0.175, 0.5)

    nv = n + n

    def energy_at(verts_flat):
        v1p = verts_flat[:2*n].reshape(n, 2)
        v2p = verts_flat[2*n:].reshape(n, 2)
        E, _, _ = _get_energy_forces([v1p, v2p], delta, box)
        return E

    all_verts = np.concatenate([v1.flatten(), v2.flatten()])
    E0, F0, _ = _get_energy_forces([v1, v2], delta, box)
    F0 = np.array(F0)

    # Check first 4 vertices of each polygon (8 total * 2 components = 16)
    max_err = 0.0
    n_checked = 0
    check_verts = list(range(min(4, n))) + list(range(n, n + min(4, n)))
    for k in check_verts:
        for d in range(2):
            idx = 2*k + d
            vp = all_verts.copy(); vp[idx] += eps
            vm = all_verts.copy(); vm[idx] -= eps
            Ep = energy_at(vp)
            Em = energy_at(vm)
            fd_grad = (Ep - Em) / (2*eps)
            cuda_force = F0[idx]  # force = -gradient
            # Compare: cuda_force ≈ -fd_grad
            err = abs(cuda_force + fd_grad)
            scale = abs(fd_grad) + atol
            max_err = max(max_err, err / scale)
            n_checked += 1

    ok = max_err < rtol
    print(f"test_gradient_fd: max_rel_err={max_err:.3f} (tol={rtol})  "
          f"checked {n_checked} components  {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================
# Test 4: energy is zero when polygons don't overlap
# ============================================================
def test_zero_when_separated(tol=1e-12):
    n = 6
    R = 0.1
    delta = 0.01
    # Place polygons 0.5 apart (much larger than R + delta)
    v1 = regular_polygon(n, R, 0.25, 0.5)
    v2 = regular_polygon(n, R, 0.75, 0.5)

    E, F, _ = _get_energy_forces([v1, v2], delta)
    ok = abs(E) < tol
    print(f"test_zero_when_separated: E={E:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================
# Test 5: energy increases monotonically as polygons overlap more
# ============================================================
def test_energy_monotone():
    n = 6
    R = 0.12
    delta = 0.015
    cx1, cy1 = 0.5, 0.5
    energies = []
    offsets = np.linspace(0.30, 0.18, 6)  # decreasing separation → more overlap
    for dx in offsets:
        v1 = regular_polygon(n, R, cx1, cy1)
        v2 = regular_polygon(n, R, cx1 + dx, cy1)
        E, _, _ = _get_energy_forces([v1, v2], delta)
        energies.append(E)

    diffs = np.diff(energies)
    ok = all(d > -1e-10 for d in diffs)  # allow tiny numerical noise
    print(f"test_energy_monotone: energies={[f'{e:.4e}' for e in energies]}  "
          + ("PASS" if ok else "FAIL"))
    return ok


# ============================================================
# Test 6: energy equals polygon area when one fully contains other
# ============================================================
def test_full_containment():
    """
    When polygon B is fully inside A, Area(Ã∩B̃) ≈ Area(B̃).
    We compare with the Python reference.
    """
    try:
        from smooth_polygon import SmoothedPolygon, intersection_area
    except ImportError:
        print("SKIP test_full_containment")
        return True

    n = 6
    R_big = 0.20
    R_small = 0.06
    delta = 0.008
    cx, cy = 0.5, 0.5

    v1 = regular_polygon(n, R_big, cx, cy)
    v2 = regular_polygon(n, R_small, cx, cy)   # small one inside big one

    A = SmoothedPolygon(v1, delta)
    B = SmoothedPolygon(v2, delta)
    ref = intersection_area(A, B)
    ref_B = B.area()

    E, _, _ = _get_energy_forces([v1, v2], delta)

    err = abs(E - ref) / (abs(ref) + 1e-14)
    ok = err < 0.01   # 1% relative
    print(f"test_full_containment: cuda={E:.6e}  ref={ref:.6e}  "
          f"B_area={ref_B:.6e}  rel_err={err:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("Rounded polygon kernel tests")
    print("=" * 60)

    results = []
    results.append(test_zero_when_separated())
    results.append(test_translation_invariance())
    results.append(test_energy_vs_python())
    results.append(test_full_containment())
    results.append(test_energy_monotone())
    results.append(test_gradient_fd())

    print("=" * 60)
    n_pass = sum(results)
    print(f"Results: {n_pass}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
