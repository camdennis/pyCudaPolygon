import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyCudaPolygon'))
import pyCudaPolygon as pcp
import numpy as np

rng = np.random.default_rng(42)

DELTA = 1e-3   # vertex-disk rounding radius

m = pcp.model(size=1)
m.loadModel(os.path.join(os.path.dirname(__file__), 'testSave'))
m.updatePolygonGeometry()
m.setDelta(DELTA)
print(f'Loaded: numVertices={m.getNumVertices()}, numPolygons={m.getNumPolygons()}, phi={m.getPhi():.4f}, delta={DELTA:.1e}', flush=True)

print('Starting FIRE minimization...', flush=True)
instability_step = None
best_force = float('inf')
best_step = 0
total_step = 0
max_total_steps = 2000000

best_save_path = os.path.join(os.path.dirname(__file__), 'tmp_best_state')

# Phase schedule: (dtMax, stagnation_window, max_steps_this_phase)
phases = [
    (1e-2,  2000,  50000),
    (1e-3,  3000, 100000),
    (1e-4,  5000, 200000),
    (1e-5,  8000, 500000),
    (1e-6, 10000, 500000),
]

MAX_KICKS  = 8     # kick attempts per phase stagnation
KICK_BURST = 3000  # FIRE steps to run after each kick

def apply_kick(sigma_trans, sigma_rot=None):
    """Reload best state, perturb each polygon rigidly (COM translation + rotation)."""
    if os.path.isdir(best_save_path):
        m.loadModel(best_save_path)
    nP = m.getNumPolygons()
    nV = m.getNumVertices()
    n = nV // nP  # vertices per polygon (all same size)
    pos = m.getPositions().copy().reshape(-1, 2)
    for p in range(nP):
        s, e = p * n, (p + 1) * n
        com = pos[s:e].mean(axis=0)
        rel = pos[s:e] - com
        if sigma_rot is not None:
            theta = rng.normal(0, sigma_rot)
            c, s_ = np.cos(theta), np.sin(theta)
            rel = np.column_stack([c * rel[:, 0] - s_ * rel[:, 1],
                                   s_ * rel[:, 0] + c * rel[:, 1]])
        com = (com + rng.normal(0, sigma_trans, 2)) % 1.0
        pos[s:e] = com + rel
    m.setPositions(pos.reshape(-1))
    m.updatePolygonGeometry()
    m.updateNeighborCells()
    m.updateNeighbors()
    m.updateOutersections()
    m.updateForceEnergy()
    m.projectForce()
    m.resetVelocities()

def run_burst(dtMax, max_steps, stag_win):
    """Run FIRE for up to max_steps. Returns True if stagnated."""
    global best_force, best_step, total_step, instability_step
    last_improvement = total_step
    for energy, maxForce, dt, step in m.minimizeFIRELoop(
            maxForceThreshold=1e-14, dt=min(1e-3, dtMax),
            maxSteps=max_steps, dtMax=dtMax):
        total_step += 1
        if total_step % 25 == 0:
            print(f'step={total_step:7d}, energy={energy:.6e}, maxForce={maxForce:.6e}, '
                  f'best={best_force:.6e}, dt={dt:.6e}', flush=True)
        if maxForce < best_force:
            best_force = maxForce
            best_step  = total_step
            last_improvement = total_step
            m.saveModel(best_save_path, overwrite=True)
        if np.isnan(maxForce) or np.isinf(maxForce) or maxForce > 1e2:
            print(f'EXPLOSION at step {total_step}! maxForce={maxForce:.6e}', flush=True)
            instability_step = total_step
            return False
        if maxForce <= 1e-14:
            return False
        if total_step - last_improvement >= stag_win:
            return True  # stagnated
    return False  # exhausted steps

for phase_idx, (dtMax, stag_win, max_phase_steps) in enumerate(phases):
    phase = phase_idx + 1

    if phase_idx > 0 and os.path.isdir(best_save_path):
        m.loadModel(best_save_path)

    print(f'--- Phase {phase}: dtMax={dtMax:.1e}, stag_win={stag_win}, '
          f'starting at step {total_step}, best_force={best_force:.6e} ---', flush=True)

    stagnated = run_burst(dtMax, max_phase_steps, stag_win)

    if not stagnated or instability_step is not None or best_force <= 1e-14:
        break

    # Stagnated — try basin-hopping kicks
    print(f'  Stagnated at step {total_step}: best={best_force:.6e}. Trying kicks.', flush=True)
    for k in range(1, MAX_KICKS + 1):
        sigma_t = 5e-3 * (1.5 ** (k - 1))  # COM: 5e-3, 7.5e-3, 1.1e-2, ...
        sigma_r = 0.05 * (1.5 ** (k - 1))   # rot: 0.05, 0.075, 0.11 rad, ...
        force_before = best_force
        print(f'  Kick {k}/{MAX_KICKS}: sigma_t={sigma_t:.2e}, sigma_r={sigma_r:.2e}, from best={force_before:.6e}', flush=True)
        apply_kick(sigma_t, sigma_r)
        run_burst(dtMax, KICK_BURST, KICK_BURST)
        if instability_step is not None or best_force <= 1e-14:
            break
        if best_force < force_before:
            print(f'  Kick {k} improved: {force_before:.6e} -> {best_force:.6e}', flush=True)
        else:
            print(f'  Kick {k} did not improve force.', flush=True)

    if instability_step is not None or best_force <= 1e-14:
        break
    if total_step >= max_total_steps:
        break

if best_force <= 1e-14:
    print(f'Converged! best_force={best_force:.6e} at step {best_step}', flush=True)
else:
    print(f'Ended at step {total_step}, best_force={best_force:.6e} at step {best_step}', flush=True)

if os.path.isdir(best_save_path):
    m.loadModel(best_save_path)

save_path = os.path.join(os.path.dirname(__file__), 'nearInstability')
m.saveModel(save_path, overwrite=True)
print(f'Saved packing to {save_path}', flush=True)
print(f'constraintViolation: {m.getConstraintViolation()}', flush=True)
