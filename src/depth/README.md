# src/depth — the contact potential

Empty on purpose. This is where the depth-based contact law lands on the device
side, and nothing goes in until the algorithm has stopped changing shape.

## Why nothing is here yet

M1 and M2 are written in Python, in `pyCudaPolygon/depth.py` and checked against
`pyCudaPolygon/oracle.py`. Porting an algorithm you are still discovering is the
reliable way to lose a month to a rewrite. The port starts at M3, once the span
construction is settled, and finishes at M4/M5.

## Where the holes are in the existing code

Two `TODO` markers were left by the strip, and they are the seams this directory
eventually fills:

- `src/model.cpp`, `Model::updateForceEnergy`, cases `normal` and `hybrid` —
  the contact energy and forces. By the time this is called,
  `updateOutersections()` has already run, so the intersection list, the
  outersection pairing and the `tu`/`ut` crossing parameters are available.
  Accumulate into `force` (2 × size doubles) and `energy` (one double).
- `src/model.cpp`, `Model::minimizeFIREStep` — constraint projection, between
  the Verlet update and the outersection rebuild.

## What already exists that you should not rewrite

`updateOverlapArea` and `updateOverlapAreaKernel` are **not** an energy. They
are the grid-sampled ground-truth area estimator — the GPU ancestor of
`oracle.py`. The CUDA version of the depth oracle is an assignment later on, and
that kernel is the pattern to follow.

`intersect`-style crossing detection, the outersection pairing, the parity
flags, and the neighbour cells and balls are all yours already and all reusable.
A large part of M3 is working out precisely which parts of the span problem they
already solve and which they do not.

## Planned layout

Names are a suggestion, not a specification — the module boundaries are part of
what M3 asks you to decide.

    distance.cuh    nearest feature and signed distance        (M3 port of depth.py)
    spans.cuh       span construction and event location       (M3)
    contact.cuh     the closed-form energy per span            (M4)
    contactGrad.cuh forces                                     (M5)
    shake.h         constraint projection                      (M6)

Stubs arrive with each assignment rather than all at once, so that the signature
you get reflects what you have already worked out rather than what I guessed in
September.
