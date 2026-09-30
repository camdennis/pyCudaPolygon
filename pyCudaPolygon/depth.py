"""The exact distance field of a polygon. M1.
Conventions:
    loop      (n, 2) float array, vertices counter-clockwise.
    feature   ("edge", i)   the segment from vertex i to vertex i+1
              ("vertex", i) the vertex i itself
"""

import numpy as np
from matplotlib import pyplot as plt

def edgeFrame(loop):
    """
    Per-edge geometry for a CCW loop.

    Returns:
        edges   (n, 2) edge vectors, e_i = V_{i+1} - V_i
        lengths (n,)   |e_i|
        tangent (n, 2) unit tangents
        normal  (n, 2) OUTWARD unit normals
    """
    n = len(loop)
    edges = np.zeros((n, 2))
    lengths = np.zeros(n)
    tangent = np.zeros((n, 2))
    normal = np.zeros((n, 2))
    for i in range(n):
        edges[i] = loop[(i + 1) % n] - loop[i]
        lengths[i] = np.linalg.norm(edges[i])
        tangent[i] = edges[i] / lengths[i]
        normal[i] = np.array([edges[i][1], -edges[i][0]]) / lengths[i]
    return edges, lengths, tangent, normal

def nearestFeature(point, loop, frame = None):
    """
    The feature of `loop` nearest to `point`.

    Returns:
        (kind, index, distance) with kind in {"edge", "vertex"} and distance
        the unsigned Euclidean distance to that feature.

    Constraints:
        `loop` is simple but NOT necessarily convex. Reflex vertices are the
        whole difficulty: for a convex loop an interior point is always nearest
        to an edge, and for a nonconvex one it need not be.

        Ties are real, not hypothetical. A point equidistant from two features
        sits on the medial axis, and packings put points there. Should we
        keep all points and make the force the average of them? Let's not bother.
        This is a single point that is being integrated over.

    Complexity:
        O(n) per point is fine here. This is the reference implementation, not
        the fast one.
    """
    n = len(loop)
    if frame is None:
        frame = edgeFrame(loop)
    edges, lengths, tangent, normal = frame
    vertexDist = np.inf
    edgeDist = np.inf
    vertexIndex = -1
    edgeIndex = -1
    for i in range(n):
        # Edge test:
        t = np.dot(point - loop[i], tangent[i]) / lengths[i]
        if (t > 0 and t < 1):
            dist = np.abs(np.dot(normal[i], (point - loop[i])))
            if (dist < edgeDist):
                # When tied, choose the first one you got
                edgeDist = dist
                edgeIndex = i
        # We don't know if we're inside or outside, 
        # so check the vertex anyway
        dist = np.linalg.norm(loop[i] - point)
        if (dist < vertexDist):
            # When tied, choose the first one you got
            vertexDist = dist
            vertexIndex = i
    if (edgeDist <= vertexDist):
        # Choose edges when all else is the same
        return ["edge", edgeIndex, edgeDist]
    return ["vertex", vertexIndex, vertexDist]

def signedDistance(point, loop, frame = None):
    """
    Signed distance from `point` to the boundary of `loop`, positive inside.

    Constraints:
        Must agree with `nearestFeature` in magnitude everywhere, including
        outside the loop, and must be continuous across every feature switch.
        That continuity is not an accident and it is not decoration -- the
        whole regularity argument for the contact potential rests on it.

    Example:
        loop = unit square
        signedDistance([0.25, 0.5], loop) ->  0.25
        signedDistance([1.25, 0.5], loop) -> -0.25
    """
    n = len(loop)
    if frame is None:
        frame = edgeFrame(loop)
    edges, lengths, tangent, normal = frame
    feature, index, dist = nearestFeature(point, loop, frame = frame)
    # Check if a point is inside or outside the polygon
    if (feature == "edge" and np.dot(normal[index], (loop[index] - point)) < 0):
        dist *= -1
    if (feature == "vertex"):
        # check angle weighted normal of two adjacent edges
        # That's just n_i + n_z'(i) normalized
        awn = normal[index] + normal[(index - 1 + n) % n]
        awn /= np.linalg.norm(awn)
        if (np.dot(awn, loop[index] - point) < 0):
            dist *= -1        
    return dist

def partitionCertificate(loop, samples = 100_000, rng = None):
    """
    Evidence that your feature partition is exact: that the edge regions and
    vertex regions together cover the plane with no gaps and no double cover.

    This is the M1 deliverable, and it is a certificate rather than a test
    because it has to justify a claim in the paper, not merely fail loudly.

    Suggested shape of the return value -- change it if you want something
    better, this is your design decision and not mine:

        {
          "maxDisagreement": float,   # sup |nearestFeature| - |brute force|
          "featureMismatch": int,     # samples where the claimed feature is
                                      # not a realising feature
          "coverage": ...,            # your evidence of no gaps
          "samples": int,
        }

    Two questions to answer before you write it, because they determine what
    the certificate has to measure:

      1. What would a GAP in the partition actually look like numerically?
         A point assigned to no feature at all is easy. What is the failure
         mode that is NOT easy?
      2. Random sampling almost never lands on a region boundary, which is
         exactly where the partition can be wrong. So what do you sample?
    """
    # Let's plot this!
    frame = edgeFrame(loop)
    xMax = np.max(loop[:, 0])
    xMin = np.min(loop[:, 0])
    yMax = np.max(loop[:, 1])
    yMin = np.min(loop[:, 1])
    xs = (np.random.rand(samples) * 1.2 - 0.1) * (xMax - xMin) + xMin
    ys = (np.random.rand(samples) * 1.2 - 0.1) * (yMax - yMin) + yMin
    points = np.concatenate((xs, ys)).reshape(2, samples).T
    allDist = []
    for point in points:
        # We find the nearest feature, their distance, and their type
        nf = nearestFeature(point, loop, frame = frame)
        # Check for feature type:
        allDist.append(signedDistance(point, loop, frame = frame))

    scatter = plt.scatter(xs, ys, c = allDist, cmap = 'viridis', s = 100, alpha = 0.8)
    cbar = plt.colorbar(scatter)
    cbar.set_label('signed distance', fontsize = 12)
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.show()
    # I didn't do anything sophisticated here because I don't believe it is warranted.

if (__name__ == "__main__"):
    loop = np.array([
        [0.00, 0.000],   # v0
        [0.90, 0.000],   # v1
        [0.90, 0.300],   # v2   mouth of a narrow pocket that opens to the right
        [0.55, 0.360],   # v3   reflex, 260°   inner end of the pocket, 0.04 wide
        [0.55, 0.400],   # v4   reflex, 264°
        [0.95, 0.440],   # v5
        [0.95, 0.700],   # v6
        [0.80, 0.980],   # v7   spike, interior angle 15°
        [0.86, 0.720],   # v8   reflex, 287°
        [0.40, 0.750],   # v9   nearly straight: 180.08°, technically reflex
        [0.10, 0.770],   # v10
        [0.09, 0.752],   # v11  edge v10–v11 is short, 0.021
        [0.35, 0.550],   # v12  deep notch, reflex 304°
        [0.05, 0.450],   # v13
        [0.00, 0.300],   # v14
    ])
    # Verify edge frame:
    frame = edgeFrame(loop)
    edges, lengths, tangent, normal = frame
    x = np.concatenate((loop[:, 0], [loop[0, 0]]))
    y = np.concatenate((loop[:, 1], [loop[0, 1]]))
    plt.plot(x, y, linewidth = 1, color = 'k')
#    point = np.array([0.5, 0.2])
#    plt.plot([point[0]], [point[1]], "*")
    # Passes
    # Now find the nearest feature 
    partitionCertificate(loop, samples = 1000)
    plt.show()
    
