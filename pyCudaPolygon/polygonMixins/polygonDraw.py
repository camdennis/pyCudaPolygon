import numpy as np
from matplotlib import pyplot as plt

class Mixin():
    def __init__(self, rng = None):
        # random number generator fallback
        self.rng = rng if rng is not None else np.random.default_rng()

    def draw(self, ax = None, numbering = False, forces = None, arrowColor = 'r', axisSize = 1, ms = 3):

        def fixPXPY(px, py):
            minX = min(px)
            maxX = max(px)
            minY = min(py)
            maxY = max(py)
            px += 1.5 - minX
            py += 1.5 - minY
            px %= 1
            py %= 1
            px += minX - 0.5
            py += minY - 0.5
            return px, py

        pos = self.getPositions()
        start = 0
        fStart = 0
        if ax is None:
            fig, ax = plt.subplots()

        nArray = self.getnArray()
        cmap = plt.get_cmap('tab20')  # choose any colormap you prefer

        for poly_idx, n in enumerate(nArray):
            color = cmap(poly_idx % cmap.N)
            px = pos[start:start + 2 * n][::2]
            py = pos[start:start + 2 * n][1::2]
            px = np.concatenate((px, [px[0]]))
            py = np.concatenate((py, [py[0]]))
            px, py = fixPXPY(px, py)
            if forces is not None:
                fx = forces[fStart:fStart + 2 * n][::2]
                fy = forces[fStart:fStart + 2 * n][1::2]
                fx = np.concatenate((fx, [fx[0]]))
                fy = np.concatenate((fy, [fy[0]]))

            for i in range(3):
                for j in range(3):
                    ax.plot((px + i - 1) * axisSize, (py + j - 1) * axisSize,
                            '-o', markersize=ms,
                            color=color,
                            markerfacecolor=color,
                            markeredgecolor=color)
                    if (forces is not None):
                        ax.quiver(px + i - 1, py + j - 1,
                            fx, fy, angles = 'xy',
                            scale_units = 'xy',
                            scale = 1,
                            color = arrowColor,
                            width = 0.003,
                            headwidth = 3,
                            headlength = 4
                        )

            if (numbering):
                for k in range(len(px) - 1):
                    textX = (px[k] + 1) % 1
                    textY = (py[k] + 1) % 1
                    ax.text(textX * axisSize, textY * axisSize, str(start // 2 + k),
                            fontsize = 8, color = 'k', ha = 'left', va = 'bottom')

            start += 2 * n
            fStart += 2 * n

        ax.set_xlim([0, axisSize])
        ax.set_ylim([0, axisSize])
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax
