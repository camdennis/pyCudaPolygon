import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pyCudaPolygon'))
import pyCudaPolygon as pcp
import matplotlib.pyplot as plt

save_name = sys.argv[1] if len(sys.argv) > 1 else 'nearInstability'
save_path = os.path.join(os.path.dirname(__file__), save_name)

m = pcp.model(size=1)
m.loadModel(save_path)
m.updatePolygonGeometry()

phi = m.getPhi()
N = m.getNumPolygons()
E = m.getEnergy()

fig, ax = plt.subplots(figsize=(8, 8))
m.draw(ax=ax)
ax.set_title(f'{save_name}  |  φ={phi:.3f}, N={N}, E={E:.3e}')

out_path = os.path.join(os.path.dirname(__file__), f'viz_{save_name}.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved → {out_path}')
plt.show()
