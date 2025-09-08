"""
Dynamically detects enumerated types from libpyCudaPacking

This module implicitly assumes that all enum supertypes
end in "Enum"
"""
#from .pyCudaPolygonLink import libpyCudaPolygon as lpcp
import libpyCudaPolygon as lpcp

__all__ = []
for name in lpcp.__dict__:
    if name[-4:] == "Enum":
        __all__.append(name)
        globals()[name] = getattr(lpcp, name)
