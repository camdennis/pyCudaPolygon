"""
Dynamically detects enumerated types from libpyCudaPacking

This module implicitly assumes that all enum supertypes
end in "Enum"
"""
if __package__:
    from . import libpyCudaPolygon as lpcp       # the package's own build
else:
    import libpyCudaPolygon as lpcp

__all__ = []
for name in lpcp.__dict__:
    if name[-4:] == "Enum":
        __all__.append(name)
        globals()[name] = getattr(lpcp, name)
