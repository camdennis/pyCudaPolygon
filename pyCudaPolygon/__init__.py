# The compiled model loads only when asked for, so that pure-Python modules
# such as oracle and depth can be imported without building the extension.
def __getattr__(name):
    if name == "model":
        from .pyCudaPolygon import model
        return model
    raise AttributeError(f"module 'pyCudaPolygon' has no attribute {name!r}")
