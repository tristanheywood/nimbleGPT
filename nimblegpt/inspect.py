"""Development utilities for inspecting various objects."""


from typing import Dict


def param_shapes(params: Dict, depth: int = 1e10):
    """Given a flax parameter dictionary, return a dictionary with the same structure
    but with the shape of each parameter instead of the parameter itself. Optionally
    omit the shape of parameters with depth > `depth`."""
    try:
        return f"({', '.join(map(str, params.shape))})"
    except:
        pass
    if depth == 0:
        return "<omitted>"
    return {k: param_shapes(v, depth - 1) for k, v in params.items()}
