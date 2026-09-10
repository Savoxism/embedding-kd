"""The method registry: the one place that knows which methods exist.

`main.py` reads `METHOD_NAMES` for its `--method` choices and `config_cls` for
the config to build; `distiller.py` reads the flags and hooks. Neither compares
`distill_method` against a string literal any more, so a new method is one
module plus one line here.
"""

from src.methods import cdm, dskd, emo, ggpkd, pointwise, rkd, stella, talas
from src.methods.spec import MethodSpec

REGISTRY: dict[str, MethodSpec] = {
    spec.name: spec
    for spec in (
        cdm.SPEC,
        dskd.SPEC,
        emo.SPEC,
        ggpkd.SPEC,
        pointwise.SPEC,
        rkd.SPEC,
        stella.SPEC,
        talas.SPEC,
    )
}

METHOD_NAMES: tuple[str, ...] = tuple(sorted(REGISTRY))


def get_method(name: str) -> MethodSpec:
    """Return the spec for `name`, or raise naming the ones that exist."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown distillation method {name!r}; "
            f"expected one of: {', '.join(METHOD_NAMES)}"
        ) from None


__all__ = ["METHOD_NAMES", "REGISTRY", "MethodSpec", "get_method"]
