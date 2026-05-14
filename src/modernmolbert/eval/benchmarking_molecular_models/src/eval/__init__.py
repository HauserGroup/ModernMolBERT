__all__ = [
    "eval_procedure",
    "AVAILABLE_HEADS",
]


def __getattr__(name):
    if name == "eval_procedure":
        from .supervised.procedure import eval_procedure

        return eval_procedure
    if name == "AVAILABLE_HEADS":
        from .supervised.models import AVAILABLE_HEADS

        return AVAILABLE_HEADS
    raise AttributeError(name)
