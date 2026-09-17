"""ReAM: Reliability-Aware Address Matching."""

__all__ = ["ReAM"]


def __getattr__(name):
    if name == "ReAM":
        from .model import ReAM
        return ReAM
    raise AttributeError(name)
