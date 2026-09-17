"""Shared code for the GAPA repository.

`gapa.paths` holds every canonical repo location; `gapa.utils` holds the prompt,
split, metric, and plotting helpers used across all analysis sections.
"""

from gapa import paths  # noqa: F401

__all__ = ["paths", "utils"]
