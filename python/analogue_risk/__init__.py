"""analogue-risk: regime-conditioned analogue forecasting of financial risk."""

from ._core import *  # noqa: F401,F403  (low-level Rust functions)
from .api import Analogue, volatility_forecast  # noqa: F401

__version__ = "2.0.0"
