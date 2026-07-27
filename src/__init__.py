# MLS Ensemble Predictor Package
from .ensemble import EnsembleFootballPredictor
from .data_loader import load_mls_data, validate_stats
from .isports_client import iSportsClient, fetch_mls_for_ensemble
from .exceptions import (
    EnsembleError,
    DataValidationError,
    DataNotIngestedError,
    TeamNotFoundError,
    iSportsAPIError,
    ConfigurationError,
)

__version__ = "1.0.0"
__all__ = [
    "EnsembleFootballPredictor",
    "load_mls_data",
    "validate_stats",
    "iSportsClient",
    "fetch_mls_for_ensemble",
    "EnsembleError",
    "DataValidationError",
    "DataNotIngestedError",
    "TeamNotFoundError",
    "iSportsAPIError",
    "ConfigurationError",
]
