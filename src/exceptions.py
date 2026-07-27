"""
Excepciones personalizadas para el Ensemble MLS
"""

from __future__ import annotations


class EnsembleError(Exception):
    """Excepción base para el ensemble."""
    pass


class DataValidationError(EnsembleError):
    """Error al validar datos de entrada."""
    pass


class DataNotIngestedError(EnsembleError):
    """Error cuando se intenta predecir sin datos ingestionados."""
    pass


class TeamNotFoundError(EnsembleError):
    """Error cuando un equipo no se encuentra en los datos."""
    pass


class iSportsAPIError(EnsembleError):
    """Error relacionado con la API de iSports."""
    pass


class ConfigurationError(EnsembleError):
    """Error de configuración (ej: falta API key)."""
    pass
