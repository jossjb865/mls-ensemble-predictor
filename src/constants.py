"""
Constantes globales y configuración del Ensemble MLS
"""

from __future__ import annotations

from typing import Final

# Configuración de la API
BASE_URLS: Final[list[str]] = [
    "http://api.isportsapi.com",
    "http://api2.isportsapi.com",
]

# Tipos de estadísticas en iSportsAPI
STAT_TYPES: Final[dict[str, int]] = {
    "xg": 52,              # Expected goals
    "shots": 3,            # Total shots
    "shots_on_target": 4,  # Shots on target
}

# Columnas requeridas
REQUIRED_COLUMNS: Final[set[str]] = {"team", "gf", "ga", "matches"}
OPTIONAL_COLUMNS: Final[set[str]] = {"xg", "points", "rank", "teamId"}

# Parámetros del modelo Ensemble
ENSEMBLE_WEIGHTS: Final[dict[str, float]] = {
    "poisson": 0.25,
    "xgboost_sim": 0.30,
    "catboost_sim": 0.25,
    "lstm_sim": 0.20,
}

# Configuración de lambdas y defensas
MIN_DEFENSE: Final[float] = 0.35  # Defensa mínima permitida
MIN_LAMBDA: Final[float] = 0.25   # Lambda mínima permitida
MIN_TREND: Final[float] = 0.6     # Tendencia mínima
MAX_TREND: Final[float] = 1.4     # Tendencia máxima

# Valores por defecto
DEFAULT_HOME_ADVANTAGE: Final[float] = 1.18
DEFAULT_MAX_GOALS: Final[int] = 8
DEFAULT_LEAGUE_AVG_GOALS: Final[float] = 2.75
DEFAULT_FORM_HOME: Final[float] = 0.55
DEFAULT_FORM_AWAY: Final[float] = 0.45
DEFAULT_MOMENTUM: Final[float] = 0.50
DEFAULT_HIERARCHY: Final[float] = 1.0

# Configuración de validación
MIN_FORM: Final[float] = 0.0
MAX_FORM: Final[float] = 1.0
XG_PROXY_FACTOR: Final[float] = 0.95  # gf * 0.95 como proxy cuando no hay xG real
XG_WEIGHT: Final[float] = 0.45
GF_WEIGHT: Final[float] = 0.55

# Coeficientes de modelos
XGBOOST_FORM_ALPHA: Final[float] = 0.82
XGBOOST_FORM_BETA: Final[float] = 0.36
XGBOOST_MOMENTUM_ALPHA: Final[float] = 0.82
XGBOOST_MOMENTUM_BETA: Final[float] = 0.36

LSTM_BASE: Final[float] = 0.88
LSTM_FORM_FACTOR: Final[float] = 0.24
LSTM_MOMENTUM_FACTOR: Final[float] = 0.18

CATBOOST_HOME_FACTOR: Final[float] = 0.97
CATBOOST_AWAY_FACTOR: Final[float] = 1.03

LEAGUE_AVG_MIN: Final[float] = 1.8
LEAGUE_AVG_DIVISOR: Final[float] = 1.35
