"""
Data loading and validation utilities for MLS Ensemble Predictor
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

from .constants import (
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
    XG_WEIGHT,
    GF_WEIGHT,
    XG_PROXY_FACTOR,
)
from .exceptions import DataValidationError

logger = logging.getLogger(__name__)


class DataValidator:
    """Centraliza la validación de datos del ensemble."""

    @staticmethod
    def validate_columns(df: pd.DataFrame) -> None:
        """Valida que el DataFrame tenga las columnas requeridas."""
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")

    @staticmethod
    def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza los tipos de datos del DataFrame."""
        df = df.copy()
        df["team"] = df["team"].astype(str).str.strip()
        df["gf"] = pd.to_numeric(df["gf"], errors="coerce").fillna(0).astype(int)
        df["ga"] = pd.to_numeric(df["ga"], errors="coerce").fillna(0).astype(int)
        df["matches"] = pd.to_numeric(df["matches"], errors="coerce").fillna(1).clip(lower=1).astype(int)

        if "xg" in df.columns:
            df["xg"] = pd.to_numeric(df["xg"], errors="coerce")

        return df

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Elimina duplicados manteniendo el último registro de cada equipo."""
        return df.drop_duplicates(subset=["team"], keep="last")


def validate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Valida y normaliza estadísticas del DataFrame."""
    validator = DataValidator()
    validator.validate_columns(df)
    df = validator.normalize_types(df)
    df = validator.remove_duplicates(df)
    logger.info(f"Validated {len(df)} teams")
    return df


def load_mls_data(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Carga datos MLS desde archivo CSV o DataFrame."""
    if isinstance(source, pd.DataFrame):
        df = source
    else:
        path = Path(source)
        if not path.exists():
            raise DataValidationError(f"Data file not found: {path}")
        df = pd.read_csv(path)
    return validate_stats(df)


def create_sample_mls_csv(output_path: Union[str, Path] = "data/sample_mls_stats.csv") -> Path:
    """Genera un CSV de ejemplo con estadísticas MLS."""
    sample = {
        "team": [
            "Inter Miami CF", "LA Galaxy", "Columbus Crew", "FC Cincinnati",
            "Real Salt Lake", "Seattle Sounders FC", "Houston Dynamo",
            "New York Red Bulls", "Philadelphia Union", "Orlando City SC",
            "Minnesota United", "Portland Timbers", "Atlanta United",
            "New York City FC", "Chicago Fire FC", "Sporting Kansas City",
            "FC Dallas", "Nashville SC", "Austin FC", "San Jose Earthquakes",
            "Vancouver Whitecaps", "CF Montreal", "Toronto FC", "DC United",
            "Colorado Rapids", "St. Louis CITY SC", "Charlotte FC", "New England Revolution"
        ],
        "gf": [62, 58, 55, 52, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26],
        "ga": [38, 42, 35, 36, 40, 39, 41, 38, 37, 40, 42, 44, 45, 43, 46, 47, 48, 45, 49, 52, 41, 50, 55, 53, 48, 51, 49, 54],
        "matches": [28] * 28,
        "xg": [58.2, 55.1, 52.4, 50.8, 47.9, 46.5, 45.2, 44.8, 43.7, 42.9, 41.5, 40.8, 39.6, 38.9, 37.4, 36.8, 35.9, 35.1, 34.2, 33.5, 32.8, 31.9, 30.7, 29.8, 28.9, 28.1, 27.4, 26.5]
    }
    df = pd.DataFrame(sample)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Sample MLS data written to {path}")
    return path
