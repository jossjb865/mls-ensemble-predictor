"""
Ensemble Football Predictor for MLS
Models: Poisson + XGBoost-sim + CatBoost-sim + LSTM-sim
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import poisson

from .constants import (
    CATBOOST_AWAY_FACTOR,
    CATBOOST_HOME_FACTOR,
    DEFAULT_HOME_ADVANTAGE,
    DEFAULT_LEAGUE_AVG_GOALS,
    DEFAULT_MAX_GOALS,
    ENSEMBLE_WEIGHTS,
    GF_WEIGHT,
    LEAGUE_AVG_DIVISOR,
    LEAGUE_AVG_MIN,
    LSTM_BASE,
    LSTM_FORM_FACTOR,
    LSTM_MOMENTUM_FACTOR,
    MAX_TREND,
    MIN_DEFENSE,
    MIN_LAMBDA,
    MIN_TREND,
    XGBOOST_FORM_ALPHA,
    XGBOOST_FORM_BETA,
    XG_WEIGHT,
)
from .exceptions import DataNotIngestedError, TeamNotFoundError

logger = logging.getLogger(__name__)


class StatsProcessor:
    """Procesa estadísticas de equipos para el modelo."""

    @staticmethod
    def compute_form_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Computa ataque y defensa basados en GF, GA y xG."""
        df = df.copy()
        
        # Ataque base: goles por partido
        df["attack"] = df["gf"] / df["matches"].clip(lower=1)
        
        # Defensa base: goles en contra por partido
        df["defense"] = df["ga"] / df["matches"].clip(lower=1)

        # Ajuste con xG si está disponible
        if "xg" in df.columns:
            df["xg"] = df["xg"].fillna(df["gf"])
            xg_per_match = df["xg"] / df["matches"].clip(lower=1)
            df["attack"] = GF_WEIGHT * df["attack"] + XG_WEIGHT * xg_per_match

        return df

    @staticmethod
    def normalize_league_goals(df: pd.DataFrame) -> float:
        """Normaliza el promedio de goles de la liga."""
        total_goals = df["gf"].sum() + df["ga"].sum()
        total_matches = df["matches"].sum()
        if total_matches > 0:
            league_avg = total_goals / (2 * total_matches)
            return max(LEAGUE_AVG_MIN, league_avg)
        return DEFAULT_LEAGUE_AVG_GOALS


class EnsembleFootballPredictor:
    """
    Ensemble model for any league (optimized defaults for MLS).
    Weights: Poisson 25% | XGBoost-sim 30% | CatBoost-sim 25% | LSTM-sim 20%
    """

    def __init__(
        self,
        home_advantage: float = DEFAULT_HOME_ADVANTAGE,
        max_goals: int = DEFAULT_MAX_GOALS,
        league_avg_goals: float = DEFAULT_LEAGUE_AVG_GOALS,
    ):
        self.home_advantage = home_advantage
        self.max_goals = max_goals
        self.league_avg_goals = league_avg_goals
        self.team_stats: Optional[pd.DataFrame] = None
        self._poisson_cache: Dict[tuple[float, float], np.ndarray] = {}

    def ingest_data(self, data: Union[pd.DataFrame, Dict, str, Path]) -> pd.DataFrame:
        """Ingesta y procesa datos para el modelo."""
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        elif isinstance(data, dict):
            df = pd.DataFrame.from_dict(data, orient="index").reset_index()
            df = df.rename(columns={"index": "team"})
        else:
            df = data.copy()

        # Validar columnas
        required = {"team", "gf", "ga", "matches"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Procesar métricas
        processor = StatsProcessor()
        df = processor.compute_form_metrics(df)
        
        # Normalizar promedio de goles de la liga
        self.league_avg_goals = processor.normalize_league_goals(df)

        # Indexar por equipo para búsquedas rápidas
        self.team_stats = df.set_index("team")
        logger.info(f"Data ingested: {len(df)} teams, league_avg_goals={self.league_avg_goals:.2f}")
        return df

    def _poisson_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        """Calcula matriz de Poisson con cache."""
        cache_key = (round(lambda_home, 4), round(lambda_away, 4))
        if cache_key in self._poisson_cache:
            return self._poisson_cache[cache_key]

        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                matrix[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
        
        s = matrix.sum()
        result = matrix / s if s > 0 else matrix
        self._poisson_cache[cache_key] = result
        return result

    def _simulate_xgboost(
        self,
        attack_h: float,
        defense_a: float,
        attack_a: float,
        defense_h: float,
        form_h: float = 0.5,
        form_a: float = 0.5,
    ) -> Dict[str, float]:
        """Simula predicción XGBoost-sim."""
        defense_a = max(MIN_DEFENSE, defense_a)
        defense_h = max(MIN_DEFENSE, defense_h)
        
        base_h = (attack_h / defense_a) * self.home_advantage
        base_a = attack_a / defense_h
        
        adj_h = base_h * (XGBOOST_FORM_ALPHA + XGBOOST_FORM_BETA * form_h)
        adj_a = base_a * (XGBOOST_FORM_ALPHA + XGBOOST_FORM_BETA * form_a)
        
        return {
            "lambda_h": max(MIN_LAMBDA, adj_h),
            "lambda_a": max(MIN_LAMBDA, adj_a),
            "weight": ENSEMBLE_WEIGHTS["xgboost_sim"],
        }

    def _simulate_catboost(
        self,
        attack_h: float,
        defense_a: float,
        attack_a: float,
        defense_h: float,
        hierarchy: float = 1.0,
    ) -> Dict[str, float]:
        """Simula predicción CatBoost-sim."""
        defense_a = max(MIN_DEFENSE, defense_a)
        defense_h = max(MIN_DEFENSE, defense_h)
        
        base_h = (attack_h / defense_a) * self.home_advantage * hierarchy
        base_a = attack_a / defense_h
        
        return {
            "lambda_h": max(MIN_LAMBDA, base_h * CATBOOST_HOME_FACTOR),
            "lambda_a": max(MIN_LAMBDA, base_a * CATBOOST_AWAY_FACTOR),
            "weight": ENSEMBLE_WEIGHTS["catboost_sim"],
        }

    def _simulate_lstm(
        self,
        form_h: float,
        form_a: float,
        momentum_h: float = 0.5,
        momentum_a: float = 0.5,
    ) -> Dict[str, float]:
        """Simula predicción LSTM-sim basada en forma y momentum."""
        trend_h = LSTM_BASE + LSTM_FORM_FACTOR * (form_h - 0.5) + LSTM_MOMENTUM_FACTOR * (momentum_h - 0.5)
        trend_a = LSTM_BASE + LSTM_FORM_FACTOR * (form_a - 0.5) + LSTM_MOMENTUM_FACTOR * (momentum_a - 0.5)
        
        return {
            "trend_h": max(MIN_TREND, min(MAX_TREND, trend_h)),
            "trend_a": max(MIN_TREND, min(MAX_TREND, trend_a)),
            "weight": ENSEMBLE_WEIGHTS["lstm_sim"],
        }

    def predict(
        self,
        home_team: str,
        away_team: str,
        form_home: float = 0.55,
        form_away: float = 0.45,
        momentum_home: float = 0.50,
        momentum_away: float = 0.50,
        hierarchy: float = 1.0,
    ) -> Dict[str, Any]:
        """Realiza predicción de partido entre dos equipos."""
        if self.team_stats is None:
            raise DataNotIngestedError("No data ingested. Call ingest_data() first.")

        # Validar que los equipos existan
        if home_team not in self.team_stats.index:
            raise TeamNotFoundError(f"Home team '{home_team}' not found in ingested data")
        if away_team not in self.team_stats.index:
            raise TeamNotFoundError(f"Away team '{away_team}' not found in ingested data")

        # Obtener estadísticas
        stats_h = self.team_stats.loc[home_team]
        stats_a = self.team_stats.loc[away_team]

        attack_h = float(stats_h["attack"])
        defense_h = float(stats_h["defense"])
        attack_a = float(stats_a["attack"])
        defense_a = float(stats_a["defense"])

        # Calcular lambdas Poisson
        defense_a_safe = max(MIN_DEFENSE, defense_a)
        defense_h_safe = max(MIN_DEFENSE, defense_h)
        lambda_h_p = (attack_h / defense_a_safe) * self.home_advantage * (self.league_avg_goals / LEAGUE_AVG_DIVISOR)
        lambda_a_p = (attack_a / defense_h_safe) * (self.league_avg_goals / LEAGUE_AVG_DIVISOR)
        poisson_w = ENSEMBLE_WEIGHTS["poisson"]

        # Obtener predicciones de submodelos
        xgb = self._simulate_xgboost(attack_h, defense_a, attack_a, defense_h, form_home, form_away)
        cat = self._simulate_catboost(attack_h, defense_a, attack_a, defense_h, hierarchy)
        lstm = self._simulate_lstm(form_home, form_away, momentum_home, momentum_away)

        # Combinar predicciones ponderadas
        lambda_home = (
            poisson_w * lambda_h_p
            + xgb["weight"] * xgb["lambda_h"]
            + cat["weight"] * cat["lambda_h"]
            + lstm["weight"] * (lambda_h_p * lstm["trend_h"])
        )
        lambda_away = (
            poisson_w * lambda_a_p
            + xgb["weight"] * xgb["lambda_a"]
            + cat["weight"] * cat["lambda_a"]
            + lstm["weight"] * (lambda_a_p * lstm["trend_a"])
        )

        # Calcular matriz de probabilidades
        matrix = self._poisson_matrix(lambda_home, lambda_away)

        # Extraer probabilidades 1X2
        p_home = float(np.sum(np.tril(matrix, -1)))
        p_draw = float(np.trace(matrix))
        p_away = float(np.sum(np.triu(matrix, 1)))

        # Calcular over/under 2.5
        over_25 = 1.0 - float(
            matrix[0, 0] + matrix[0, 1] + matrix[1, 0] + matrix[1, 1]
            + matrix[0, 2] + matrix[2, 0]
        )
        
        # Calcular BTTS
        btts_yes = 1.0 - float(np.sum(matrix[0, :]) + np.sum(matrix[:, 0]) - matrix[0, 0])

        # Encontrar marcador más probable
        max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        most_probable = f"{max_idx[0]}-{max_idx[1]}"
        most_prob_value = float(matrix[max_idx])

        return {
            "home_team": home_team,
            "away_team": away_team,
            "lambda_home": round(lambda_home, 3),
            "lambda_away": round(lambda_away, 3),
            "prob_1X2": {
                "home": round(p_home * 100, 1),
                "draw": round(p_draw * 100, 1),
                "away": round(p_away * 100, 1),
            },
            "most_probable_score": most_probable,
            "most_probable_prob": round(most_prob_value * 100, 1),
            "over_2_5": round(over_25 * 100, 1),
            "under_2_5": round((1 - over_25) * 100, 1),
            "btts_yes": round(btts_yes * 100, 1),
            "btts_no": round((1 - btts_yes) * 100, 1),
            "ensemble_weights": {
                "poisson": poisson_w,
                "xgboost_sim": xgb["weight"],
                "catboost_sim": cat["weight"],
                "lstm_sim": lstm["weight"],
            },
            "league_avg_goals": round(self.league_avg_goals, 3),
        }

    def predict_batch(self, fixtures: list[dict]) -> list[dict]:
        """Realiza predicciones en batch."""
        results = []
        for f in fixtures:
            pred = self.predict(
                home_team=f["home"],
                away_team=f["away"],
                form_home=f.get("form_home", 0.55),
                form_away=f.get("form_away", 0.45),
                momentum_home=f.get("momentum_home", 0.50),
                momentum_away=f.get("momentum_away", 0.50),
                hierarchy=f.get("hierarchy", 1.0),
            )
            results.append(pred)
        return results

    def to_json(self, prediction: dict, indent: int = 2) -> str:
        """Serializa predicción a JSON."""
        return json.dumps(prediction, indent=indent, ensure_ascii=False)
