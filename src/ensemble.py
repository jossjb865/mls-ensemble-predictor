"""
Ensemble Football Predictor for MLS
Models: Poisson + XGBoost-sim + CatBoost-sim + LSTM-sim
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from typing import Dict, Optional, Union, Any
import json
from pathlib import Path


class EnsembleFootballPredictor:
    """
    Ensemble model for any league (optimized defaults for MLS).
    Weights: Poisson 25% | XGBoost-sim 30% | CatBoost-sim 25% | LSTM-sim 20%
    """

    def __init__(
        self,
        home_advantage: float = 1.18,
        max_goals: int = 8,
        league_avg_goals: float = 2.75
    ):
        self.home_advantage = home_advantage
        self.max_goals = max_goals
        self.league_avg_goals = league_avg_goals
        self.team_stats: Optional[pd.DataFrame] = None

    def ingest_data(self, data: Union[pd.DataFrame, Dict, str, Path]) -> pd.DataFrame:
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        elif isinstance(data, dict):
            df = pd.DataFrame.from_dict(data, orient="index").reset_index()
            df = df.rename(columns={"index": "team"})
        else:
            df = data.copy()

        required = {"team", "gf", "ga", "matches"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["attack"] = df["gf"] / df["matches"].clip(lower=1)
        df["defense"] = df["ga"] / df["matches"].clip(lower=1)

        if "xg" in df.columns:
            df["xg"] = df["xg"].fillna(df["gf"])
            df["attack"] = 0.55 * df["attack"] + 0.45 * (df["xg"] / df["matches"].clip(lower=1))

        total_goals = df["gf"].sum() + df["ga"].sum()
        total_matches = df["matches"].sum()
        if total_matches > 0:
            self.league_avg_goals = max(1.8, total_goals / (2 * total_matches))

        self.team_stats = df.set_index("team")
        return df

    def _poisson_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                matrix[i, j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
        s = matrix.sum()
        return matrix / s if s > 0 else matrix

    def _simulate_xgboost(
        self,
        attack_h: float,
        defense_a: float,
        attack_a: float,
        defense_h: float,
        form_h: float = 0.5,
        form_a: float = 0.5
    ) -> Dict[str, float]:
        defense_a = max(0.35, defense_a)
        defense_h = max(0.35, defense_h)
        base_h = (attack_h / defense_a) * self.home_advantage
        base_a = attack_a / defense_h
        adj_h = base_h * (0.82 + 0.36 * form_h)
        adj_a = base_a * (0.82 + 0.36 * form_a)
        return {
            "lambda_h": max(0.25, adj_h),
            "lambda_a": max(0.25, adj_a),
            "weight": 0.30
        }

    def _simulate_catboost(
        self,
        attack_h: float,
        defense_a: float,
        attack_a: float,
        defense_h: float,
        hierarchy: float = 1.0
    ) -> Dict[str, float]:
        defense_a = max(0.35, defense_a)
        defense_h = max(0.35, defense_h)
        base_h = (attack_h / defense_a) * self.home_advantage * hierarchy
        base_a = attack_a / defense_h
        return {
            "lambda_h": max(0.25, base_h * 0.97),
            "lambda_a": max(0.25, base_a * 1.03),
            "weight": 0.25
        }

    def _simulate_lstm(
        self,
        form_h: float,
        form_a: float,
        momentum_h: float = 0.5,
        momentum_a: float = 0.5
    ) -> Dict[str, float]:
        trend_h = 0.88 + 0.24 * (form_h - 0.5) + 0.18 * (momentum_h - 0.5)
        trend_a = 0.88 + 0.24 * (form_a - 0.5) + 0.18 * (momentum_a - 0.5)
        return {
            "trend_h": max(0.6, min(1.4, trend_h)),
            "trend_a": max(0.6, min(1.4, trend_a)),
            "weight": 0.20
        }

    def predict(
        self,
        home_team: str,
        away_team: str,
        form_home: float = 0.55,
        form_away: float = 0.45,
        momentum_home: float = 0.50,
        momentum_away: float = 0.50,
        hierarchy: float = 1.0
    ) -> Dict[str, Any]:
        if self.team_stats is None:
            raise RuntimeError("No data ingested. Call ingest_data() first.")

        if home_team not in self.team_stats.index:
            raise ValueError(f"Home team '{home_team}' not found in ingested data")
        if away_team not in self.team_stats.index:
            raise ValueError(f"Away team '{away_team}' not found in ingested data")

        stats_h = self.team_stats.loc[home_team]
        stats_a = self.team_stats.loc[away_team]

        attack_h = float(stats_h["attack"])
        defense_h = float(stats_h["defense"])
        attack_a = float(stats_a["attack"])
        defense_a = float(stats_a["defense"])

        defense_a_safe = max(0.35, defense_a)
        defense_h_safe = max(0.35, defense_h)
        lambda_h_p = (attack_h / defense_a_safe) * self.home_advantage * (self.league_avg_goals / 1.35)
        lambda_a_p = (attack_a / defense_h_safe) * (self.league_avg_goals / 1.35)
        poisson_w = 0.25

        xgb = self._simulate_xgboost(attack_h, defense_a, attack_a, defense_h, form_home, form_away)
        cat = self._simulate_catboost(attack_h, defense_a, attack_a, defense_h, hierarchy)
        lstm = self._simulate_lstm(form_home, form_away, momentum_home, momentum_away)

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

        matrix = self._poisson_matrix(lambda_home, lambda_away)

        p_home = float(np.sum(np.tril(matrix, -1)))
        p_draw = float(np.trace(matrix))
        p_away = float(np.sum(np.triu(matrix, 1)))

        over_25 = 1.0 - float(
            matrix[0, 0] + matrix[0, 1] + matrix[1, 0] + matrix[1, 1]
            + matrix[0, 2] + matrix[2, 0]
        )
        btts_yes = 1.0 - float(np.sum(matrix[0, :]) + np.sum(matrix[:, 0]) - matrix[0, 0])

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
                "away": round(p_away * 100, 1)
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
                "lstm_sim": lstm["weight"]
            },
            "league_avg_goals": round(self.league_avg_goals, 3)
        }

    def predict_batch(self, fixtures: list[dict]) -> list[dict]:
        results = []
        for f in fixtures:
            pred = self.predict(
                home_team=f["home"],
                away_team=f["away"],
                form_home=f.get("form_home", 0.55),
                form_away=f.get("form_away", 0.45),
                momentum_home=f.get("momentum_home", 0.50),
                momentum_away=f.get("momentum_away", 0.50),
                hierarchy=f.get("hierarchy", 1.0)
            )
            results.append(pred)
        return results

    def to_json(self, prediction: dict, indent: int = 2) -> str:
        return json.dumps(prediction, indent=indent, ensure_ascii=False)
