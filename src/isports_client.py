"""
iSportsAPI Client – extrae data necesaria para el Ensemble MLS
Requiere API_KEY de isportsapi.com (plan Live Data o Stats)
"""

from __future__ import annotations

import os
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URLS = [
    "http://api.isportsapi.com",
    "http://api2.isportsapi.com",
]

# Códigos de stats relevantes
XG_TYPE = 52          # Expected goals — xG
SHOTS_TYPE = 3
SHOTS_ON_TARGET = 4


class iSportsClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("ISPORTS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ISPORTS_API_KEY no configurada. "
                "Exporta la variable o pásala al constructor."
            )
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MLS-Ensemble/1.0"})

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        params = params or {}
        params["api_key"] = self.api_key

        last_error = None
        for base in BASE_URLS:
            url = f"{base}{path}"
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") == 0:
                    return data
                last_error = data.get("message", "Unknown error")
            except Exception as e:
                last_error = str(e)
                continue
        raise RuntimeError(f"iSportsAPI error: {last_error}")

    def find_mls_league_id(self) -> str:
        """Busca el leagueId de Major League Soccer."""
        data = self._get("/sport/football/league/basic")
        leagues = data.get("data", [])
        for lg in leagues:
            name = (lg.get("name") or "").lower()
            short = (lg.get("shortName") or "").lower()
            if "major league soccer" in name or short in ("mls", "usa mls", "usa major"):
                league_id = str(lg["leagueId"])
                logger.info(f"MLS leagueId encontrado: {league_id}")
                return league_id
        raise ValueError(
            "No se encontró Major League Soccer. "
            "Revisa la lista de ligas o pasa league_id manualmente."
        )

    def get_standings(self, league_id: str) -> List[Dict]:
        """Standing de la liga (gf, ga, matches, points)."""
        data = self._get(
            "/sport/football/standing/league",
            {"leagueId": league_id}
        )
        # La estructura puede variar según sub-ligas; normalizamos
        result = []
        raw = data.get("data", {})
        # Algunos planes devuelven lista directa, otros anidada
        items = raw if isinstance(raw, list) else raw.get("scoreItems", raw.get("total", []))
        if not items and isinstance(raw, dict):
            # Intentar subLeague
            for key in raw:
                if isinstance(raw[key], list):
                    items = raw[key]
                    break
        for row in items or []:
            result.append({
                "teamId": str(row.get("teamId", "")),
                "team": row.get("teamName") or row.get("name", ""),
                "gf": int(row.get("getScore") or row.get("goalsFor") or 0),
                "ga": int(row.get("loseScore") or row.get("goalsAgainst") or 0),
                "matches": int(row.get("totalCount") or row.get("played") or 0),
                "points": int(row.get("integral") or row.get("points") or 0),
                "rank": int(row.get("rank") or 0),
            })
        return result

    def get_schedule(self, league_id: str, season: Optional[str] = None) -> List[Dict]:
        """Fixtures + resultados (últimos partidos para forma)."""
        params = {"leagueId": league_id}
        if season:
            params["season"] = season
        data = self._get("/sport/football/schedule", params)
        matches = data.get("data", [])
        return matches if isinstance(matches, list) else []

    def get_match_stats(self, date: Optional[str] = None, match_id: Optional[str] = None) -> List[Dict]:
        """Stats del día (incluye xG type=52). Limitado a ~1 mes."""
        params = {}
        if date:
            params["date"] = date
        if match_id:
            params["matchId"] = match_id
        data = self._get("/sport/football/stats", params)
        return data.get("data", [])

    def build_ensemble_dataset(
        self,
        league_id: Optional[str] = None,
        season: Optional[str] = None,
        output_csv: str = "data/mls_isports_stats.csv"
    ) -> pd.DataFrame:
        """
        Construye el DataFrame exacto que necesita el Ensemble:
        team, gf, ga, matches, xg (si disponible)
        """
        if league_id is None:
            league_id = self.find_mls_league_id()

        logger.info(f"Extrayendo standings MLS (leagueId={league_id})...")
        standings = self.get_standings(league_id)

        # Agregar xG aproximado desde stats recientes (si el plan lo permite)
        # Nota: el endpoint /stats es principalmente live/día; 
        # para xG de temporada se recomienda acumular históricos o usar otra fuente.
        xg_map: Dict[str, float] = {}

        # Fallback: si no hay xG real, usamos gf * 0.95 como proxy
        rows = []
        for s in standings:
            team = s["team"]
            gf = s["gf"]
            ga = s["ga"]
            matches = max(1, s["matches"])
            xg = xg_map.get(team, round(gf * 0.95, 1))  # proxy si no hay data
            rows.append({
                "team": team,
                "gf": gf,
                "ga": ga,
                "matches": matches,
                "xg": xg,
                "points": s.get("points", 0),
                "rank": s.get("rank", 0),
                "teamId": s.get("teamId", ""),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("No se obtuvieron equipos de standings")

        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Dataset guardado en {output_csv} ({len(df)} equipos)")
        return df


def fetch_mls_for_ensemble(
    api_key: str,
    league_id: Optional[str] = None,
    output_csv: str = "data/mls_isports_stats.csv"
) -> pd.DataFrame:
    """Función de conveniencia."""
    client = iSportsClient(api_key=api_key)
    return client.build_ensemble_dataset(league_id=league_id, output_csv=output_csv)


if __name__ == "__main__":
    import sys
    key = os.getenv("ISPORTS_API_KEY")
    if not key:
        print("Exporta ISPORTS_API_KEY=tu_clave")
        sys.exit(1)
    df = fetch_mls_for_ensemble(key)
    print(df.head(10).to_string())
