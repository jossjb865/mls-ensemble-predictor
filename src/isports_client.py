"""
iSportsAPI Client – extrae data necesaria para el Ensemble MLS
Requiere API_KEY de isportsapi.com (plan Live Data o Stats)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .constants import BASE_URLS, STAT_TYPES, XG_PROXY_FACTOR
from .exceptions import ConfigurationError, iSportsAPIError

logger = logging.getLogger(__name__)


class iSportsClient:
    """Cliente para interactuar con iSportsAPI."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("ISPORTS_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "ISPORTS_API_KEY no configurada. "
                "Exporta la variable o pásala al constructor."
            )
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MLS-Ensemble/1.0"})
        logger.info("iSportsClient initialized")

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Realiza petición GET a la API con fallback entre servidores."""
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
            except requests.RequestException as e:
                last_error = str(e)
                logger.debug(f"Failed to fetch from {url}: {e}")
                continue
        
        raise iSportsAPIError(f"iSportsAPI error: {last_error}")

    def find_mls_league_id(self) -> str:
        """Busca el leagueId de Major League Soccer."""
        logger.info("Searching for MLS league ID...")
        data = self._get("/sport/football/league/basic")
        leagues = data.get("data", [])
        
        for lg in leagues:
            name = (lg.get("name") or "").lower()
            short = (lg.get("shortName") or "").lower()
            if "major league soccer" in name or short in ("mls", "usa mls", "usa major"):
                league_id = str(lg["leagueId"])
                logger.info(f"MLS leagueId found: {league_id}")
                return league_id
        
        raise iSportsAPIError(
            "MLS not found. Check the leagues list or pass league_id manually."
        )

    def get_standings(self, league_id: str) -> List[Dict[str, Any]]:
        """Obtiene standings de la liga (gf, ga, matches, points)."""
        logger.info(f"Fetching standings for league {league_id}...")
        data = self._get(
            "/sport/football/standing/league",
            {"leagueId": league_id},
        )
        
        result = []
        raw = data.get("data", {})
        
        # Normalizar estructura de respuesta (puede variar según plan de API)
        items = raw if isinstance(raw, list) else raw.get("scoreItems", raw.get("total", []))
        if not items and isinstance(raw, dict):
            # Intentar extraer de subLeague si existe
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
        
        logger.info(f"Standings retrieved for {len(result)} teams")
        return result

    def get_schedule(self, league_id: str, season: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtiene fixtures y resultados (útil para calcular forma reciente)."""
        logger.info(f"Fetching schedule for league {league_id}...")
        params: Dict[str, Any] = {"leagueId": league_id}
        if season:
            params["season"] = season
        
        data = self._get("/sport/football/schedule", params)
        matches = data.get("data", [])
        return matches if isinstance(matches, list) else []

    def get_match_stats(
        self,
        date: Optional[str] = None,
        match_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Obtiene stats del día (incluye xG type=52). Limitado a ~1 mes."""
        logger.info("Fetching match stats...")
        params: Dict[str, Any] = {}
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
        output_csv: str = "data/mls_isports_stats.csv",
    ) -> pd.DataFrame:
        """
        Construye el DataFrame exacto que necesita el Ensemble:
        team, gf, ga, matches, xg (si disponible)
        """
        if league_id is None:
            league_id = self.find_mls_league_id()

        logger.info(f"Building ensemble dataset (leagueId={league_id})...")
        standings = self.get_standings(league_id)

        if not standings:
            raise iSportsAPIError("No teams retrieved from standings")

        # Agregar xG aproximado desde stats recientes (si el plan lo permite)
        # Nota: el endpoint /stats es principalmente live/día;
        # para xG de temporada se recomienda acumular históricos.
        xg_map: Dict[str, float] = {}

        rows = []
        for s in standings:
            team = s["team"]
            gf = s["gf"]
            ga = s["ga"]
            matches = max(1, s["matches"])
            # Fallback: si no hay xG real, usamos gf * proxy como aproximación
            xg = xg_map.get(team, round(gf * XG_PROXY_FACTOR, 1))
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
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Dataset saved to {output_csv} ({len(df)} teams)")
        return df


def fetch_mls_for_ensemble(
    api_key: str,
    league_id: Optional[str] = None,
    output_csv: str = "data/mls_isports_stats.csv",
) -> pd.DataFrame:
    """Función de conveniencia para obtener datos MLS para el ensemble."""
    client = iSportsClient(api_key=api_key)
    return client.build_ensemble_dataset(league_id=league_id, output_csv=output_csv)


if __name__ == "__main__":
    import sys
    
    key = os.getenv("ISPORTS_API_KEY")
    if not key:
        print("Set ISPORTS_API_KEY environment variable")
        sys.exit(1)
    
    try:
        df = fetch_mls_for_ensemble(key)
        print("\nTop 10 teams:")
        print(df.head(10).to_string())
    except iSportsAPIError as e:
        print(f"API Error: {e}")
        sys.exit(1)
