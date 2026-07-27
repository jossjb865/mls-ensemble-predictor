"""
iSportsAPI Client – extrae data necesaria para el Ensemble MLS
Requiere API_KEY de isportsapi.com (plan Live Data o Stats)
"""

from __future__ import annotations
"""
iSportsAPI Client – extrae data necesaria para el Ensemble MLS
Requiere API_KEY de isportsapi.com (plan Live Data o Stats)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .constants import BASE_URLS, STAT_TYPES, XG_PROXY_FACTOR
from .exceptions import ConfigurationError, iSportsAPIError

logger = logging.getLogger(__name__)

# Cache de datos para evitar múltiples llamadas durante sesión
_CACHE: Dict[str, tuple[pd.DataFrame, datetime]] = {}
CACHE_EXPIRY_MINUTES = 60


class iSportsClient:
    """Cliente para interactuar con iSportsAPI."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("ISPORTS_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "ISPORTS_API_KEY not configured. "
                "Set the environment variable or pass it to the constructor. "
                "For GitHub Actions, add it as a repository secret named 'ISPORTS_API_KEY'."
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
                logger.debug(f"Fetching {url}")
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") == 0:
                    logger.debug(f"Success: {path}")
                    return data
                last_error = data.get("message", "Unknown error")
                logger.debug(f"API returned error code {data.get('code')}: {last_error}")
            except requests.exceptions.Timeout:
                last_error = f"Timeout connecting to {base}"
                logger.debug(last_error)
                continue
            except requests.exceptions.ConnectionError:
                last_error = f"Connection error to {base}"
                logger.debug(last_error)
                continue
            except requests.RequestException as e:
                last_error = str(e)
                logger.debug(f"Request failed: {e}")
                continue
        
        raise iSportsAPIError(f"iSportsAPI error after all retries: {last_error}")

    def find_mls_league_id(self) -> str:
        """Busca el leagueId de Major League Soccer."""
        logger.info("Searching for MLS league ID...")
        try:
            data = self._get("/sport/football/league/basic")
        except iSportsAPIError as e:
            logger.error(f"Failed to fetch leagues: {e}")
            raise iSportsAPIError(
                "Could not fetch league list. Verify your ISPORTS_API_KEY is valid and the API service is accessible."
            ) from e
        
        leagues = data.get("data", [])
        if not leagues:
            raise iSportsAPIError("No leagues returned from API")
        
        for lg in leagues:
            name = (lg.get("name") or "").lower()
            short = (lg.get("shortName") or "").lower()
            if "major league soccer" in name or short in ("mls", "usa mls", "usa major"):
                league_id = str(lg["leagueId"])
                logger.info(f"MLS leagueId found: {league_id}")
                return league_id
        
        raise iSportsAPIError(
            f"MLS not found in {len(leagues)} leagues. Available leagues: "
            f"{[lg.get('name') for lg in leagues[:5]]}"
        )

    def get_standings(self, league_id: str) -> List[Dict[str, Any]]:
        """Obtiene standings de la liga (gf, ga, matches, points)."""
        logger.info(f"Fetching standings for league {league_id}...")
        try:
            data = self._get(
                "/sport/football/standing/league",
                {"leagueId": league_id},
            )
        except iSportsAPIError as e:
            logger.error(f"Failed to fetch standings: {e}")
            raise
        
        result = []
        raw = data.get("data", {})
        
        # Normalizar estructura de respuesta (puede variar según plan de API)
        items = raw if isinstance(raw, list) else raw.get("scoreItems", raw.get("total", []))
        if not items and isinstance(raw, dict):
            # Intentar extraer de subLeague si existe
            for key in raw:
                if isinstance(raw[key], list):
                    items = raw[key]
                    logger.debug(f"Found standings in subkey: {key}")
                    break
        
        if not items:
            logger.warning(f"No standings items found. Raw data keys: {raw.keys() if isinstance(raw, dict) else type(raw)}")
        
        for row in items or []:
            team_name = row.get("teamName") or row.get("name", "")
            if not team_name:
                logger.debug(f"Skipping row with no team name: {row}")
                continue
            
            result.append({
                "teamId": str(row.get("teamId", "")),
                "team": team_name,
                "gf": int(row.get("getScore") or row.get("goalsFor") or 0),
                "ga": int(row.get("loseScore") or row.get("goalsAgainst") or 0),
                "matches": int(row.get("totalCount") or row.get("played") or 0),
                "points": int(row.get("integral") or row.get("points") or 0),
                "rank": int(row.get("rank") or 0),
            })
        
        logger.info(f"Standings retrieved for {len(result)} teams")
        if len(result) == 0:
            logger.warning("No teams found in standings")
        return result

    def get_schedule(self, league_id: str, season: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtiene fixtures y resultados (útil para calcular forma reciente)."""
        logger.info(f"Fetching schedule for league {league_id}...")
        params: Dict[str, Any] = {"leagueId": league_id}
        if season:
            params["season"] = season
        
        try:
            data = self._get("/sport/football/schedule", params)
        except iSportsAPIError as e:
            logger.warning(f"Could not fetch schedule: {e}")
            return []
        
        matches = data.get("data", [])
        logger.info(f"Schedule retrieved: {len(matches)} matches")
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
        
        try:
            data = self._get("/sport/football/stats", params)
        except iSportsAPIError as e:
            logger.warning(f"Could not fetch match stats: {e}")
            return []
        
        stats = data.get("data", [])
        logger.info(f"Match stats retrieved: {len(stats)} records")
        return stats

    def build_ensemble_dataset(
        self,
        league_id: Optional[str] = None,
        season: Optional[str] = None,
        output_csv: str = "data/mls_isports_stats.csv",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Construye el DataFrame exacto que necesita el Ensemble:
        team, gf, ga, matches, xg (si disponible)
        """
        # Verificar cache
        if use_cache and output_csv in _CACHE:
            df_cached, timestamp = _CACHE[output_csv]
            elapsed_minutes = (datetime.now() - timestamp).total_seconds() / 60
            if elapsed_minutes < CACHE_EXPIRY_MINUTES:
                logger.info(f"Using cached data from {elapsed_minutes:.1f} minutes ago")
                return df_cached
            else:
                logger.info("Cache expired, fetching fresh data")
                del _CACHE[output_csv]
        
        if league_id is None:
            league_id = self.find_mls_league_id()

        logger.info(f"Building ensemble dataset (leagueId={league_id})...")
        standings = self.get_standings(league_id)

        if not standings:
            raise iSportsAPIError(
                "No teams retrieved from standings. "
                "Check that the API key has access to this league data."
            )

        # Intentar obtener xG desde stats recientes
        logger.info("Extracting xG data from recent match stats...")
        xg_map = self._extract_xg_from_stats(league_id)
        logger.info(f"xG data obtained for {len(xg_map)} teams")

        rows = []
        for s in standings:
            team = s["team"]
            gf = s["gf"]
            ga = s["ga"]
            matches = max(1, s["matches"])
            
            # Prioridad: xG real > proxy calculation
            if team in xg_map:
                xg = xg_map[team]
                logger.debug(f"{team}: using real xG = {xg}")
            else:
                xg = round(gf * XG_PROXY_FACTOR, 1)
                logger.debug(f"{team}: using proxy xG = {xg}")
            
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
            raise iSportsAPIError("Resulting DataFrame is empty")
        
        # Guardar a CSV
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Dataset saved to {output_csv} ({len(df)} teams)")
        
        # Cachear resultado
        _CACHE[output_csv] = (df.copy(), datetime.now())
        
        return df

    def _extract_xg_from_stats(
        self,
        league_id: str,
        days_back: int = 7,
    ) -> Dict[str, float]:
        """Extrae xG acumulado desde los últimos N días."""
        xg_map: Dict[str, float] = {}
        
        # Obtener stats de los últimos N días
        for i in range(days_back):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            try:
                stats = self.get_match_stats(date=date)
                if not stats:
                    logger.debug(f"No stats for {date}")
                    continue
                
                # Procesar xG por equipo
                for stat in stats:
                    team_name = stat.get("teamName", "")
                    if not team_name:
                        continue
                    
                    # Buscar stat type 52 (xG)
                    stat_details = stat.get("stat", [])
                    if not isinstance(stat_details, list):
                        stat_details = [stat_details] if stat_details else []
                    
                    for detail in stat_details:
                        if detail.get("type") == STAT_TYPES["xg"]:
                            xg_value = float(detail.get("value", 0))
                            xg_map[team_name] = xg_map.get(team_name, 0) + xg_value
                            logger.debug(f"{team_name} xG +{xg_value} on {date}")
            except iSportsAPIError as e:
                logger.debug(f"Could not fetch stats for {date}: {e}")
                continue
        
        # Promediar xG por partido (aproximado)
        for team in xg_map:
            xg_map[team] = round(xg_map[team] / days_back, 1)
        
        return xg_map


def fetch_mls_for_ensemble(
    api_key: Optional[str] = None,
    league_id: Optional[str] = None,
    output_csv: str = "data/mls_isports_stats.csv",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Función de conveniencia para obtener datos MLS para el ensemble.
    
    Args:
        api_key: iSportsAPI key (si no se proporciona, usa env var ISPORTS_API_KEY)
        league_id: League ID (si no se proporciona, busca automáticamente MLS)
        output_csv: Ruta de salida del CSV
        use_cache: Si es True, usa datos en caché si son recientes
    
    Returns:
        DataFrame con datos del ensemble
    """
    client = iSportsClient(api_key=api_key)
    return client.build_ensemble_dataset(
        league_id=league_id,
        output_csv=output_csv,
        use_cache=use_cache,
    )


if __name__ == "__main__":
    import sys
    
    # Configurar logging más verbose para debug
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    key = os.getenv("ISPORTS_API_KEY")
    if not key:
        print("ERROR: ISPORTS_API_KEY environment variable not set")
        print("\nTo set it:")
        print("  export ISPORTS_API_KEY='your_key_here'")
        print("\nOr in GitHub Actions, add it as a repository secret.")
        sys.exit(1)
    
    print(f"Using ISPORTS_API_KEY: {key[:10]}...")
    
    try:
        print("\nFetching real MLS data from iSportsAPI...\n")
        df = fetch_mls_for_ensemble(api_key=key, use_cache=False)
        print("\n✅ Success! Retrieved MLS data:\n")
        print(df.to_string())
        print(f"\n📊 Total teams: {len(df)}")
        print(f"\n💾 Data saved to: data/mls_isports_stats.csv")
    except iSportsAPIError as e:
        print(f"❌ API Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .constants import BASE_URLS, STAT_TYPES, XG_PROXY_FACTOR
from .exceptions import ConfigurationError, iSportsAPIError

logger = logging.getLogger(__name__)

# Cache de datos para evitar múltiples llamadas durante sesión
_CACHE: Dict[str, tuple[pd.DataFrame, datetime]] = {}
CACHE_EXPIRY_MINUTES = 60


class iSportsClient:
    """Cliente para interactuar con iSportsAPI."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("ISPORTS_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "ISPORTS_API_KEY not configured. "
                "Set the environment variable or pass it to the constructor. "
                "For GitHub Actions, add it as a repository secret named 'ISPORTS_API_KEY'."
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
                logger.debug(f"Fetching {url}")
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") == 0:
                    logger.debug(f"Success: {path}")
                    return data
                last_error = data.get("message", "Unknown error")
                logger.debug(f"API returned error code {data.get('code')}: {last_error}")
            except requests.exceptions.Timeout:
                last_error = f"Timeout connecting to {base}"
                logger.debug(last_error)
                continue
            except requests.exceptions.ConnectionError:
                last_error = f"Connection error to {base}"
                logger.debug(last_error)
                continue
            except requests.RequestException as e:
                last_error = str(e)
                logger.debug(f"Request failed: {e}")
                continue
        
        raise iSportsAPIError(f"iSportsAPI error after all retries: {last_error}")

    def find_mls_league_id(self) -> str:
        """Busca el leagueId de Major League Soccer."""
        logger.info("Searching for MLS league ID...")
        try:
            data = self._get("/sport/football/league/basic")
        except iSportsAPIError as e:
            logger.error(f"Failed to fetch leagues: {e}")
            raise iSportsAPIError(
                "Could not fetch league list. Verify your ISPORTS_API_KEY is valid and the API service is accessible."
            ) from e
        
        leagues = data.get("data", [])
        if not leagues:
            raise iSportsAPIError("No leagues returned from API")
        
        for lg in leagues:
            name = (lg.get("name") or "").lower()
            short = (lg.get("shortName") or "").lower()
            if "major league soccer" in name or short in ("mls", "usa mls", "usa major"):
                league_id = str(lg["leagueId"])
                logger.info(f"MLS leagueId found: {league_id}")
                return league_id
        
        raise iSportsAPIError(
            f"MLS not found in {len(leagues)} leagues. Available leagues: "
            f"{[lg.get('name') for lg in leagues[:5]]}"
        )

    def get_standings(self, league_id: str) -> List[Dict[str, Any]]:
        """Obtiene standings de la liga (gf, ga, matches, points)."""
        logger.info(f"Fetching standings for league {league_id}...")
        try:
            data = self._get(
                "/sport/football/standing/league",
                {"leagueId": league_id},
            )
        except iSportsAPIError as e:
            logger.error(f"Failed to fetch standings: {e}")
            raise
        
        result = []
        raw = data.get("data", {})
        
        # Normalizar estructura de respuesta (puede variar según plan de API)
        items = raw if isinstance(raw, list) else raw.get("scoreItems", raw.get("total", []))
        if not items and isinstance(raw, dict):
            # Intentar extraer de subLeague si existe
            for key in raw:
                if isinstance(raw[key], list):
                    items = raw[key]
                    logger.debug(f"Found standings in subkey: {key}")
                    break
        
        if not items:
            logger.warning(f"No standings items found. Raw data keys: {raw.keys() if isinstance(raw, dict) else type(raw)}")
        
        for row in items or []:
            team_name = row.get("teamName") or row.get("name", "")
            if not team_name:
                logger.debug(f"Skipping row with no team name: {row}")
                continue
            
            result.append({
                "teamId": str(row.get("teamId", "")),
                "team": team_name,
                "gf": int(row.get("getScore") or row.get("goalsFor") or 0),
                "ga": int(row.get("loseScore") or row.get("goalsAgainst") or 0),
                "matches": int(row.get("totalCount") or row.get("played") or 0),
                "points": int(row.get("integral") or row.get("points") or 0),
                "rank": int(row.get("rank") or 0),
            })
        
        logger.info(f"Standings retrieved for {len(result)} teams")
        if len(result) == 0:
            logger.warning("No teams found in standings")
        return result

    def get_schedule(self, league_id: str, season: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtiene fixtures y resultados (útil para calcular forma reciente)."""
        logger.info(f"Fetching schedule for league {league_id}...")
        params: Dict[str, Any] = {"leagueId": league_id}
        if season:
            params["season"] = season
        
        try:
            data = self._get("/sport/football/schedule", params)
        except iSportsAPIError as e:
            logger.warning(f"Could not fetch schedule: {e}")
            return []
        
        matches = data.get("data", [])
        logger.info(f"Schedule retrieved: {len(matches)} matches")
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
        
        try:
            data = self._get("/sport/football/stats", params)
        except iSportsAPIError as e:
            logger.warning(f"Could not fetch match stats: {e}")
            return []
        
        stats = data.get("data", [])
        logger.info(f"Match stats retrieved: {len(stats)} records")
        return stats

    def build_ensemble_dataset(
        self,
        league_id: Optional[str] = None,
        season: Optional[str] = None,
        output_csv: str = "data/mls_isports_stats.csv",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Construye el DataFrame exacto que necesita el Ensemble:
        team, gf, ga, matches, xg (si disponible)
        """
        # Verificar cache
        if use_cache and output_csv in _CACHE:
            df_cached, timestamp = _CACHE[output_csv]
            elapsed_minutes = (datetime.now() - timestamp).total_seconds() / 60
            if elapsed_minutes < CACHE_EXPIRY_MINUTES:
                logger.info(f"Using cached data from {elapsed_minutes:.1f} minutes ago")
                return df_cached
            else:
                logger.info("Cache expired, fetching fresh data")
                del _CACHE[output_csv]
        
        if league_id is None:
            league_id = self.find_mls_league_id()

        logger.info(f"Building ensemble dataset (leagueId={league_id})...")
        standings = self.get_standings(league_id)

        if not standings:
            raise iSportsAPIError(
                "No teams retrieved from standings. "
                "Check that the API key has access to this league data."
            )

        # Intentar obtener xG desde stats recientes
        logger.info("Extracting xG data from recent match stats...")
        xg_map = self._extract_xg_from_stats(league_id)
        logger.info(f"xG data obtained for {len(xg_map)} teams")

        rows = []
        for s in standings:
            team = s["team"]
            gf = s["gf"]
            ga = s["ga"]
            matches = max(1, s["matches"])
            
            # Prioridad: xG real > proxy calculation
            if team in xg_map:
                xg = xg_map[team]
                logger.debug(f"{team}: using real xG = {xg}")
            else:
                xg = round(gf * XG_PROXY_FACTOR, 1)
                logger.debug(f"{team}: using proxy xG = {xg}")
            
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
            raise iSportsAPIError("Resulting DataFrame is empty")
        
        # Guardar a CSV
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Dataset saved to {output_csv} ({len(df)} teams)")
        
        # Cachear resultado
        _CACHE[output_csv] = (df.copy(), datetime.now())
        
        return df

    def _extract_xg_from_stats(
        self,
        league_id: str,
        days_back: int = 7,
    ) -> Dict[str, float]:
        """Extrae xG acumulado desde los últimos N días."""
        xg_map: Dict[str, float] = {}
        
        # Obtener stats de los últimos N días
        for i in range(days_back):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            try:
                stats = self.get_match_stats(date=date)
                if not stats:
                    logger.debug(f"No stats for {date}")
                    continue
                
                # Procesar xG por equipo
                for stat in stats:
                    team_name = stat.get("teamName", "")
                    if not team_name:
                        continue
                    
                    # Buscar stat type 52 (xG)
                    stat_details = stat.get("stat", [])
                    if not isinstance(stat_details, list):
                        stat_details = [stat_details] if stat_details else []
                    
                    for detail in stat_details:
                        if detail.get("type") == STAT_TYPES["xg"]:
                            xg_value = float(detail.get("value", 0))
                            xg_map[team_name] = xg_map.get(team_name, 0) + xg_value
                            logger.debug(f"{team_name} xG +{xg_value} on {date}")
            except iSportsAPIError as e:
                logger.debug(f"Could not fetch stats for {date}: {e}")
                continue
        
        # Promediar xG por partido (aproximado)
        for team in xg_map:
            xg_map[team] = round(xg_map[team] / days_back, 1)
        
        return xg_map


def fetch_mls_for_ensemble(
    api_key: Optional[str] = None,
    league_id: Optional[str] = None,
    output_csv: str = "data/mls_isports_stats.csv",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Función de conveniencia para obtener datos MLS para el ensemble.
    
    Args:
        api_key: iSportsAPI key (si no se proporciona, usa env var ISPORTS_API_KEY)
        league_id: League ID (si no se proporciona, busca automáticamente MLS)
        output_csv: Ruta de salida del CSV
        use_cache: Si es True, usa datos en caché si son recientes
    
    Returns:
        DataFrame con datos del ensemble
    """
    client = iSportsClient(api_key=api_key)
    return client.build_ensemble_dataset(
        league_id=league_id,
        output_csv=output_csv,
        use_cache=use_cache,
    )


if __name__ == "__main__":
    import sys
    
    # Configurar logging más verbose para debug
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    key = os.getenv("ISPORTS_API_KEY")
    if not key:
        print("ERROR: ISPORTS_API_KEY environment variable not set")
        print("\nTo set it:")
        print("  export ISPORTS_API_KEY='your_key_here'")
        print("\nOr in GitHub Actions, add it as a repository secret.")
        sys.exit(1)
    
    print(f"Using ISPORTS_API_KEY: {key[:10]}...")
    
    try:
        print("\nFetching real MLS data from iSportsAPI...\n")
        df = fetch_mls_for_ensemble(api_key=key, use_cache=False)
        print("\n✅ Success! Retrieved MLS data:\n")
        print(df.to_string())
        print(f"\n📊 Total teams: {len(df)}")
        print(f"\n💾 Data saved to: data/mls_isports_stats.csv")
    except iSportsAPIError as e:
        print(f"❌ API Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
