"""
CLI and batch prediction entrypoint for MLS Ensemble
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import create_sample_mls_csv, load_mls_data
from src.ensemble import EnsembleFootballPredictor
from src.exceptions import ConfigurationError, EnsembleError, iSportsAPIError
from src.isports_client import fetch_mls_for_ensemble

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def run_prediction(
    data_path: str,
    home: str,
    away: str,
    form_home: float = 0.55,
    form_away: float = 0.45,
    momentum_home: float = 0.50,
    momentum_away: float = 0.50,
    hierarchy: float = 1.0,
    output_json: bool = True,
) -> Dict[str, Any]:
    """Ejecuta predicción individual de partido."""
    logger.info(f"Running prediction: {home} vs {away}")
    
    try:
        model = EnsembleFootballPredictor()
        model.ingest_data(data_path)
        pred = model.predict(
            home_team=home,
            away_team=away,
            form_home=form_home,
            form_away=form_away,
            momentum_home=momentum_home,
            momentum_away=momentum_away,
            hierarchy=hierarchy,
        )
        if output_json:
            print(model.to_json(pred))
        return pred
    except EnsembleError as e:
        logger.error(f"Prediction error: {e}")
        raise


def run_batch(
    data_path: str,
    fixtures: List[Dict],
    output_file: Optional[str] = None,
) -> List[Dict]:
    """Ejecuta predicciones en batch."""
    logger.info(f"Running batch predictions for {len(fixtures)} fixtures")
    
    try:
        model = EnsembleFootballPredictor()
        model.ingest_data(data_path)
        results = model.predict_batch(fixtures)
        
        if output_file:
            Path(output_file).write_text(json.dumps(results, indent=2, ensure_ascii=False))
            logger.info(f"Batch predictions saved to {output_file}")
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        
        return results
    except EnsembleError as e:
        logger.error(f"Batch prediction error: {e}")
        raise


def fetch_from_isports(
    output_csv: str = "data/mls_isports_stats.csv",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Obtiene datos directamente desde iSportsAPI usando la API key del environment."""
    logger.info("Fetching real MLS data from iSportsAPI...")
    
    try:
        # fetch_mls_for_ensemble usará automáticamente ISPORTS_API_KEY del environment
        df = fetch_mls_for_ensemble(
            api_key=None,  # None = usa env var ISPORTS_API_KEY
            output_csv=output_csv,
            use_cache=not force_refresh,
        )
        logger.info(f"Successfully fetched {len(df)} teams from iSportsAPI")
        print(f"\n✅ Data successfully fetched and saved to {output_csv}")
        print(f"\n📊 Top 10 teams by goals scored:")
        print(df.nlargest(10, 'gf')[['team', 'gf', 'ga', 'matches', 'xg']].to_string(index=False))
        return df
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\n❌ Configuration Error: {e}")
        print("\n💡 To fix this:")
        print("   1. Local: export ISPORTS_API_KEY='your_key'")
        print("   2. GitHub: Add 'ISPORTS_API_KEY' as a repository secret")
        raise
    except iSportsAPIError as e:
        logger.error(f"API error fetching from iSportsAPI: {e}")
        print(f"\n❌ API Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching from iSportsAPI: {e}")
        print(f"\n❌ Error: {e}")
        raise


def ensure_data_exists(
    data_path: str,
    try_isports: bool = True,
) -> str:
    """Asegura que exista archivo de datos, intentando iSportsAPI primero."""
    path = Path(data_path)
    
    if path.exists():
        logger.info(f"Data file exists: {data_path}")
        return str(path)
    
    logger.warning(f"Data file not found: {data_path}")
    
    # Intentar obtener datos de iSportsAPI si se configuró
    if try_isports and os.getenv("ISPORTS_API_KEY"):
        logger.info("Attempting to fetch from iSportsAPI...")
        try:
            fetch_from_isports(output_csv=data_path, force_refresh=False)
            logger.info("Successfully fetched data from iSportsAPI")
            return str(path)
        except (ConfigurationError, iSportsAPIError) as e:
            logger.warning(f"Could not fetch from iSportsAPI: {e}. Using sample data instead.")
    
    # Fallback: generar datos de ejemplo
    logger.info(f"Generating sample data at {data_path}...")
    create_sample_mls_csv(data_path)
    print(f"\n⚠️  Generated sample data at {data_path}")
    print("   For real predictions, use: python main.py --fetch-isports")
    
    return str(path)


def main() -> None:
    """Punto de entrada principal de la CLI."""
    parser = argparse.ArgumentParser(
        description="MLS Ensemble Football Predictor - Real-time predictions powered by iSportsAPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction (auto-fetches from iSportsAPI if configured)
  python main.py --home "Inter Miami CF" --away "LA Galaxy"
  
  # Batch predictions
  python main.py --batch fixtures.json --output predictions.json
  
  # Generate sample data
  python main.py --generate-sample
  
  # Fetch fresh data from iSportsAPI
  python main.py --fetch-isports
  
  # Use specific data file
  python main.py --data custom_data.csv --home "Inter Miami CF" --away "LA Galaxy"
  
Configuration:
  ISPORTS_API_KEY environment variable or GitHub repository secret
        """,
    )
    
    # Predicción individual
    parser.add_argument("--home", type=str, help="Home team name")
    parser.add_argument("--away", type=str, help="Away team name")
    parser.add_argument("--form-home", type=float, default=0.55, help="Home team form (0.0-1.0)")
    parser.add_argument("--form-away", type=float, default=0.45, help="Away team form (0.0-1.0)")
    parser.add_argument("--momentum-home", type=float, default=0.50, help="Home team momentum (0.0-1.0)")
    parser.add_argument("--momentum-away", type=float, default=0.50, help="Away team momentum (0.0-1.0)")
    parser.add_argument("--hierarchy", type=float, default=1.0, help="Home team hierarchy multiplier")
    
    # Datos
    parser.add_argument(
        "--data",
        type=str,
        default="data/mls_isports_stats.csv",
        help="Path to data CSV (tries iSportsAPI first if missing)",
    )
    
    # Generación de datos
    parser.add_argument(
        "--generate-sample",
        action="store_true",
        help="Generate sample data and exit",
    )
    parser.add_argument(
        "--fetch-isports",
        action="store_true",
        help="Fetch real data from iSportsAPI (requires ISPORTS_API_KEY)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cache when fetching from iSportsAPI",
    )
    
    # Batch
    parser.add_argument("--batch", type=str, default=None, help="Path to batch fixtures JSON")
    parser.add_argument("--output", type=str, default=None, help="Output file for predictions")
    
    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")

    try:
        # Generar datos de ejemplo
        if args.generate_sample:
            logger.info(f"Generating sample data at {args.data}...")
            create_sample_mls_csv(args.data)
            print(f"✅ Sample data created at {args.data}")
            return

        # Obtener datos de iSportsAPI
        if args.fetch_isports:
            try:
                fetch_from_isports(
                    output_csv=args.data,
                    force_refresh=args.refresh,
                )
                print("✅ Ready for predictions with real MLS data!\n")
                print(f"   python main.py --data {args.data} --home 'Inter Miami CF' --away 'LA Galaxy'")
            except (ConfigurationError, iSportsAPIError) as e:
                logger.error(f"Failed to fetch from iSportsAPI: {e}")
                sys.exit(1)
            return

        # Asegurar que exista datos (intenta iSportsAPI primero)
        data_path = ensure_data_exists(args.data, try_isports=True)

        # Predicción en batch
        if args.batch:
            batch_path = Path(args.batch)
            if not batch_path.exists():
                logger.error(f"Batch file not found: {batch_path}")
                sys.exit(1)
            fixtures = json.loads(batch_path.read_text())
            run_batch(data_path, fixtures, args.output)
            return

        # Predicción individual
        if not args.home or not args.away:
            parser.error("--home and --away are required (or use --batch / --generate-sample / --fetch-isports)")

        run_prediction(
            data_path=data_path,
            home=args.home,
            away=args.away,
            form_home=args.form_home,
            form_away=args.form_away,
            momentum_home=args.momentum_home,
            momentum_away=args.momentum_away,
            hierarchy=args.hierarchy,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except (ConfigurationError, iSportsAPIError) as e:
        logger.error(f"API/Configuration error: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
