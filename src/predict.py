"""
CLI and batch prediction entrypoint for MLS Ensemble
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import create_sample_mls_csv, load_mls_data
from src.ensemble import EnsembleFootballPredictor
from src.exceptions import EnsembleError
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
    output_file: str | None = None,
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
) -> None:
    """Obtiene datos directamente desde iSportsAPI."""
    logger.info("Fetching data from iSportsAPI...")
    
    try:
        df = fetch_mls_for_ensemble(output_csv=output_csv)
        logger.info(f"Successfully fetched {len(df)} teams from iSportsAPI")
        print(f"\nData saved to {output_csv}")
        print(f"\nTop 10 teams:")
        print(df.head(10).to_string())
    except Exception as e:
        logger.error(f"Error fetching from iSportsAPI: {e}")
        raise


def main() -> None:
    """Punto de entrada principal de la CLI."""
    parser = argparse.ArgumentParser(
        description="MLS Ensemble Football Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python main.py --home "Inter Miami CF" --away "LA Galaxy"
  
  # Batch predictions
  python main.py --batch fixtures.json --output predictions.json
  
  # Generate sample data
  python main.py --generate-sample
  
  # Fetch from iSportsAPI
  python main.py --fetch-isports
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
    parser.add_argument("--data", type=str, default="data/sample_mls_stats.csv", help="Path to data CSV")
    
    # Generación de datos
    parser.add_argument("--generate-sample", action="store_true", help="Generate sample data and exit")
    parser.add_argument("--fetch-isports", action="store_true", help="Fetch data from iSportsAPI and exit")
    
    # Batch
    parser.add_argument("--batch", type=str, default=None, help="Path to batch fixtures JSON")
    parser.add_argument("--output", type=str, default=None, help="Output file for predictions")
    
    args = parser.parse_args()

    try:
        # Generar datos de ejemplo
        if args.generate_sample:
            logger.info(f"Generating sample data at {args.data}...")
            create_sample_mls_csv(args.data)
            print(f"Sample data created at {args.data}")
            return

        # Obtener datos de iSportsAPI
        if args.fetch_isports:
            fetch_from_isports(output_csv=args.data)
            return

        # Validar que exista archivo de datos
        data_path = Path(args.data)
        if not data_path.exists():
            logger.warning(f"Data file not found: {data_path}. Generating sample...")
            create_sample_mls_csv(data_path)

        # Predicción en batch
        if args.batch:
            batch_path = Path(args.batch)
            if not batch_path.exists():
                logger.error(f"Batch file not found: {batch_path}")
                sys.exit(1)
            fixtures = json.loads(batch_path.read_text())
            run_batch(str(data_path), fixtures, args.output)
            return

        # Predicción individual
        if not args.home or not args.away:
            parser.error("--home and --away are required (or use --batch / --generate-sample / --fetch-isports)")

        run_prediction(
            data_path=str(data_path),
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
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
