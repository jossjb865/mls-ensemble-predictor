"""
CLI and batch prediction entrypoint for MLS Ensemble
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ensemble import EnsembleFootballPredictor
from src.data_loader import load_mls_data, create_sample_mls_csv


def run_prediction(
    data_path: str,
    home: str,
    away: str,
    form_home: float = 0.55,
    form_away: float = 0.45,
    momentum_home: float = 0.50,
    momentum_away: float = 0.50,
    hierarchy: float = 1.0,
    output_json: bool = True
) -> Dict[str, Any]:
    model = EnsembleFootballPredictor()
    model.ingest_data(data_path)
    pred = model.predict(
        home_team=home,
        away_team=away,
        form_home=form_home,
        form_away=form_away,
        momentum_home=momentum_home,
        momentum_away=momentum_away,
        hierarchy=hierarchy
    )
    if output_json:
        print(model.to_json(pred))
    return pred


def run_batch(data_path: str, fixtures: List[Dict], output_file: str | None = None) -> List[Dict]:
    model = EnsembleFootballPredictor()
    model.ingest_data(data_path)
    results = model.predict_batch(fixtures)
    if output_file:
        Path(output_file).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Batch predictions saved to {output_file}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    return results


def main():
    parser = argparse.ArgumentParser(description="MLS Ensemble Football Predictor")
    parser.add_argument("--data", type=str, default="data/sample_mls_stats.csv")
    parser.add_argument("--home", type=str)
    parser.add_argument("--away", type=str)
    parser.add_argument("--form-home", type=float, default=0.55)
    parser.add_argument("--form-away", type=float, default=0.45)
    parser.add_argument("--momentum-home", type=float, default=0.50)
    parser.add_argument("--momentum-away", type=float, default=0.50)
    parser.add_argument("--hierarchy", type=float, default=1.0)
    parser.add_argument("--generate-sample", action="store_true")
    parser.add_argument("--batch", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    if args.generate_sample:
        create_sample_mls_csv(args.data)
        print(f"Sample data created at {args.data}")
        return

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data file not found: {data_path}. Generating sample...")
        create_sample_mls_csv(data_path)

    if args.batch:
        fixtures = json.loads(Path(args.batch).read_text())
        run_batch(str(data_path), fixtures, args.output)
        return

    if not args.home or not args.away:
        parser.error("--home and --away are required (or use --batch / --generate-sample)")

    run_prediction(
        data_path=str(data_path),
        home=args.home,
        away=args.away,
        form_home=args.form_home,
        form_away=args.form_away,
        momentum_home=args.momentum_home,
        momentum_away=args.momentum_away,
        hierarchy=args.hierarchy
    )


if __name__ == "__main__":
    main()
