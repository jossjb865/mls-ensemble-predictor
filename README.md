# MLS Ensemble Football Predictor

Sistema de Machine Learning listo para predicciones de la **Major League Soccer (MLS)** usando un ensemble de modelos:

| Modelo | Peso | Descripción |
|--------|------|-------------|
| Poisson | 25% | Distribución de goles esperados (xG base) |
| XGBoost-sim | 30% | Features tabulares + forma reciente |
| CatBoost-sim | 25% | Robustez categórica / jerarquía |
| LSTM-sim | 20% | Secuencia temporal + momentum |

**Salidas**: Probabilidades 1X2, Over/Under 2.5, BTTS, marcador más probable y lambdas de goles.

## Requisitos

```bash
pip install -r requirements.txt
