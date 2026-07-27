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
# Generar datos de ejemplo
python main.py --generate-sample

# Predicción individual
python main.py --home "Inter Miami CF" --away "LA Galaxy" --form-home 0.70 --form-away 0.45

# Batch
python main.py --batch fixtures.json --output predictions.json
team,gf,ga,matches,xg
Inter Miami CF,62,38,28,58.2
LA Galaxy,58,42,28,55.1

---

### data/sample_mls_stats.csv
(Genera automáticamente con `python main.py --generate-sample` o copia el contenido generado por el script `create_sample_mls_csv`.)

---

**Instrucciones finales**  
1. Crea un repositorio vacío en GitHub.  
2. Copia todos los archivos arriba con la estructura exacta.  
3. Haz push.  
4. Ve a **Actions → MLS Ensemble Predictions → Run workflow** e introduce los equipos.  

El sistema queda 100% listo para ejecutar en local y en GitHub Actions.
