# MLS Ensemble Football Predictor

Sistema de Machine Learning listo para predicciones de la **Major League Soccer (MLS)** usando un ensemble de modelos:

| Modelo | Peso | Descripción |
|--------|------|-------------|
| Poisson | 25% | Distribución de goles esperados (xG base) |
| XGBoost-sim | 30% | Features tabulares + forma reciente |
| CatBoost-sim | 25% | Robustez categórica / jerarquía |
| LSTM-sim | 20% | Secuencia temporal + momentum |

**Salidas**: Probabilidades 1X2, Over/Under 2.5, BTTS, marcador más probable y lambdas de goles.

## Mejoras en v1.0.1 (Refactorización)

- ✅ **Módulo de constantes**: Centralización de parámetros mágicos en `constants.py`
- ✅ **Excepciones personalizadas**: Sistema de errores robusto en `exceptions.py`
- ✅ **Validador de datos**: Clase `DataValidator` que elimina duplicación
- ✅ **Procesador de estadísticas**: Clase `StatsProcessor` para cálculos de forma
- ✅ **Cache de matrices Poisson**: Mejora de rendimiento
- ✅ **Integración iSportsAPI**: Flujo mejorado con manejo de errores
- ✅ **Logging estruturado**: Trazabilidad completa de ejecución
- ✅ **CLI mejorada**: Soporte para múltiples modos (predicción, batch, iSportsAPI)
- ✅ **Type hints completos**: Mejora de legibilidad y verificación estática
- ✅ **Eliminación de código duplicado**: Máxima reutilización

## Requisitos

```bash
pip install -r requirements.txt
```

## Uso

### Generar datos de ejemplo

```bash
python main.py --generate-sample
```

### Predicción individual

```bash
python main.py --home "Inter Miami CF" --away "LA Galaxy" \
  --form-home 0.70 --form-away 0.45
```

### Predicción en batch

```bash
python main.py --batch fixtures.json --output predictions.json
```

Estructura de `fixtures.json`:
```json
[
  {
    "home": "Inter Miami CF",
    "away": "LA Galaxy",
    "form_home": 0.70,
    "form_away": 0.45,
    "momentum_home": 0.55,
    "momentum_away": 0.40
  }
]
```

### Obtener datos desde iSportsAPI

Primero, configura tu API key:

```bash
export ISPORTS_API_KEY="tu_clave_aqui"
```

Luego obtén los datos:

```bash
python main.py --fetch-isports
```

## Estructura de datos

### CSV de entrada

Archivo: `data/sample_mls_stats.csv`

```csv
team,gf,ga,matches,xg
Inter Miami CF,62,38,28,58.2
LA Galaxy,58,42,28,55.1
```

**Columnas requeridas**:
- `team`: Nombre del equipo (string)
- `gf`: Goles a favor (int)
- `ga`: Goles en contra (int)
- `matches`: Partidos jugados (int)

**Columnas opcionales**:
- `xg`: Expected goals (float) - mejora la precisión del modelo
- `points`: Puntos acumulados
- `rank`: Clasificación
- `teamId`: ID del equipo en iSportsAPI

### Salida de predicción

```json
{
  "home_team": "Inter Miami CF",
  "away_team": "LA Galaxy",
  "lambda_home": 1.842,
  "lambda_away": 1.234,
  "prob_1X2": {
    "home": 52.3,
    "draw": 27.1,
    "away": 20.6
  },
  "most_probable_score": "2-1",
  "most_probable_prob": 15.2,
  "over_2_5": 45.8,
  "under_2_5": 54.2,
  "btts_yes": 38.5,
  "btts_no": 61.5,
  "ensemble_weights": {
    "poisson": 0.25,
    "xgboost_sim": 0.30,
    "catboost_sim": 0.25,
    "lstm_sim": 0.20
  },
  "league_avg_goals": 2.75
}
```

## Arquitectura

```
src/
  __init__.py           Exports públicos
  constants.py          Parámetros centralizados
  exceptions.py         Excepciones personalizadas
  data_loader.py        Carga y validación de datos
  ensemble.py           Modelo ensemble principal
  isports_client.py     Cliente para iSportsAPI
  predict.py            CLI y entrypoint
  
main.py                 Wrapper de entrypoint
requirements.txt        Dependencias
```

## Flujo de datos

```
Datos (CSV o iSportsAPI)
         ↓
  DataValidator
         ↓
 StatsProcessor (ataque, defensa, xG ajustado)
         ↓
EnsembleFootballPredictor.ingest_data()
         ↓
Submodelos:
  ├─ Poisson (25%)
  ├─ XGBoost-sim (30%)
  ├─ CatBoost-sim (25%)
  └─ LSTM-sim (20%)
         ↓
 Combinación ponderada → lambdas finales
         ↓
Matriz de Poisson → probabilidades 1X2, Over/Under, BTTS, etc.
```

## Parámetros ajustables

- **form_home / form_away** (0.0-1.0): Forma actual del equipo
- **momentum_home / momentum_away** (0.0-1.0): Momentum reciente
- **hierarchy** (1.0+): Multiplicador de ventaja local (default=1.0)
- **home_advantage** (default=1.18): Factor de ventaja de local
- **league_avg_goals** (default=2.75): Promedio de goles de la liga

## Desarrollo

### Agregar un nuevo modelo

1. Crear método `_simulate_<modelo>()` en `EnsembleFootballPredictor`
2. Agregar peso en `ENSEMBLE_WEIGHTS` en `constants.py`
3. Integrar en el método `predict()`

### Agregar constantes nuevas

Siempre centralizar en `src/constants.py` usando `Final` para inmutabilidad.

### Tests

Para ejecutar con datos de ejemplo:

```bash
python main.py --generate-sample
python main.py --home "Inter Miami CF" --away "LA Galaxy"
```

## Licencia

MIT
