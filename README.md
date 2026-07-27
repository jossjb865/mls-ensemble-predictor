# MLS Ensemble Football Predictor

Sistema de Machine Learning listo para predicciones de la **Major League Soccer (MLS)** usando un ensemble de modelos con **datos en tiempo real desde iSportsAPI**:

| Modelo | Peso | Descripción |
|--------|------|-------------|
| Poisson | 25% | Distribución de goles esperados (xG base) |
| XGBoost-sim | 30% | Features tabulares + forma reciente |
| CatBoost-sim | 25% | Robustez categórica / jerarquía |
| LSTM-sim | 20% | Secuencia temporal + momentum |

**Salidas**: Probabilidades 1X2, Over/Under 2.5, BTTS, marcador más probable y lambdas de goles.

## 🚀 Inicio Rápido

### 1. Configurar API Key (Solo si usas local)

**Local:**
```bash
export ISPORTS_API_KEY="tu_clave_aqui"
```

**GitHub Actions:**
- Ya está configurada en repository secrets como `ISPORTS_API_KEY`
- El código la obtiene automáticamente

### 2. Obtener datos en vivo

```bash
python main.py --fetch-isports
```

Esto:
- ✅ Busca automáticamente la liga MLS en iSportsAPI
- ✅ Obtiene standings actualizados (goles, partidos jugados, etc.)
- ✅ Extrae xG de los últimos 7 días de matches
- ✅ Cachea los datos para no agotar cuota de API
- ✅ Guarda todo en `data/mls_isports_stats.csv`

### 3. Realizar predicción

**Usando datos en vivo (intenta iSportsAPI automáticamente):**
```bash
python main.py --home "Inter Miami CF" --away "LA Galaxy"
```

**Con datos custom:**
```bash
python main.py --data data/mls_isports_stats.csv --home "Inter Miami CF" --away "LA Galaxy"
```

## 📊 Características de Integración iSportsAPI

### Datos Automáticamente Extraídos

| Dato | Fuente | Uso |
|------|--------|-----|
| **Goles a favor/en contra** | Standings API | Cálculo de ataque/defensa |
| **Partidos jugados** | Standings API | Normalización de métricas |
| **Expected Goals (xG)** | Match Stats API (últimos 7 días) | Mejora de precisión del ataque |
| **Puntos** | Standings API | Metadata para análisis |
| **Ranking** | Standings API | Validación de datos |

### Caché Inteligente

- **Duración**: 60 minutos
- **Beneficio**: Evita múltiples llamadas innecesarias a la API
- **Control**: Usa `--refresh` para forzar actualización

```bash
# Usar cache (por defecto)
python main.py --fetch-isports

# Forzar actualización
python main.py --fetch-isports --refresh
```

### Manejo Robusto de Errores

- ✅ Usa automáticamente ISPORTS_API_KEY del environment
- ✅ Fallback automático a datos de ejemplo si API_KEY no está configurada
- ✅ Retry automático entre múltiples servidores de iSportsAPI
- ✅ Logs detallados para debugging
- ✅ Mensajes claros sobre qué salió mal

```bash
# Debug detallado
python main.py --debug --fetch-isports
```

## 📋 Ejemplos de Uso

### Predicción Individual
```bash
python main.py --home "Inter Miami CF" --away "LA Galaxy" \
  --form-home 0.70 --form-away 0.45
```

### Predicción en Batch
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
    "form_away": 0.45
  },
  {
    "home": "Columbus Crew",
    "away": "FC Cincinnati",
    "form_home": 0.60,
    "form_away": 0.40
  }
]
```

### Datos de Ejemplo (sin API Key)
```bash
python main.py --generate-sample
python main.py --data data/sample_mls_stats.csv --home "Inter Miami CF" --away "LA Galaxy"
```

## 🔄 Flujo Automático de Datos

```
python main.py --home Team1 --away Team2
        ↓
¿Existe data/mls_isports_stats.csv?
        ├─ Si → Usar archivo existente
        └─ No ↓
         ¿ISPORTS_API_KEY configurada?
         ├─ Si → Conectar iSportsAPI
         │       ├─ Buscar MLS League ID
         │       ├─ Obtener Standings
         │       ├─ Extraer xG
         │       └─ Guardar CSV
         └─ No → Generar datos de ejemplo
        ↓
Cargar datos → Calcular métricas
        ↓
Ensemble (Poisson + XGBoost + CatBoost + LSTM)
        ↓
Salida JSON con predicciones
```

## 📦 Estructura de Datos CSV

### Archivo de entrada: `data/mls_isports_stats.csv`

**Obtenido automáticamente de iSportsAPI:**
```csv
team,gf,ga,matches,xg,points,rank,teamId
Inter Miami CF,62,38,28,58.2,82,1,12345
LA Galaxy,58,42,28,55.1,79,2,12346
Columbus Crew,55,35,28,52.4,76,3,12347
```

**Columnas:**
- `team`: Nombre del equipo (string)
- `gf`: Goles a favor (int) - de standings
- `ga`: Goles en contra (int) - de standings  
- `matches`: Partidos jugados (int) - de standings
- `xg`: Expected Goals (float) - extraído de match stats últimos 7 días
- `points`: Puntos acumulados (int) - de standings
- `rank`: Posición en liga (int) - de standings
- `teamId`: ID en iSportsAPI (string) - para validación

### Salida de predicción: `predictions.json`

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

## 🔧 Configuración Avanzada

### Variables de Entorno

```bash
# Requerida para datos en vivo
export ISPORTS_API_KEY="tu_clave_de_isportsapi"

# Opcional: customize timeouts
export ISPORTS_TIMEOUT="60"  # segundos
```

### Parámetros de Predicción

```python
from src.ensemble import EnsembleFootballPredictor

model = EnsembleFootballPredictor(
    home_advantage=1.18,      # Factor de ventaja de local
    max_goals=8,              # Máximo goles para matriz Poisson
    league_avg_goals=2.75     # Promedio de goles de la liga
)
```

### Forma y Momentum

```bash
# form: 0.0-1.0 (0=muy mala forma, 1=forma excelente)
# momentum: 0.0-1.0 (0=perdiendo últimos partidos, 1=ganando)

python main.py \
  --home "Inter Miami CF" --away "LA Galaxy" \
  --form-home 0.70 --form-away 0.45 \
  --momentum-home 0.60 --momentum-away 0.35
```

## 🏗️ Arquitectura

```
src/
  __init__.py              Package exports
  constants.py             Parámetros centralizados
  exceptions.py            Excepciones personalizadas
  data_loader.py           Carga y validación de datos
  ensemble.py              Modelo ensemble principal
  isports_client.py        Cliente iSportsAPI (MEJORADO)
  predict.py               CLI y entrypoint (MEJORADO)
  
main.py                    Wrapper
requirements.txt           Dependencias
README.md                  Documentación
```

## 📝 GitHub Actions Integration

### Configuración Automática

La API key se obtiene automáticamente desde el environment en GitHub Actions:

```yaml
jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      
      # La API key se pasa automáticamente
      - name: Fetch data from iSportsAPI
        run: python main.py --fetch-isports
      
      - name: Run predictions
        run: |
          python main.py --batch fixtures.json --output predictions.json
```

Gracias a la integración, **no es necesario exportar manualmente la variable**. El código la obtiene directamente.

### Workflow de Ejemplo

```yaml
name: MLS Predictions
on:
  schedule:
    - cron: '0 9 * * SAT'  # Cada sábado a las 9 AM

jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - run: pip install -r requirements.txt
      
      # fetch_mls_for_ensemble() detecta ISPORTS_API_KEY automáticamente
      - name: Fetch data from iSportsAPI
        run: python main.py --fetch-isports
      
      - name: Run batch predictions
        run: python main.py --batch fixtures.json --output predictions.json
      
      - name: Upload predictions
        uses: actions/upload-artifact@v3
        with:
          name: predictions
          path: predictions.json
```

## 🐛 Troubleshooting

### Error: "ISPORTS_API_KEY not configured"

**Local:**
```bash
export ISPORTS_API_KEY="tu_clave"
python main.py --fetch-isports
```

**GitHub Actions:**
Verifica que el secret esté en `Settings → Secrets and variables → Actions`

### Error: "MLS not found"

**Causas:**
- API key sin acceso a datos de MLS
- API key expirada o inválida

**Solución:**
```bash
python main.py --debug --fetch-isports  # Ver logs detallados
```

### Cache desactualizado

```bash
python main.py --fetch-isports --refresh  # Forzar actualización
```

## 📄 Licencia

MIT
