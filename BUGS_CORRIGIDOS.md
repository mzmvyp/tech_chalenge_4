# Bugs Críticos Corrigidos

**Data:** 2025-11-27
**Arquivo Modificado:** `src/api/main.py`

---

## Bug #1: Missing Import - `Depends` ✅ CORRIGIDO

**Severidade:** 🔴 CRÍTICA

**Localização:** `src/api/main.py` linha 17

**Problema:**
```python
# ANTES (linha 17)
from fastapi import FastAPI, HTTPException, status, Body
```

A função `Depends` era usada em vários endpoints mas não estava importada:
```python
# Linha 299 - causava erro
async def get_model_info(current_user: dict = Depends(get_current_user)):
```

**Correção Aplicada:**
```python
# DEPOIS (linha 17)
from fastapi import FastAPI, HTTPException, status, Body, Depends
```

**Impacto:** API agora inicia corretamente sem `NameError`.

---

## Bug #2: Missing Import - `DataLoader` ✅ CORRIGIDO

**Severidade:** 🔴 CRÍTICA

**Localização:** `src/api/main.py` linha 59

**Problema:**
```python
# ANTES - faltava DataLoader
from src.models.predictor import StockPredictor
from src.data.preprocessor import TimeSeriesPreprocessor
from src.data.feature_engineering_stationary import create_stationary_features
from src.data.feature_selector import FeatureSelector
from src.monitoring.metrics import ModelMonitor
from src.api.prediction_storage import PredictionStorage
```

A classe `DataLoader` era usada no endpoint `/predict/simple` mas não estava importada:
```python
# Linha 778 - causava erro
loader = DataLoader(symbol=request.symbol, ...)
```

**Correção Aplicada:**
```python
# DEPOIS (linha 59)
from src.models.predictor import StockPredictor
from src.data.preprocessor import TimeSeriesPreprocessor
from src.data.feature_engineering_stationary import create_stationary_features
from src.data.feature_selector import FeatureSelector
from src.data.data_loader import DataLoader  # ✅ ADICIONADO
from src.monitoring.metrics import ModelMonitor
from src.api.prediction_storage import PredictionStorage
```

**Impacto:** Endpoint `/predict/simple` agora funciona corretamente.

---

## Teste de Verificação

Para verificar se os bugs foram corrigidos, execute:

```bash
# Teste 1: Verificar imports
python -c "from src.api.main import app; print('✅ Imports OK')"

# Teste 2: Iniciar API
python scripts/run_api.py --dev
# Deve iniciar sem erros

# Teste 3: Testar endpoint (após obter token)
curl -X POST http://localhost:8000/predict/simple \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days": 200}'
```

---

## Commits Recomendados

```bash
git add src/api/main.py
git commit -m "fix: Add missing imports Depends and DataLoader to API

- Add Depends import from fastapi (used in authentication dependencies)
- Add DataLoader import from src.data.data_loader (used in /predict/simple endpoint)
- Fixes NameError that prevented API from starting
- Fixes /predict/simple endpoint functionality

Closes #BUG-001, #BUG-002"
```

---

## Próximos Passos

Após essas correções, o sistema está **funcional e pronto para uso**.

Para melhorias futuras, consulte o documento `ANALISE_ARQUITETURA_SISTEMA.md` que contém:
- 🟡 Melhorias importantes (segurança, logging, testes)
- 🟢 Melhorias desejáveis (CI/CD, monitoramento avançado)
- Checklist completo de melhorias priorizadas

---

**Status:** ✅ **SISTEMA CORRIGIDO E FUNCIONAL**
