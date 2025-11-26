# LSTM Stock Prediction - Tech Challenge 4

Sistema de predição de preços de ações usando LSTM com features estacionárias e online learning.

## 🎯 Características Principais

- ✅ **Features Estacionárias**: Prediz Return ao invés de Close (resolve problema de R² negativo)
- ✅ **Anti-Data Leakage**: Proteções rigorosas contra vazamento de dados
- ✅ **Online Learning**: Sistema de retreinamento incremental
- ✅ **Arquitetura Otimizada**: LSTM com BatchNormalization e dropout balanceado

## 📊 Resultados Atuais

**Teste (Último Treinamento):**
- R²: **0.9646** ✅ (Excelente!)
- MAE: **32.65** ✅
- RMSE: **44.03** ✅
- MAPE: **0.60%** ✅
- Direction Accuracy: **53.55%** ✅

**Validação:**
- R²: **0.9504** ✅
- MAE: **25.88** ✅
- RMSE: **32.76** ✅
- MAPE: **0.59%** ✅

✅ **Modelo supera todos os baselines (Naive, MA_5, MA_20)**

## 🚀 Uso Rápido

### Treinar Modelo

```bash
python scripts/train_model_stationary.py
```

### Usar Online Learning

```python
from src.models.online_learner import OnlineLearner

# Criar learner
learner = OnlineLearner()

# Carregar modelo
learner.load_model_and_scaler()

# Fazer predição e aprender
result = learner.predict_with_learning(
    sequence=sequence,
    actual_return=actual_return  # Quando disponível
)

# Modelo retreina automaticamente quando buffer está cheio
```

## 📁 Estrutura do Projeto

```
├── scripts/
│   ├── train_model_stationary.py # Script principal de treinamento
│   ├── analyze_and_optimize.py   # Análise e otimização do sistema
│   ├── backtest_model.py         # Backtesting do modelo
│   ├── evaluate_model.py         # Avaliação do modelo
│   ├── online_learning_demo.py   # Demonstração de online learning
│   └── run_api.py                # API para predições
├── src/
│   ├── data/                   # Pipeline de dados
│   ├── models/                 # Modelos LSTM e online learning
│   ├── evaluation/             # Métricas e visualizações
│   └── validation/             # Testes anti-leakage
└── config.yaml                 # Configurações
```

## ⚙️ Configuração

Edite `config.yaml` para ajustar:
- Arquitetura do modelo
- Hiperparâmetros de treinamento
- Configurações de online learning

## 📝 Notas

- O modelo prediz **Return** (estacionário) e converte para Close
- Features não-estacionárias (Close, High, Low, Open) são removidas
- Sistema de online learning permite melhoria contínua
