# LSTM Stock Prediction - Tech Challenge 4

Sistema de predição de preços de ações usando LSTM com features estacionárias e online learning.

## 🎯 Características Principais

- ✅ **Features Estacionárias**: Prediz Return ao invés de Close (resolve problema de R² negativo)
- ✅ **Anti-Data Leakage**: Proteções rigorosas contra vazamento de dados
- ✅ **Online Learning**: Sistema de retreinamento incremental
- ✅ **Arquitetura Otimizada**: LSTM com BatchNormalization e dropout balanceado

## 📊 Resultados Atuais

**Teste:**
- R²: **0.8031** ✅
- MAE: 81.78
- RMSE: 103.11
- MAPE: 1.50%

## 🚀 Uso Rápido

### Treinar Modelo

```bash
python scripts/train_model.py
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
│   ├── train_model.py          # Script principal de treinamento
│   ├── online_learning_demo.py # Demonstração de online learning
│   └── run_api.py              # API para predições
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
