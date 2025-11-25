# 🏗️ Arquitetura do Sistema

## Visão Geral

Este documento descreve a arquitetura completa do sistema de predição de preços de ações usando LSTM.

---

## 1. Arquitetura em Camadas

```
┌─────────────────────────────────────────────────────────────┐
│                      CAMADA DE APRESENTAÇÃO                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │   FastAPI      │  │  Swagger UI    │  │  Notebooks     │ │
│  │   REST API     │  │  (Docs)        │  │  Jupyter       │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     CAMADA DE APLICAÇÃO                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │  Predictor     │  │   Trainer      │  │  Evaluator     │ │
│  │  (inference)   │  │  (training)    │  │  (metrics)     │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      CAMADA DE MODELO                        │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │   LSTM Model   │  │   Callbacks    │  │   Optimizer    │ │
│  │  (TensorFlow)  │  │  (ES, LR, MC)  │  │    (Adam)      │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    CAMADA DE PROCESSAMENTO                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │ Preprocessor   │  │ Feature Eng.   │  │  Validator     │ │
│  │ (anti-leakage) │  │  (features)    │  │  (leakage)     │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                       CAMADA DE DADOS                        │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │  Data Loader   │  │   yfinance     │  │   Storage      │ │
│  │  (download)    │  │   (API)        │  │  (CSV, H5)     │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Fluxo de Dados

### 2.1 Pipeline de Treinamento

```
┌──────────────┐
│   Config     │
│  (YAML)      │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ 1. DATA COLLECTION                                       │
│    ┌──────────────┐         ┌──────────────┐           │
│    │  S&P 500     │         │     VIX      │           │
│    │  (yfinance)  │         │  (yfinance)  │           │
│    └──────┬───────┘         └──────┬───────┘           │
│           │                        │                    │
│           └────────────┬───────────┘                    │
│                        ▼                                │
│               ┌─────────────────┐                       │
│               │  Raw Data (CSV) │                       │
│               └────────┬────────┘                       │
└────────────────────────┼──────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────┐
│ 2. FEATURE ENGINEERING                                   │
│    ┌─────────────────────────────────────────────────┐  │
│    │ - OHLCV (básico)                                 │  │
│    │ - VIX (volatilidade)                             │  │
│    │ - Returns (retornos percentuais)                 │  │
│    │ - Volatility (desvio padrão rolling)             │  │
│    │ - Momentum (variação percentual)                 │  │
│    │ - Volume Features (relativo, mudança)            │  │
│    └────────────────────┬────────────────────────────┘  │
│                         ▼                                │
│               ┌──────────────────┐                       │
│               │ Features Dataset │                       │
│               └────────┬─────────┘                       │
└─────────────────────────┼─────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────┐
│ 3. PREPROCESSING (ANTI-LEAKAGE!)                         │
│    ┌─────────────────────────────────────────────────┐  │
│    │ ⚠️  SPLIT TEMPORAL PRIMEIRO                      │  │
│    │    - Train: 70% (2019-01 a 2021-11)             │  │
│    │    - Val:   15% (2021-11 a 2022-04)             │  │
│    │    - Test:  15% (2022-04 a 2024-11)             │  │
│    └────────────────────┬────────────────────────────┘  │
│                         ▼                                │
│    ┌─────────────────────────────────────────────────┐  │
│    │ ⚠️  FIT SCALER APENAS NO TREINO                  │  │
│    │    - MinMaxScaler(0, 1)                          │  │
│    │    - fit() no train                              │  │
│    │    - transform() em val e test                   │  │
│    └────────────────────┬────────────────────────────┘  │
│                         ▼                                │
│    ┌─────────────────────────────────────────────────┐  │
│    │ CRIAR SEQUÊNCIAS TEMPORAIS                       │  │
│    │    - Sequência: 60 dias                          │  │
│    │    - Alvo: dia 61 (Close)                        │  │
│    └────────────────────┬────────────────────────────┘  │
│                         ▼                                │
│         ┌────────────────────────────────┐              │
│         │ X_train, y_train               │              │
│         │ X_val, y_val                   │              │
│         │ X_test, y_test                 │              │
│         └────────────┬───────────────────┘              │
└──────────────────────┼──────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│ 4. MODEL TRAINING                                        │
│    ┌─────────────────────────────────────────────────┐  │
│    │ LSTM Architecture:                               │  │
│    │  - LSTM(128) + Dropout(0.2)                      │  │
│    │  - LSTM(64)  + Dropout(0.2)                      │  │
│    │  - LSTM(32)  + Dropout(0.2)                      │  │
│    │  - Dense(16, relu)                               │  │
│    │  - Dense(1)                                      │  │
│    └────────────────────┬────────────────────────────┘  │
│                         ▼                                │
│    ┌─────────────────────────────────────────────────┐  │
│    │ Training:                                        │  │
│    │  - Optimizer: Adam (lr=0.001)                    │  │
│    │  - Loss: MSE                                     │  │
│    │  - Batch: 32                                     │  │
│    │  - Epochs: 100 (with Early Stopping)            │  │
│    │  - Callbacks: ES, ModelCheckpoint, ReduceLR     │  │
│    └────────────────────┬────────────────────────────┘  │
│                         ▼                                │
│               ┌──────────────────┐                       │
│               │  Trained Model   │                       │
│               │  lstm_model.h5   │                       │
│               └────────┬─────────┘                       │
└─────────────────────────┼─────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────┐
│ 5. EVALUATION                                            │
│    ┌─────────────────────────────────────────────────┐  │
│    │ Métricas:                                        │  │
│    │  - MAE, RMSE, MAPE, R², Direction Accuracy      │  │
│    └────────────────────┬────────────────────────────┘  │
│                         ▼                                │
│    ┌─────────────────────────────────────────────────┐  │
│    │ Baselines:                                       │  │
│    │  - Naive Forecast                                │  │
│    │  - Moving Average (5, 20)                        │  │
│    └────────────────────┬────────────────────────────┘  │
│                         ▼                                │
│    ┌─────────────────────────────────────────────────┐  │
│    │ Anti-Leakage Tests:                              │  │
│    │  - Temporal order                                │  │
│    │  - No overlap                                    │  │
│    │  - Suspicious performance                        │  │
│    │  - Better than baseline                          │  │
│    │  - Scaler validation                             │  │
│    └────────────────────┬────────────────────────────┘  │
│                         ▼                                │
│               ┌──────────────────┐                       │
│               │   Visualizations │                       │
│               │   (PNG files)    │                       │
│               └──────────────────┘                       │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Pipeline de Predição (API)

```
┌──────────────┐
│  HTTP POST   │
│  /predict    │
└──────┬───────┘
       │
       ▼
┌─────────────────────┐
│ Pydantic Validation │
│ - 60 data points    │
│ - OHLCV format      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Feature Engineering │
│ (same as training)  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Normalization       │
│ (saved scaler)      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ LSTM Prediction     │
│ (loaded model)      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Inverse Transform   │
│ (to original scale) │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  JSON Response      │
│  {prediction: ...}  │
└─────────────────────┘
```

---

## 3. Arquitetura do Modelo LSTM

### 3.1 Estrutura das Camadas

```
Input Shape: (60, n_features)
│
├─ LSTM Layer 1
│   ├─ Units: 128
│   ├─ Return Sequences: True
│   └─ Activation: tanh (padrão)
│
├─ Dropout (0.2)
│
├─ LSTM Layer 2
│   ├─ Units: 64
│   ├─ Return Sequences: True
│   └─ Activation: tanh (padrão)
│
├─ Dropout (0.2)
│
├─ LSTM Layer 3
│   ├─ Units: 32
│   ├─ Return Sequences: False
│   └─ Activation: tanh (padrão)
│
├─ Dropout (0.2)
│
├─ Dense Layer 1
│   ├─ Units: 16
│   └─ Activation: ReLU
│
└─ Dense Layer 2 (Output)
    ├─ Units: 1
    └─ Activation: Linear
```

### 3.2 Justificativas de Design

**Por que LSTM?**
- Captura dependências de longo prazo em séries temporais
- Resolve o problema de vanishing gradient de RNNs simples
- Eficaz para padrões temporais complexos

**Por que 3 camadas LSTM?**
- Camada 1 (128): Captura padrões de baixo nível
- Camada 2 (64): Combina padrões em features de médio nível
- Camada 3 (32): Abstração final antes da predição

**Por que Dropout de 0.2?**
- Previne overfitting
- 20% é um valor conservador e eficaz
- Aplicado após cada LSTM para regularização

**Por que Dense final com 16 unidades?**
- Camada de transição entre LSTM e output
- ReLU adiciona não-linearidade
- Permite modelo aprender combinações finais

---

## 4. Padrões de Código

### 4.1 Separação de Responsabilidades

- **Data Layer:** Apenas coleta e armazenamento
- **Processing Layer:** Transformações e features
- **Model Layer:** Definição e treinamento
- **API Layer:** Servir predições
- **Evaluation Layer:** Métricas e visualizações

### 4.2 Configuração Centralizada

Todas as configurações em `config.yaml`:
- Facilita experimentação
- Reprodutibilidade
- Sem hardcoding

### 4.3 Logging e Monitoramento

- Logs informativos em cada etapa
- Validações explícitas
- Métricas salvas em JSON

---

## 5. Decisões Técnicas

### 5.1 Por que S&P 500?

- Mais estável que criptomoedas
- Dados confiáveis e consistentes
- Menos volatilidade = melhor para demonstrar técnica
- Relevância prática

### 5.2 Por que 60 dias de sequência?

- ~3 meses de dados de negociação
- Captura padrões de curto/médio prazo
- Balanço entre contexto e computação

### 5.3 Por que Split 70/15/15?

- 70% treino: Dados suficientes para aprender
- 15% validação: Early stopping e tuning
- 15% teste: Avaliação final não enviesada

### 5.4 Por que MinMaxScaler(0,1)?

- LSTM funciona melhor com dados normalizados
- Range [0,1] evita saturação de ativações
- Simples e eficaz

---

## 6. Segurança e Produção

### 6.1 Segurança

- Validação de inputs com Pydantic
- Tratamento de exceções
- Limits de rate (TODO em produção)

### 6.2 Escalabilidade

- API stateless
- Modelo carregado uma vez (startup)
- Batch predictions para múltiplas requisições

### 6.3 Monitoramento

- Health checks
- Logs estruturados
- Métricas de performance

---

## 7. Melhorias Futuras

### Curto Prazo
- [ ] Cache de predições
- [ ] Rate limiting
- [ ] Autenticação JWT

### Médio Prazo
- [ ] Ensemble de modelos
- [ ] A/B testing
- [ ] Feature store

### Longo Prazo
- [ ] AutoML para hyperparameter tuning
- [ ] Predições em tempo real (streaming)
- [ ] Multi-asset support

---

**Última atualização:** 2024-11-16
