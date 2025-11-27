# LSTM Stock Prediction - Tech Challenge 4

Sistema completo de predição de preços de ações usando LSTM com features estacionárias, aprendizado focado em erros, retreinamento periódico automático e API REST para produção.

## 🎯 Características Principais

### Modelo e Arquitetura
- ✅ **Features Estacionárias**: Prediz Return ao invés de Close (resolve problema de R² negativo)
- ✅ **Arquitetura Otimizada**: LSTM com BatchNormalization e dropout balanceado
- ✅ **47 Features**: Returns, VIX, padrões de candles, indicadores técnicos (RSI, MACD, Bollinger Bands, etc.)
- ✅ **Sequence Length**: 60 períodos históricos

### Sistemas de Aprendizado
- ✅ **Error-Focused Learning**: Aprende especificamente dos erros, focando em casos difíceis
- ✅ **Aprendizado Imediato**: Aprende instantaneamente de erros grandes durante predições
- ✅ **Aprendizado em Batch**: Retreina periodicamente com buffer de erros acumulados
- ✅ **Retreinamento Periódico**: Sistema automático de retreinamento incremental diário

### Predição e Validação
- ✅ **Ensemble Learning**: Combina múltiplos modelos para maior robustez
- ✅ **Adaptive Threshold**: Threshold dinâmico para predição de direção baseado em volatilidade
- ✅ **Direction Accuracy**: Métrica para avaliar acerto de direção (alta/baixa)
- ✅ **Anti-Data Leakage**: Proteções rigorosas contra vazamento de dados

### Produção
- ✅ **API REST**: FastAPI para servir predições em produção
- ✅ **Docker**: Containerização para deploy fácil
- ✅ **Monitoramento**: Logs e histórico de performance
- ✅ **Preservação de Aprendizado**: Modelo aprendido de erros é preservado no retreinamento

## 📊 Resultados Atuais

### Métricas de Regressão (Preço)
- **R² Score**: 0.9643 ✅ (Excelente - próximo de 1.0)
- **MAE**: 33.01 pontos
- **RMSE**: 44.24 pontos
- **MAPE**: 0.61% ✅ (Erro percentual muito baixo)

### Métricas de Classificação (Direção)
- **Direction Accuracy**: 43-46% (melhora ao longo do tempo com error learning)
- **Evolução**: Primeira metade ~41%, Segunda metade ~46% (+5% de melhoria)

✅ **Modelo supera todos os baselines (Naive, MA_5, MA_20)**

## 🚀 Uso Rápido

### 1. Treinar Modelo Inicial

```bash
python scripts/train_model_stationary.py
```

### 2. Backtest com Error Learning

```bash
# Backtest simples
python scripts/backtest_with_error_learning.py

# Backtest com ensemble e aprendizado imediato
python scripts/backtest_with_error_learning.py --use-ensemble --immediate-learn --n-tests 500
```

### 3. Retreinamento Periódico

```bash
# Retreinamento incremental (fine-tuning)
python scripts/periodic_retrain.py --mode incremental

# Retreinamento com backtest automático
python scripts/periodic_retrain.py --mode incremental --run-backtest

# Retreinamento completo (do zero)
python scripts/periodic_retrain.py --mode full
```

### 4. Configurar Agendamento Automático

```bash
# Windows
python scripts/setup_retrain_scheduler.py --platform windows

# Linux/Mac
python scripts/setup_retrain_scheduler.py --platform linux
```

Depois, configure o Task Scheduler (Windows) ou Cron (Linux) para executar `scripts/retrain_daily.bat` ou `scripts/retrain_daily.sh` diariamente às 02:00.

### 5. Executar API

```bash
# Desenvolvimento
python scripts/run_api.py --dev

# Produção
python scripts/run_api.py

# Docker
docker-compose up
```

## 📁 Estrutura do Projeto

```
├── scripts/
│   ├── train_model_stationary.py          # Treinamento inicial do modelo
│   ├── backtest_with_error_learning.py    # Backtest com error learning
│   ├── periodic_retrain.py                 # Retreinamento periódico automático
│   ├── setup_retrain_scheduler.py         # Configurar agendamento
│   ├── run_api.py                         # Executar API FastAPI
│   ├── test_api.py                        # Testar endpoints da API
│   ├── evaluate_model.py                  # Avaliação do modelo
│   └── retrain_daily.bat                  # Script de agendamento (Windows)
│
├── src/
│   ├── data/
│   │   ├── data_loader.py                 # Coleta de dados do Yahoo Finance
│   │   ├── feature_engineering_stationary.py  # Features estacionárias
│   │   ├── technical_indicators.py        # Indicadores técnicos
│   │   └── candlestick_patterns.py       # Padrões de candles
│   │
│   ├── models/
│   │   ├── lstm_model.py                  # Arquitetura LSTM
│   │   ├── error_focused_learner.py       # Sistema de error learning
│   │   ├── ensemble_predictor.py          # Ensemble e adaptive threshold
│   │   └── predictor.py                   # Interface de predição
│   │
│   ├── api/
│   │   └── main.py                        # API FastAPI
│   │
│   ├── evaluation/
│   │   └── metrics.py                     # Métricas de avaliação
│   │
│   └── validation/
│       └── anti_leakage_tests.py          # Testes anti-leakage
│
├── models/
│   ├── lstm_model.h5                      # Modelo principal
│   ├── lstm_model_error_learned.h5        # Modelo aprendido de erros
│   ├── scaler.pkl                         # Scaler para normalização
│   └── model_info.json                    # Metadata do modelo
│
├── outputs/
│   ├── backtest_error_learning_results.json
│   ├── backtest_error_learning_predictions.csv
│   └── backtest_history.json              # Histórico de backtests
│
├── config.yaml                            # Configurações do projeto
├── requirements.txt                        # Dependências Python
├── Dockerfile                             # Container Docker
└── docker-compose.yml                     # Orquestração Docker
```

## 🔄 Fluxo de Produção Automático

### Retreinamento Diário (02:00 AM)

```
1. RETREINAMENTO INCREMENTAL
   ├─> Baixa novos dados do mercado
   ├─> Mescla com dados históricos
   ├─> Carrega modelo aprendido (se existir)
   ├─> Fine-tuning conservador (LR baixo)
   └─> Salva em ambos os arquivos (preserva aprendizado)

2. BACKTEST AUTOMÁTICO
   ├─> Executa 100 testes de validação
   ├─> Usa modelo recém-retreinado
   ├─> Calcula Direction Accuracy
   └─> Salva histórico

3. LOGS E MONITORAMENTO
   └─> Registra performance e resultados
```

### Durante o Dia (API em Produção)

```
1. PREDIÇÕES
   ├─> API recebe requisições
   ├─> Modelo faz predições
   └─> Retorna preço e direção previstos

2. ERROR LEARNING (se habilitado)
   ├─> Detecta erros grandes
   ├─> Aprende imediatamente
   └─> Melhora performance ao longo do tempo
```

## 🎯 Sistemas de Aprendizado

### Error-Focused Learning

Sistema que aprende especificamente dos erros:

- **Identificação de Erros**: Detecta predições com erro alto
- **Peso Maior**: Erros recebem peso maior no treinamento
- **Hard Examples**: Foca em casos difíceis
- **Aprendizado Imediato**: Pode aprender instantaneamente de erros grandes
- **Aprendizado em Batch**: Acumula erros e retreina periodicamente

**Uso:**
```python
from src.models.error_focused_learner import ErrorFocusedLearner

learner = ErrorFocusedLearner()
learner.load_model_and_scaler()

# Durante backtest, o sistema aprende automaticamente dos erros
```

### Retreinamento Periódico

Sistema automático de atualização do modelo:

- **Incremental**: Fine-tuning com novos dados (padrão)
- **Full**: Retreinamento completo do zero
- **Preservação**: Mantém aprendizado de erros acumulado
- **Automático**: Agendado para executar diariamente

**Configuração:**
```bash
# Gerar scripts de agendamento
python scripts/setup_retrain_scheduler.py --platform windows

# Executar manualmente
python scripts/periodic_retrain.py --mode incremental --run-backtest
```

## 🔧 Configuração

### Arquivo `config.yaml`

Principais configurações:

```yaml
data:
  symbol: "^GSPC"              # S&P 500
  vix_symbol: "^VIX"           # Volatilidade
  start_date: "2019-01-01"
  end_date: "2024-11-01"

model:
  sequence_length: 60          # Períodos históricos
  lstm_layers: [...]           # Arquitetura LSTM
  learning_rate: 0.001

training:
  batch_size: 32
  epochs: 100
  early_stopping: {...}
```

### Parâmetros de Error Learning

Configuráveis no código:
- `error_threshold_percentile`: 75.0 (percentil para considerar erro alto)
- `error_weight_multiplier`: 2.0 (multiplicador de peso para erros)
- `immediate_learn`: True/False (aprendizado imediato)

## 📡 API REST

### Endpoints

**Health Check:**
```bash
GET /health
```

**Predição Única:**
```bash
POST /predict
Content-Type: application/json

{
  "sequence": [[...], [...], ...],  # 60 períodos x 47 features
  "last_close": 4500.0
}
```

**Predição em Batch:**
```bash
POST /predict/batch
Content-Type: application/json

{
  "sequences": [[[...]], [[...]], ...],
  "last_closes": [4500.0, 4510.0, ...]
}
```

### Testar API

```bash
# Testar endpoints
python scripts/test_api.py

# Ou manualmente
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": [...], "last_close": 4500.0}'
```

## 🐳 Docker

### Build e Run

```bash
# Build
docker build -t lstm-stock-prediction .

# Run
docker run -p 8000:8000 lstm-stock-prediction

# Ou com docker-compose
docker-compose up
```

### Docker Compose

O `docker-compose.yml` inclui:
- API FastAPI
- Volumes para modelos e dados
- Configuração de rede

## 📈 Métricas e Avaliação

### Métricas de Regressão
- **MAE** (Mean Absolute Error): Erro absoluto médio
- **RMSE** (Root Mean Square Error): Erro quadrático médio
- **MAPE** (Mean Absolute Percentage Error): Erro percentual médio
- **R² Score**: Coeficiente de determinação

### Métricas de Classificação
- **Direction Accuracy**: Porcentagem de acertos na direção (alta/baixa)

### Baselines
- **Naive Forecast**: Último valor conhecido
- **MA_5**: Média móvel de 5 períodos
- **MA_20**: Média móvel de 20 períodos

## 🔍 Detalhes Técnicos

### Features Estacionárias

O modelo prediz **Return** (diferença percentual) ao invés de **Close** (preço absoluto):

- **Vantagem**: Return é estacionário, Close não é
- **Conversão**: `predicted_close = last_close * (1 + predicted_return)`
- **Resultado**: R² positivo e alto (0.96+)

### Arquitetura LSTM

```
Input: (60, 47)
  ├─ LSTM(64) + Dropout(0.25) + BatchNorm
  ├─ LSTM(32) + Dropout(0.25) + BatchNorm
  ├─ LSTM(16) + Dropout(0.25) + BatchNorm
  ├─ Dense(16) + Dropout(0.2) + BatchNorm
  ├─ Dense(8) + Dropout(0.2)
  └─ Dense(1)  # Output: Return
```

### Ensemble e Adaptive Threshold

- **Ensemble**: Combina múltiplos modelos (weighted average)
- **Adaptive Threshold**: Ajusta threshold baseado em volatilidade recente
- **Confidence Threshold**: Filtra predições de baixa confiança

## 🛠️ Requisitos

- Python 3.10+
- TensorFlow 2.x
- FastAPI
- yfinance
- pandas, numpy
- scikit-learn

Instalar dependências:
```bash
pip install -r requirements.txt
```

## 📝 Notas Importantes

1. **Preservação de Aprendizado**: O sistema preserva automaticamente o aprendizado de erros durante retreinamento periódico
2. **Detecção Automática**: Backtest e retreinamento detectam automaticamente o modelo mais recente
3. **Windows Compatibility**: Todos os emojis foram removidos para compatibilidade com Windows
4. **Data Leakage**: Sistema tem proteções rigorosas contra vazamento de dados
5. **Temporal Split**: Dados são divididos temporalmente (sem shuffle) para simular produção

## 🎓 Tech Challenge - Requisitos

Este projeto atende aos requisitos do Tech Challenge Fase 4:

- ✅ **Coleta e Pré-processamento**: yfinance, feature engineering completo
- ✅ **Modelo LSTM**: Arquitetura otimizada com múltiplas camadas
- ✅ **Treinamento e Avaliação**: Métricas completas (MAE, RMSE, MAPE, R²)
- ✅ **Salvamento do Modelo**: Modelo salvo em formato H5
- ✅ **API REST**: FastAPI com endpoints para predições
- ✅ **Docker**: Containerização completa
- ✅ **Monitoramento**: Logs e histórico de performance
- ✅ **Retreinamento Automático**: Sistema periódico implementado

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique os logs em `logs/`
2. Consulte os resultados em `outputs/`
3. Teste a API com `python scripts/test_api.py`

---

**Desenvolvido para Tech Challenge - Fase 04**
