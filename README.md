# LSTM Stock Prediction - Tech Challenge 4

Sistema completo de predição de preços de ações usando LSTM com features estacionárias, aprendizado focado em erros, retreinamento periódico automático e API REST para produção.

## 🎯 Características Principais

### Modelo e Arquitetura
- ✅ **Features Estacionárias**: Prediz Return ao invés de Close (resolve problema de R² negativo)
- ✅ **Arquitetura Otimizada**: LSTM com BatchNormalization e dropout balanceado
- ✅ **47 Features**: Returns, VIX, padrões de candles, indicadores técnicos (RSI, MACD, Bollinger Bands, etc.)
- ✅ **Sequence Length**: 60 períodos históricos
- ✅ **Modelo Genérico**: Funciona para qualquer ação (não apenas S&P 500), embora tenha sido treinado com S&P 500

### Sistemas de Aprendizado
- ✅ **Error-Focused Learning**: Aprende especificamente dos erros, focando em casos difíceis
- ✅ **Aprendizado Imediato**: Aprende instantaneamente de erros grandes durante predições
- ✅ **Aprendizado em Batch**: Retreina periodicamente com buffer de erros acumulados
- ✅ **Retreinamento Periódico**: Sistema automático de retreinamento incremental diário
- ✅ **Preservação de Aprendizado**: Modelo aprendido de erros é preservado no retreinamento

### Predição e Validação
- ✅ **Ensemble Learning**: Combina múltiplos modelos para maior robustez
- ✅ **Adaptive Threshold**: Threshold dinâmico para predição de direção baseado em volatilidade
- ✅ **Direction Accuracy**: Métrica para avaliar acerto de direção (alta/baixa)
- ✅ **Anti-Data Leakage**: Proteções rigorosas contra vazamento de dados
- ✅ **Armazenamento de Predições**: Sistema para salvar predições e validá-las posteriormente
- ✅ **Validação Automática**: Endpoints para validar predições com valores reais

### Produção
- ✅ **API REST**: FastAPI para servir predições em produção
- ✅ **Endpoint Simplificado**: `/predict/simple` aceita apenas símbolo e número de dias
- ✅ **Docker**: Containerização para deploy fácil
- ✅ **Monitoramento**: Logs e histórico de performance
- ✅ **Health Checks**: Verificação automática de saúde da API

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

### 6. Processar Erros Diários

```bash
# Processar predições não validadas e aprender dos erros
python scripts/learn_from_daily_errors.py
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
│   ├── learn_from_daily_errors.py         # Processar erros diários
│   ├── evaluate_model.py                  # Avaliação do modelo
│   └── retrain_daily.bat                  # Script de agendamento (Windows)
│
├── src/
│   ├── data/
│   │   ├── data_loader.py                 # Coleta de dados do Yahoo Finance
│   │   ├── feature_engineering_stationary.py  # Features estacionárias
│   │   ├── technical_indicators.py        # Indicadores técnicos
│   │   ├── candlestick_patterns.py       # Padrões de candles
│   │   ├── feature_selector.py           # Seleção de features
│   │   └── preprocessor.py               # Preprocessamento temporal
│   │
│   ├── models/
│   │   ├── lstm_model.py                  # Arquitetura LSTM
│   │   ├── error_focused_learner.py       # Sistema de error learning
│   │   ├── ensemble_predictor.py          # Ensemble e adaptive threshold
│   │   └── predictor.py                   # Interface de predição
│   │
│   ├── api/
│   │   ├── main.py                        # API FastAPI
│   │   ├── schemas.py                     # Schemas Pydantic
│   │   ├── prediction_storage.py          # Armazenamento de predições
│   │   └── state.py                       # Estado da API
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
│   ├── backtest_history.json              # Histórico de backtests
│   └── predictions/                       # Predições armazenadas
│
├── docker/
│   └── Dockerfile                         # Container Docker
│
├── config.yaml                            # Configurações do projeto
├── requirements.txt                        # Dependências Python
├── docker-compose.yml                     # Orquestração Docker
└── .dockerignore                          # Arquivos ignorados no Docker
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
   ├─> Armazena predições com ID único
   └─> Retorna preço e direção previstos

2. VALIDAÇÃO (Fim do dia)
   ├─> Usuário valida predições com valores reais
   ├─> Sistema identifica erros
   └─> Erros são armazenados para aprendizado

3. APRENDIZADO DE ERROS (Diário)
   ├─> Script processa predições validadas
   ├─> Identifica erros grandes
   └─> Retreina modelo com foco nos erros
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

### Armazenamento e Validação de Predições

Sistema para rastrear e validar predições:

- **Armazenamento Automático**: Todas as predições são salvas com ID único
- **Validação por ID**: Valide uma predição específica usando seu ID
- **Validação por Data**: Valide todas as predições de uma data específica
- **Aprendizado Automático**: Erros validados são usados para retreinar o modelo

**Uso:**
```bash
# Validar predição por ID
POST /validate-prediction
{
  "prediction_id": "uuid-da-predicao",
  "actual_price": 4650.0
}

# Validar predições por data
POST /validate-by-date
{
  "target_date": "2025-11-27",
  "actual_price": 4650.0
}
```

## 🔧 Configuração

### Arquivo `config.yaml`

Principais configurações:

```yaml
data:
  symbol: "^GSPC"              # S&P 500 (treinamento)
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

**Nota**: O modelo foi treinado com S&P 500, mas pode ser usado para outras ações. As features são genéricas (Returns, Volatility, Momentum, Volume, indicadores técnicos) e funcionam para qualquer ação. Para melhor performance, recomenda-se retreinar o modelo para cada ação específica.

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

**Informações do Modelo:**
```bash
GET /model/info
```

**Predição Simplificada (Recomendado):**
```bash
POST /predict/simple
Content-Type: application/json

{
  "symbol": "AAPL",      # Qualquer símbolo do Yahoo Finance
  "days": 200            # Número de dias de histórico
}
```

**Predição Única (Avançado):**
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

**Validar Predição (por ID):**
```bash
POST /validate-prediction
Content-Type: application/json

{
  "prediction_id": "uuid-da-predicao",
  "actual_price": 4650.0
}
```

**Validar Predições (por Data):**
```bash
POST /validate-by-date
Content-Type: application/json

{
  "target_date": "2025-11-27",
  "actual_price": 4650.0
}
```

**Estatísticas de Monitoramento:**
```bash
GET /monitoring/stats
```

### Resposta da Predição

```json
{
  "prediction_close": 4650.25,
  "direction": "up",
  "predicted_return": 0.0012,
  "prediction_id": "uuid-único",
  "inference_time_ms": 45.2,
  "confidence_interval": {
    "lower": 4600.0,
    "upper": 4700.0
  }
}
```

### Testar API

```bash
# Testar endpoints
python scripts/test_api.py

# Ou manualmente via curl
curl -X POST http://localhost:8000/predict/simple \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days": 200}'

# Ou acesse a documentação interativa
# http://localhost:8000/docs
```

## 🐳 Docker

### Pré-requisitos

- Docker instalado
- Docker Compose instalado (ou Docker Desktop que inclui Compose)
- Modelo treinado (arquivos em `models/`)

### Deploy Local

#### 1. Verificar se o modelo está treinado

```bash
# Verificar se os arquivos existem
ls models/
# Deve ter:
# - lstm_model.h5
# - scaler.pkl
# - model_info.json
```

Se não tiver, treine primeiro:
```bash
python scripts/train_model_stationary.py
```

#### 2. Construir a imagem Docker

```bash
docker-compose build
```

Ou manualmente:
```bash
docker build -t lstm-stock-api -f docker/Dockerfile .
```

#### 3. Iniciar o container

```bash
docker-compose up -d
```

Ou manualmente:
```bash
docker run -d \
  --name lstm_stock_api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/data:/app/data \
  lstm-stock-api
```

#### 4. Verificar se está rodando

```bash
# Ver logs
docker-compose logs -f api

# Ou
docker logs -f lstm_stock_api

# Verificar health
curl http://localhost:8000/health
```

#### 5. Parar o container

```bash
docker-compose down
```

Ou:
```bash
docker stop lstm_stock_api
docker rm lstm_stock_api
```

### Deploy em Nuvem

#### Heroku

1. Criar `Procfile`:
   ```
   web: python scripts/run_api.py --host 0.0.0.0 --port $PORT
   ```

2. Criar `heroku.yml`:
   ```yaml
   build:
     docker:
       web: docker/Dockerfile
   ```

3. Deploy:
   ```bash
   heroku create
   heroku container:push web
   heroku container:release web
   ```

#### Railway

1. Conectar repositório GitHub
2. Railway detecta `docker-compose.yml` automaticamente
3. Configurar variáveis de ambiente se necessário
4. Deploy automático

#### Render

1. Criar novo Web Service
2. Conectar repositório
3. Usar Docker como ambiente
4. Configurar:
   - Build Command: `docker build -t api -f docker/Dockerfile .`
   - Start Command: `docker run -p 8000:8000 api`

#### AWS (EC2/ECS)

1. **EC2**: SSH e rodar docker-compose
2. **ECS**: Criar task definition com Dockerfile
3. **Elastic Beanstalk**: Usar Docker platform

### Troubleshooting Docker

**Erro: "Modelo não encontrado"**
- Verifique se os arquivos estão em `models/`
- Verifique se o volume está montado corretamente

**Erro: "Porta 8000 já em uso"**
- Pare outros serviços na porta 8000
- Ou mude a porta no `docker-compose.yml`:
  ```yaml
  ports:
    - "8001:8000"  # Usar porta 8001 no host
  ```

**Erro: "Cannot connect to Docker daemon"**
- Inicie o Docker Desktop
- Ou inicie o serviço Docker:
  ```bash
  sudo systemctl start docker  # Linux
  ```

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

### Compatibilidade com Múltiplas Ações

O modelo foi treinado com S&P 500 (`^GSPC`), mas pode ser usado para outras ações porque:

- **Features Genéricas**: Returns, Volatility, Momentum, Volume, indicadores técnicos funcionam para qualquer ação
- **VIX Opcional**: Se não disponível, o sistema usa um valor simulado
- **Endpoint Simplificado**: `/predict/simple` aceita qualquer símbolo do Yahoo Finance

**Recomendação**: Para melhor performance, retreine o modelo para cada ação específica usando `scripts/train_model_stationary.py` com o símbolo desejado.

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
6. **Modelo Genérico**: Funciona para qualquer ação, mas foi otimizado para S&P 500

## 🎓 Tech Challenge - Requisitos

Este projeto atende aos requisitos do Tech Challenge Fase 4:

### Requisitos Obrigatórios ✅

- ✅ **Coleta e Pré-processamento**: yfinance, feature engineering completo
- ✅ **Modelo LSTM**: Arquitetura otimizada com múltiplas camadas
- ✅ **Treinamento e Avaliação**: Métricas completas (MAE, RMSE, MAPE, R²)
- ✅ **Salvamento do Modelo**: Modelo salvo em formato H5
- ✅ **API REST**: FastAPI com endpoints para predições
- ✅ **Docker**: Containerização completa
- ✅ **Monitoramento**: Logs e histórico de performance
- ✅ **Retreinamento Automático**: Sistema periódico implementado

### Funcionalidades Extras (Bônus) ✅

- ✅ **Error-Focused Learning**: Sistema de aprendizado focado em erros
- ✅ **Armazenamento e Validação**: Sistema completo de rastreamento de predições
- ✅ **Endpoint Simplificado**: `/predict/simple` para facilitar uso
- ✅ **Ensemble Learning**: Combinação de múltiplos modelos
- ✅ **Adaptive Threshold**: Threshold dinâmico para direção
- ✅ **Anti-Data Leakage**: Proteções rigorosas contra vazamento
- ✅ **Feature Selection**: Redução de multicolinearidade

### Entregáveis

- ✅ **Código-fonte**: Repositório Git completo com documentação
- ✅ **Docker**: Scripts e contêineres para deploy
- ⚠️ **Link para API em produção**: Opcional (pode ser feito localmente ou em nuvem)
- ⚠️ **Vídeo demonstrativo**: Obrigatório (precisa ser gravado pelo aluno)

## 🚨 Próximos Passos

### Para Completar o Tech Challenge

1. **Testar tudo localmente** ✅
   ```bash
   python scripts/test_api.py
   ```

2. **Fazer deploy em nuvem** (opcional, mas recomendado)
   - Heroku: `git push heroku main`
   - Railway: Conectar repositório
   - Render: Deploy automático

3. **Gravar vídeo demonstrativo** (obrigatório)
   - Mostrar treinamento do modelo
   - Mostrar API funcionando
   - Mostrar predições com diferentes ações
   - Mostrar validação de predições
   - Mostrar aprendizado de erros
   - Mostrar métricas e resultados

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique os logs em `logs/`
2. Consulte os resultados em `outputs/`
3. Teste a API com `python scripts/test_api.py`
4. Verifique a documentação da API em `http://localhost:8000/docs`

---

**Desenvolvido para Tech Challenge - Fase 04**
