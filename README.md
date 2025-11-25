# 📊 LSTM Stock Prediction - Tech Challenge Fase 04

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Projeto completo de **Deep Learning** para predição de preços de ações usando **redes neurais LSTM**, desenvolvido como parte do Tech Challenge da Fase 04 da pós-graduação em Machine Learning Engineering na POSTECH FIAP.

---

## 🎯 Objetivo do Projeto

Desenvolver um modelo preditivo de **redes neurais Long Short Term Memory (LSTM)** para predizer o valor de fechamento do **S&P 500 (^GSPC)** e realizar toda a pipeline de desenvolvimento, desde a criação do modelo até o deploy em uma API REST.

---

## ✨ Características Principais

- ✅ **Modelo LSTM** com múltiplas camadas e Dropout
- ✅ **Proteção Anti-Data-Leakage** rigorosa em todo o pipeline
- ✅ **API RESTful** com FastAPI para servir predições
- ✅ **Métricas completas**: MAE, RMSE, MAPE, R², Direction Accuracy
- ✅ **Comparação com Baselines** (Naive Forecast, Moving Average)
- ✅ **Visualizações detalhadas** de resultados
- ✅ **Testes automatizados** para validar ausência de leakage
- ✅ **Dockerização** para fácil deploy
- ✅ **Documentação completa** e código comentado

---

## 📂 Estrutura do Projeto

```
lstm-stock-prediction/
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências
├── config.yaml                        # Configurações centralizadas
├── docker-compose.yml                 # Docker Compose
│
├── data/
│   ├── raw/                          # Dados brutos baixados
│   └── processed/                    # Dados processados
│
├── notebooks/                        # Notebooks Jupyter (análises)
│
├── src/                              # Código-fonte principal
│   ├── config.py                     # Gerenciador de configurações
│   ├── data/
│   │   ├── data_loader.py            # Download de dados (yfinance)
│   │   ├── preprocessor.py           # Preprocessamento anti-leakage
│   │   └── feature_engineering.py    # Criação de features
│   ├── models/
│   │   ├── lstm_model.py             # Arquitetura LSTM
│   │   ├── trainer.py                # Treinamento com callbacks
│   │   └── predictor.py              # Predições
│   ├── evaluation/
│   │   ├── metrics.py                # MAE, RMSE, MAPE, R², etc
│   │   └── visualizations.py         # Gráficos e relatórios
│   ├── validation/
│   │   └── anti_leakage_tests.py     # Testes anti-leakage
│   └── api/
│       ├── main.py                   # FastAPI application
│       └── schemas.py                # Pydantic models
│
├── scripts/
│   ├── train_model.py                # Script de treinamento
│   ├── evaluate_model.py             # Script de avaliação
│   └── run_api.py                    # Rodar API
│
├── models/                           # Modelos treinados salvos
│   ├── lstm_model.h5
│   ├── scaler.pkl
│   └── model_info.json
│
├── outputs/                          # Resultados e figuras
│   └── figures/
│
├── tests/                            # Testes unitários
│
├── docker/
│   └── Dockerfile
│
└── docs/                             # Documentação técnica
    ├── architecture.md
    ├── anti_leakage_guide.md
    └── deployment_guide.md
```

---

## 🚀 Quick Start

### 1. Pré-requisitos

- Python 3.10+
- pip
- (Opcional) Docker e Docker Compose

### 2. Instalação

```bash
# Clonar repositório
git clone <repo-url>
cd lstm-stock-prediction

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Treinar o Modelo

```bash
python scripts/train_model.py
```

Este comando irá:
- ✅ Baixar dados históricos do S&P 500 (2019-2024)
- ✅ Baixar dados do VIX (índice de volatilidade)
- ✅ Criar features adicionais
- ✅ Realizar split temporal (70/15/15)
- ✅ Treinar modelo LSTM
- ✅ Avaliar com métricas completas
- ✅ Executar testes anti-leakage
- ✅ Gerar visualizações
- ✅ Salvar modelo e scaler

**Tempo estimado:** 10-30 minutos (dependendo do hardware)

### 4. Rodar a API

```bash
python scripts/run_api.py
```

A API estará disponível em:
- **Documentação interativa:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

### 5. Fazer Predições

**Via cURL:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "date": "2024-11-01",
        "open": 5730.50,
        "high": 5745.25,
        "low": 5720.00,
        "close": 5738.80,
        "volume": 1500000000
      }
      // ... mais 59 dias
    ]
  }'
```

**Via Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"data": [...]}  # 60 dias de dados
)

print(response.json())
# {"prediction": 5745.32, "model_version": "1.0.0", ...}
```

---

## 📊 Resultados

### Métricas de Performance (Conjunto de Teste)

| Métrica              | Valor    | Descrição                                    |
|----------------------|----------|----------------------------------------------|
| **MAE**              | XX.XX    | Erro Absoluto Médio                          |
| **RMSE**             | XX.XX    | Raiz do Erro Quadrático Médio                |
| **MAPE**             | XX.XX%   | Erro Percentual Absoluto Médio               |
| **R² Score**         | 0.XX     | Coeficiente de Determinação                  |
| **Direction Accuracy**| XX.XX%   | % de acertos na direção do movimento         |

### Comparação com Baselines

| Modelo            | RMSE    | MAPE   | R²     | Melhoria vs Naive |
|-------------------|---------|--------|--------|-------------------|
| **LSTM (Nosso)**  | XX.XX   | XX.XX% | 0.XX   | +XX.X%            |
| Naive Forecast    | XX.XX   | XX.XX% | 0.XX   | -                 |
| Moving Avg (5)    | XX.XX   | XX.XX% | 0.XX   | -X.X%             |
| Moving Avg (20)   | XX.XX   | XX.XX% | 0.XX   | -X.X%             |

> **Nota:** Os valores acima serão preenchidos após o primeiro treinamento.

### Visualizações

As visualizações geradas incluem:
- 📈 Histórico de treinamento (loss e métricas)
- 📊 Predições vs Valores Reais
- 🎯 Scatter Plot de predições
- 📉 Análise de resíduos
- 📊 Distribuição de erros
- 📈 Comparação com baselines

Todas as figuras são salvas em `outputs/figures/`.

---

## 🔒 Proteção Anti-Data-Leakage

Este projeto implementa **proteções rigorosas** contra data leakage:

### 1. Split Temporal PRIMEIRO
- ✅ Split acontece **ANTES** de qualquer processamento
- ✅ Treino, validação e teste são **separados temporalmente**
- ✅ **NUNCA** usamos shuffle ou KFold

### 2. Normalização Correta
- ✅ `Scaler.fit()` **APENAS** no conjunto de treino
- ✅ `Scaler.transform()` em validação e teste
- ✅ **NUNCA** usamos `fit_transform()` em val/test

### 3. Features Sem Informação Futura
- ✅ Todas as features usam **apenas dados históricos**
- ✅ Médias móveis calculadas corretamente com `min_periods`
- ✅ Nenhuma feature "espia" o futuro

### 4. Sequências Temporais Corretas
- ✅ Sequências respeitam **ordem temporal**
- ✅ Sequência de 60 dias → prediz dia 61
- ✅ Não misturamos dados de períodos diferentes

### 5. Testes Automatizados
- ✅ Verificação de ordem temporal
- ✅ Verificação de sobreposição entre conjuntos
- ✅ Detecção de performance suspeita
- ✅ Comparação com baselines
- ✅ Validação do scaler

**Veja mais detalhes em:** [`docs/anti_leakage_guide.md`](docs/anti_leakage_guide.md)

---

## 🐳 Docker

### Build e Run com Docker

```bash
# Build da imagem
docker build -f docker/Dockerfile -t lstm-stock-api .

# Run do container
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  lstm-stock-api
```

### Docker Compose

```bash
# Subir serviços
docker-compose up -d

# Ver logs
docker-compose logs -f api

# Parar serviços
docker-compose down
```

---

## 🧪 Testes

### Executar Testes Unitários

```bash
# Todos os testes
pytest tests/ -v

# Com cobertura
pytest tests/ --cov=src --cov-report=html

# Apenas testes anti-leakage
pytest tests/test_anti_leakage.py -v
```

---

## 📚 API Documentation

### Endpoints Disponíveis

#### `GET /health`
Health check da API.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-11-16T10:30:00Z",
  "version": "1.0.0"
}
```

#### `GET /model/info`
Informações sobre o modelo treinado.

**Response:**
```json
{
  "model_version": "1.0.0",
  "model_type": "LSTM",
  "training_date": "2024-11-16",
  "metrics": {
    "MAE": 12.34,
    "RMSE": 18.56,
    "MAPE": 2.15,
    "R2": 0.85,
    "Direction_Accuracy": 62.5
  },
  "sequence_length": 60,
  "n_features": 10
}
```

#### `POST /predict`
Fazer uma predição.

**Request:**
```json
{
  "data": [
    {
      "date": "2024-11-01",
      "open": 5730.50,
      "high": 5745.25,
      "low": 5720.00,
      "close": 5738.80,
      "volume": 1500000000
    }
    // ... mais 59 dias (total de 60)
  ]
}
```

**Response:**
```json
{
  "prediction": 5745.32,
  "confidence_lower": 5720.15,
  "confidence_upper": 5770.50,
  "model_version": "1.0.0",
  "timestamp": "2024-11-16T10:30:00Z"
}
```

#### `POST /predict/batch`
Fazer predições em batch.

**Documentação completa:** http://localhost:8000/docs

---

## 🛠️ Configuração

Todas as configurações estão centralizadas em `config.yaml`:

```yaml
data:
  symbol: "^GSPC"           # S&P 500
  start_date: "2019-01-01"
  end_date: "2024-11-01"
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

model:
  sequence_length: 60
  lstm_layers:
    - units: 128
      return_sequences: true
      dropout: 0.2
    # ...

training:
  batch_size: 32
  epochs: 100
  early_stopping:
    patience: 15
  # ...
```

---

## 📖 Documentação Adicional

- 🏗️ **Arquitetura:** [`docs/architecture.md`](docs/architecture.md)
- 🔒 **Guia Anti-Leakage:** [`docs/anti_leakage_guide.md`](docs/anti_leakage_guide.md)
- 🚀 **Guia de Deploy:** [`docs/deployment_guide.md`](docs/deployment_guide.md)

---

## 🤝 Contribuindo

Este é um projeto acadêmico, mas contribuições são bem-vindas!

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👨‍💻 Autor

**Desenvolvido como parte do Tech Challenge - Fase 04**
- **Instituição:** POSTECH FIAP
- **Curso:** Pós-Graduação em Machine Learning Engineering
- **Ano:** 2024

---

## 🙏 Agradecimentos

- POSTECH FIAP pelo excelente programa de ensino
- Comunidade open-source pelas ferramentas incríveis
- Yahoo Finance pela disponibilização dos dados

---

## 📞 Suporte

Se encontrar algum problema ou tiver dúvidas:

1. Verifique a [documentação](docs/)
2. Abra uma [issue](https://github.com/seu-usuario/lstm-stock-prediction/issues)
3. Consulte os [exemplos de uso](notebooks/)

---

## 🔜 Roadmap

- [ ] Adicionar mais features (sentimento, indicadores técnicos)
- [ ] Implementar ensemble de modelos
- [ ] Deploy em cloud (AWS/GCP/Azure)
- [ ] Dashboard interativo com Streamlit
- [ ] API de alertas em tempo real
- [ ] Suporte a múltiplos ativos

---

**⭐ Se este projeto foi útil, considere dar uma estrela!**

---

<div align="center">
  <strong>Tech Challenge Fase 04 - POSTECH FIAP 2024</strong>
</div>
