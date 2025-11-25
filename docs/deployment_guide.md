# 🚀 Guia de Deploy

Este guia mostra como fazer deploy da API LSTM Stock Prediction em diferentes ambientes.

---

## 📋 Pré-requisitos

Antes de fazer deploy, certifique-se de que:

- ✅ Modelo foi treinado (`models/lstm_model.h5` existe)
- ✅ Scaler foi salvo (`models/scaler.pkl` existe)
- ✅ API funciona localmente
- ✅ Testes passaram
- ✅ Docker instalado (para deploy com containers)

---

## 🏠 Opção 1: Deploy Local (Desenvolvimento)

### 1.1 Executar Diretamente

```bash
# Ativar ambiente virtual
source venv/bin/activate

# Executar API
python scripts/run_api.py

# Com reload (desenvolvimento)
python scripts/run_api.py --dev

# Especificar porta
python scripts/run_api.py --port 5000
```

### 1.2 Usar uvicorn Diretamente

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 1.3 Testar API

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info

# Fazer predição (com dados de exemplo)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @examples/prediction_request.json
```

---

## 🐳 Opção 2: Deploy com Docker

### 2.1 Build da Imagem

```bash
# Build básico
docker build -f docker/Dockerfile -t lstm-stock-api:latest .

# Build com tag específica
docker build -f docker/Dockerfile -t lstm-stock-api:v1.0.0 .

# Verificar imagem criada
docker images | grep lstm-stock-api
```

### 2.2 Run do Container

```bash
# Run básico
docker run -d \
  --name lstm-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  lstm-stock-api:latest

# Run com logs visíveis
docker run \
  --name lstm-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  lstm-stock-api:latest

# Verificar se está rodando
docker ps | grep lstm-api

# Ver logs
docker logs lstm-api -f

# Parar container
docker stop lstm-api

# Remover container
docker rm lstm-api
```

### 2.3 Docker Compose

```bash
# Subir serviços
docker-compose up -d

# Ver logs
docker-compose logs -f api

# Verificar status
docker-compose ps

# Parar serviços
docker-compose down

# Rebuild e restart
docker-compose up -d --build
```

---

## ☁️ Opção 3: Deploy em Cloud

### 3.1 Render (Recomendado - Free Tier)

**Render** é uma plataforma de deploy moderna com free tier generoso.

#### Passo a Passo:

1. **Criar conta** em [render.com](https://render.com)

2. **Criar novo Web Service**
   - Conectar repositório GitHub
   - Escolher "Docker"
   - Configurar:
     - **Name:** `lstm-stock-api`
     - **Region:** Oregon (US West) ou Frankfurt (EU)
     - **Branch:** `main`
     - **Dockerfile Path:** `docker/Dockerfile`

3. **Variáveis de Ambiente**
   ```
   PYTHONUNBUFFERED=1
   TF_CPP_MIN_LOG_LEVEL=2
   ```

4. **Persistent Disk (opcional)**
   - Criar disk para `/app/models`
   - Size: 1GB
   - Mount path: `/app/models`

5. **Deploy**
   - Click em "Create Web Service"
   - Aguardar build (5-10 min)

6. **Testar**
   ```bash
   curl https://lstm-stock-api.onrender.com/health
   ```

#### render.yaml (Blueprint)

Crie na raiz do projeto:

```yaml
services:
  - type: web
    name: lstm-stock-api
    env: docker
    dockerfilePath: ./docker/Dockerfile
    region: oregon
    plan: free
    envVars:
      - key: PYTHONUNBUFFERED
        value: 1
      - key: TF_CPP_MIN_LOG_LEVEL
        value: 2
    healthCheckPath: /health
    disk:
      name: models
      mountPath: /app/models
      sizeGB: 1
```

Depois:
```bash
render deploy
```

### 3.2 Railway (Alternativa)

**Railway** é similar ao Render com deploy fácil.

#### Passo a Passo:

1. **Criar conta** em [railway.app](https://railway.app)

2. **New Project** → **Deploy from GitHub repo**

3. **Configurar**
   - Railway detecta Dockerfile automaticamente
   - Adicionar variáveis de ambiente
   - Configurar domínio

4. **Deploy**
   - Push para GitHub → deploy automático

### 3.3 Heroku (Clássico)

#### Pré-requisitos
```bash
# Instalar Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh

# Login
heroku login
```

#### Deploy

```bash
# Criar app
heroku create lstm-stock-api

# Configurar container stack
heroku stack:set container -a lstm-stock-api

# Deploy
git push heroku main

# Ver logs
heroku logs --tail -a lstm-stock-api

# Abrir app
heroku open -a lstm-stock-api
```

#### heroku.yml

```yaml
build:
  docker:
    web: docker/Dockerfile
run:
  web: python scripts/run_api.py
```

### 3.4 Google Cloud Run

#### Pré-requisitos
```bash
# Instalar gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Autenticar
gcloud auth login

# Configurar projeto
gcloud config set project YOUR_PROJECT_ID
```

#### Deploy

```bash
# Build e push para Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/lstm-stock-api

# Deploy para Cloud Run
gcloud run deploy lstm-stock-api \
  --image gcr.io/YOUR_PROJECT_ID/lstm-stock-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2

# Ver URL
gcloud run services describe lstm-stock-api --platform managed --region us-central1
```

### 3.5 AWS (ECS/Fargate)

#### Pré-requisitos
```bash
# Instalar AWS CLI
pip install awscli

# Configurar
aws configure
```

#### Deploy com ECS

```bash
# 1. Criar ECR repository
aws ecr create-repository --repository-name lstm-stock-api

# 2. Login no ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# 3. Build e tag
docker build -f docker/Dockerfile -t lstm-stock-api .
docker tag lstm-stock-api:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/lstm-stock-api:latest

# 4. Push
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/lstm-stock-api:latest

# 5. Criar task definition e service via console ou CLI
```

---

## 🔧 Configurações de Produção

### 4.1 Variáveis de Ambiente

Crie `.env` (NÃO versionar!):

```bash
# Environment
ENV=production

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model
MODEL_PATH=/app/models/lstm_model.h5
SCALER_PATH=/app/models/scaler.pkl

# Logging
LOG_LEVEL=INFO

# Security (opcional)
API_KEY=your-secret-api-key

# Monitoring (opcional)
SENTRY_DSN=your-sentry-dsn
```

### 4.2 Gunicorn (Workers)

Para múltiplos workers:

```bash
# Install gunicorn
pip install gunicorn

# Run with workers
gunicorn src.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

Dockerfile para produção:

```dockerfile
# Última linha
CMD ["gunicorn", "src.api.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

### 4.3 NGINX (Reverse Proxy)

`nginx.conf`:

```nginx
upstream api {
    server localhost:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4.4 HTTPS (Let's Encrypt)

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obter certificado
sudo certbot --nginx -d your-domain.com

# Auto-renew
sudo certbot renew --dry-run
```

---

## 📊 Monitoramento

### 5.1 Logs

```python
# src/api/main.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(request: PredictionRequest):
    logger.info(f"Prediction request received: {len(request.data)} data points")
    # ...
    logger.info(f"Prediction result: {prediction}")
```

### 5.2 Sentry (Error Tracking)

```bash
pip install sentry-sdk[fastapi]
```

```python
# src/api/main.py
import sentry_sdk

sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    traces_sample_rate=1.0,
)
```

### 5.3 Prometheus (Metrics)

```bash
pip install prometheus-fastapi-instrumentator
```

```python
# src/api/main.py
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)
```

Acesse métricas em: `http://localhost:8000/metrics`

---

## 🔐 Segurança

### 6.1 Rate Limiting

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, data: PredictionRequest):
    # ...
```

### 6.2 API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    # ...
```

### 6.3 CORS

Já configurado em `src/api/main.py`. Para produção, especifique origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],  # Específico
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## 🧪 CI/CD

### GitHub Actions

`.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Build Docker image
        run: docker build -f docker/Dockerfile -t lstm-stock-api .

      - name: Run tests
        run: docker run lstm-stock-api pytest tests/

      - name: Deploy to Render
        run: |
          curl -X POST "https://api.render.com/deploy/srv-xxxxx?key=${{ secrets.RENDER_DEPLOY_KEY }}"
```

---

## 📋 Checklist de Deploy

Antes de fazer deploy para produção:

- [ ] Modelo treinado e testado
- [ ] Testes anti-leakage passaram
- [ ] API funciona localmente
- [ ] Docker build funciona
- [ ] Variáveis de ambiente configuradas
- [ ] Secrets não estão no código
- [ ] .gitignore configurado
- [ ] Logging implementado
- [ ] Error handling adequado
- [ ] Rate limiting configurado
- [ ] HTTPS habilitado
- [ ] Monitoring configurado
- [ ] Backup do modelo
- [ ] Documentação atualizada
- [ ] Health check funciona
- [ ] Load testing realizado

---

## 🆘 Troubleshooting

### Problema: Container não inicia

```bash
# Ver logs
docker logs lstm-api

# Entrar no container
docker exec -it lstm-api bash

# Verificar arquivos
ls -la /app/models/
```

### Problema: Modelo não carrega

```bash
# Verificar se modelo existe
docker exec lstm-api ls -la /app/models/

# Verificar permissões
docker exec lstm-api ls -l /app/models/lstm_model.h5
```

### Problema: Out of Memory

```bash
# Aumentar memória do container
docker run -m 4g lstm-stock-api

# Docker Compose
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Problema: API lenta

- Aumentar workers (Gunicorn)
- Usar caching (Redis)
- Otimizar modelo (quantização)
- Escalar horizontalmente

---

**Última atualização:** 2024-11-16
