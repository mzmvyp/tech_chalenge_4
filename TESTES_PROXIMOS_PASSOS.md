# 🧪 Próximos Passos para Testar o Sistema

## ✅ Status Atual

- ✅ Modelo treinado (`lstm_model.h5` e `lstm_model_error_learned.h5`)
- ✅ Error learning implementado
- ✅ Retreinamento periódico implementado
- ✅ API FastAPI criada
- ✅ Scripts de teste criados

## 🎯 Próximos Passos (Ordem de Prioridade)

### 1. TESTAR A API (PRIORIDADE ALTA) ⭐

A API é o ponto final do sistema e precisa estar funcionando para produção.

#### Passo 1.1: Iniciar a API

```bash
# Terminal 1: Iniciar API
python scripts/run_api.py --dev
```

A API estará disponível em: `http://localhost:8000`
- Documentação: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

#### Passo 1.2: Testar Endpoints

```bash
# Terminal 2: Executar testes
python scripts/test_api.py
```

**O que será testado:**
- ✅ Health check (`/health`)
- ✅ Informações do modelo (`/model/info`)
- ✅ Predição única (`/predict`)
- ✅ Monitoramento (`/monitoring/stats`)

#### Passo 1.3: Testar Manualmente (Opcional)

Acesse `http://localhost:8000/docs` no navegador e teste os endpoints via interface Swagger.

---

### 2. TESTAR RETREINAMENTO COMPLETO COM BACKTEST

Verificar se o fluxo completo de retreinamento + backtest funciona.

```bash
# Testar retreinamento incremental com backtest automático
python scripts/periodic_retrain.py --mode incremental --run-backtest --backtest-tests 100
```

**O que será testado:**
- ✅ Download de novos dados
- ✅ Mesclagem com dados históricos
- ✅ Fine-tuning do modelo
- ✅ Preservação do aprendizado de erros
- ✅ Backtest automático
- ✅ Salvamento de resultados

---

### 3. TESTAR BACKTEST COM ERROR LEARNING

Verificar se o sistema de error learning está funcionando corretamente.

```bash
# Backtest simples
python scripts/backtest_with_error_learning.py --n-tests 100

# Backtest com ensemble e aprendizado imediato
python scripts/backtest_with_error_learning.py --use-ensemble --immediate-learn --n-tests 200
```

**O que será testado:**
- ✅ Carregamento do modelo aprendido
- ✅ Predições com ensemble
- ✅ Adaptive threshold
- ✅ Error learning (imediato e batch)
- ✅ Melhoria de accuracy ao longo do tempo

---

### 4. VERIFICAR FLUXO COMPLETO DE PRODUÇÃO

Simular o fluxo completo que acontecerá em produção.

#### Passo 4.1: Verificar Modelos Disponíveis

```bash
# Verificar quais modelos existem
ls -lh models/lstm_model*.h5

# Verificar data de modificação (modelo mais recente será usado)
```

#### Passo 4.2: Simular Dia de Produção

```bash
# 1. API fazendo predições (simular)
python scripts/run_api.py

# 2. Em outro terminal, fazer algumas predições via API
python scripts/test_api.py

# 3. Verificar se error learning está funcionando (se habilitado)
# (Isso acontece automaticamente durante predições se configurado)
```

---

## 🔍 Checklist de Validação

### API
- [ ] API inicia sem erros
- [ ] Health check retorna OK
- [ ] Model info retorna informações corretas
- [ ] Predição funciona com dados válidos
- [ ] Predição retorna erro apropriado com dados inválidos
- [ ] Documentação Swagger está acessível

### Retreinamento
- [ ] Retreinamento incremental funciona
- [ ] Modelo aprendido é preservado
- [ ] Backtest automático executa após retreinamento
- [ ] Resultados são salvos corretamente

### Error Learning
- [ ] Modelo aprendido é carregado corretamente
- [ ] Erros são detectados e aprendidos
- [ ] Accuracy melhora ao longo do tempo
- [ ] Ensemble funciona corretamente

### Produção
- [ ] Modelo mais recente é detectado automaticamente
- [ ] API usa modelo correto
- [ ] Logs são gerados corretamente
- [ ] Histórico de backtests é mantido

---

## 🚨 Problemas Comuns e Soluções

### API não inicia
- Verificar se modelo existe: `ls models/lstm_model.h5`
- Verificar se scaler existe: `ls models/scaler.pkl`
- Verificar logs de erro

### Predição retorna erro
- Verificar se dados têm formato correto (OHLCV)
- Verificar se tem dados suficientes (mínimo 60+ dias)
- Verificar se features estão corretas

### Retreinamento falha
- Verificar conexão com internet (para download de dados)
- Verificar se dados históricos existem
- Verificar espaço em disco

---

## 📊 Resultados Esperados

### API
- ✅ Health check: Status "healthy"
- ✅ Model info: Modelo carregado, features corretas
- ✅ Predição: Preço predito com intervalo de confiança

### Retreinamento
- ✅ Modelo atualizado com novos dados
- ✅ Aprendizado de erros preservado
- ✅ Backtest mostra Direction Accuracy

### Error Learning
- ✅ Accuracy melhora ao longo do tempo
- ✅ Erros são aprendidos e incorporados
- ✅ Modelo salvo com aprendizado acumulado

---

## 🎯 Ordem Recomendada de Testes

1. **API** (mais importante - ponto final do sistema)
2. **Backtest com Error Learning** (validar aprendizado)
3. **Retreinamento Completo** (validar manutenção)
4. **Fluxo Completo** (validar integração)

---

**Comece testando a API primeiro!** É o componente mais crítico para produção.

