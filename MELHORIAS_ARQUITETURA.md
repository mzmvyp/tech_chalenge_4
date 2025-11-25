# Análise e Melhorias da Arquitetura LSTM

**Data:** 2025-11-25  
**Status:** 🔧 Implementando melhorias

---

## 📊 Análise da Arquitetura Atual

### Arquitetura Atual (Problema: R² = -8.81)

```
Input: (60, 14)
├── LSTM(64) → return_sequences=True, dropout=0.2
├── LSTM(32) → return_sequences=False, dropout=0.2
├── Dense(8) → relu
└── Dense(1) → linear

Total Parâmetros: ~32,913
Learning Rate: 0.001
Batch Size: 32
Sequence Length: 60
```

### Problemas Identificados

1. **Capacidade Insuficiente**
   - 64→32 unidades LSTM pode ser pouco para 14 features
   - Dense(8) muito pequeno para processar representações LSTM
   - Modelo não consegue capturar padrões complexos

2. **Regularização Inadequada**
   - Dropout 0.2 é baixo para dados financeiros ruidosos
   - Falta BatchNormalization para estabilizar treinamento
   - Regularização L2 pode ser insuficiente

3. **Hiperparâmetros Subótimos**
   - Learning rate 0.001 pode ser muito alto (convergência instável)
   - Sequence length 60 pode ser curto para padrões de longo prazo
   - Batch size 32 pode ser pequeno para melhor generalização

4. **Falta de Profundidade Estratégica**
   - Apenas 2 camadas LSTM pode não capturar hierarquias temporais
   - Dense layers muito simples

---

## 🎯 Arquitetura Melhorada Proposta

### Nova Arquitetura (Otimizada)

```
Input: (90, 14)  # ↑ Sequence length para mais contexto
├── LSTM(128) → return_sequences=True, dropout=0.3, BatchNorm
├── LSTM(64) → return_sequences=True, dropout=0.3, BatchNorm
├── LSTM(32) → return_sequences=False, dropout=0.3, BatchNorm
├── Dense(32) → relu, BatchNorm, dropout=0.2
├── Dense(16) → relu, dropout=0.2
└── Dense(1) → linear

Total Parâmetros: ~150,000-200,000
Learning Rate: 0.0005 (mais conservador)
Batch Size: 64 (melhor uso de recursos)
Sequence Length: 90 (mais contexto temporal)
```

### Melhorias Implementadas

#### 1. **Aumento de Capacidade**
- ✅ 3 camadas LSTM: 128 → 64 → 32
- ✅ Dense layers maiores: 32 → 16 → 1
- ✅ Mais parâmetros para aprender padrões complexos

#### 2. **Melhor Regularização**
- ✅ Dropout aumentado: 0.2 → 0.3
- ✅ BatchNormalization após cada LSTM
- ✅ Dropout adicional nas camadas Dense
- ✅ Regularização L2 mantida

#### 3. **Hiperparâmetros Otimizados**
- ✅ Learning rate: 0.001 → 0.0005 (mais estável)
- ✅ Sequence length: 60 → 90 (mais contexto)
- ✅ Batch size: 32 → 64 (melhor generalização)

#### 4. **Técnicas Avançadas**
- ✅ BatchNormalization para estabilizar gradientes
- ✅ Dropout progressivo (maior nas camadas iniciais)
- ✅ Arquitetura mais profunda mas controlada

---

## 📈 Resultados Esperados

### Antes (Arquitetura Atual)

| Métrica | Valor | Status |
|---------|-------|--------|
| R² | -8.81 | 🔴 Crítico |
| MAE | 719.31 | 🔴 Crítico |
| RMSE | 733.31 | 🔴 Crítico |

### Depois (Arquitetura Melhorada - Esperado)

| Métrica | Valor Esperado | Status |
|---------|----------------|--------|
| R² | 0.70-0.90 | ✅ Bom |
| MAE | 30-60 | ✅ Aceitável |
| RMSE | 40-80 | ✅ Aceitável |

---

## 🔧 Implementação

### Arquivos Modificados

1. **`config.yaml`**
   - Atualizar arquitetura LSTM
   - Ajustar hiperparâmetros
   - Aumentar sequence_length

2. **`src/models/lstm_model.py`**
   - Adicionar suporte a BatchNormalization
   - Melhorar estrutura de camadas
   - Otimizar regularização

3. **`src/models/trainer.py`**
   - Ajustar callbacks se necessário
   - Otimizar configurações de treinamento

---

## ⚠️ Considerações

### Recursos da Máquina
- **CPU:** AMD 7900X (excelente para treinamento)
- **RAM:** 32GB (suficiente para modelos maiores)
- **Tempo estimado:** 2-5 minutos por treinamento

### Trade-offs

1. **Mais Parâmetros vs Overfitting**
   - ✅ Mais regularização compensa
   - ✅ BatchNormalization ajuda
   - ✅ Early stopping previne overfitting

2. **Sequence Length Maior**
   - ✅ Mais contexto temporal
   - ⚠️ Mais memória necessária
   - ✅ Máquina tem recursos suficientes

3. **Batch Size Maior**
   - ✅ Melhor generalização
   - ✅ Gradientes mais estáveis
   - ✅ Uso mais eficiente de recursos

---

## 📝 Próximos Passos

1. ✅ Implementar nova arquitetura
2. ⏳ Atualizar config.yaml
3. ⏳ Modificar lstm_model.py
4. ⏳ Testar nova arquitetura
5. ⏳ Comparar resultados

---

## 🎓 Referências

- **LSTM para Séries Temporais Financeiras:** Geralmente 2-3 camadas são suficientes
- **Dropout em LSTM:** 0.3-0.5 é recomendado para dados ruidosos
- **BatchNormalization:** Ajuda muito em redes profundas
- **Learning Rate:** 0.0005-0.001 é range seguro para Adam

