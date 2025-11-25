# Análise dos Resultados do Treinamento

**Data:** 2025-11-25  
**Status:** 🔴 Problema Crítico Identificado

---

## 📊 Resultados Atuais

### Métricas no Conjunto de TESTE

| Métrica | Valor | Status |
|---------|-------|--------|
| MAE | 719.31 | 🔴 Crítico (~15% do valor do S&P 500) |
| RMSE | 733.31 | 🔴 Crítico |
| MAPE | 13.12% | 🔴 Inaceitável |
| **R²** | **-8.8126** | 🔴 **CRÍTICO: Pior que média!** |
| Direction Accuracy | 43.87% | 🔴 Pior que aleatório (50%) |

### Comparação com Baselines

| Modelo | MAE | RMSE | MAPE | R² | Dir_Acc |
|--------|-----|------|------|----|---------| 
| **LSTM** | **719.31** | **733.31** | **13.12%** | **-8.81** | **43.87%** |
| Naive | 32.84 | 44.21 | 0.61% | 0.96 | 0.00% |
| MA_5 | 51.94 | 66.69 | 0.96% | 0.92 | 46.67% |
| MA_20 | 100.49 | 116.32 | 1.84% | 0.73 | 48.89% |

**⚠️ O modelo LSTM está MUITO pior que todos os baselines!**

---

## ✅ Correções Aplicadas

1. **✅ target_idx hardcoded** - Corrigido
2. **✅ Scaler load incorreto** - Corrigido  
3. **✅ Validações ausentes** - Adicionadas
4. **✅ Inverse transform** - Verificado e funcionando corretamente

**Diagnóstico confirmou:**
- `target_idx = 0` está correto (Close na primeira posição)
- `inverse_transform` funciona perfeitamente (erro < 1e-12)
- Scaler carrega metadados corretamente

---

## 🔍 Possíveis Causas do Problema

### 1. Modelo Não Está Aprendendo Corretamente

**Sintomas:**
- Loss diminui durante treinamento (0.0195 final)
- Mas métricas finais são terríveis
- R² negativo indica predições piores que média

**Possíveis causas:**
- Modelo muito simples para a complexidade dos dados
- Overfitting (loss baixo mas generalização ruim)
- Features não estão capturando padrões relevantes
- Problema na arquitetura do modelo

### 2. Problema de Escala nas Predições

**Sintomas:**
- MAE muito alto (719 vs ~30-50 esperado)
- Predições podem estar em escala errada

**Verificação necessária:**
- Verificar se predições normalizadas estão no range [0, 1]
- Verificar se há valores extremos nas predições

### 3. Features Não-Estacionárias

**Sintomas:**
- 4 features não-estacionárias (Close, High, Low, Open)
- Modelo pode estar confuso com tendências

**Recomendação:**
- Considerar usar apenas features estacionárias
- Ou aplicar diferenciação nas features não-estacionárias

### 4. Problema na Criação de Sequências

**Sintomas:**
- Sequências podem não estar alinhadas corretamente
- Target pode estar deslocado

**Verificação necessária:**
- Verificar alinhamento de X e y nas sequências
- Verificar se target está correto

---

## 🎯 Próximos Passos Recomendados

### 1. Análise Imediata

```bash
# Verificar predições do modelo diretamente
python scripts/verificar_predicoes.py  # (criar este script)
```

**Verificar:**
- Range das predições normalizadas
- Comparação visual predições vs valores reais
- Distribuição dos erros

### 2. Melhorias no Modelo

**Opções:**
1. **Aumentar complexidade:**
   - Mais unidades LSTM (64 → 128)
   - Mais camadas (2 → 3)
   - Mais neurônios na Dense (8 → 16)

2. **Ajustar hiperparâmetros:**
   - Learning rate (atual: 0.001)
   - Batch size (atual: 32)
   - Sequence length (atual: 60)

3. **Regularização:**
   - Aumentar dropout (atual: 0.2)
   - Adicionar L2 regularization

### 3. Melhorias nas Features

**Opções:**
1. **Remover features não-estacionárias:**
   - Usar apenas Return ao invés de Close/High/Low/Open
   - Manter apenas features estacionárias

2. **Adicionar mais features:**
   - Features técnicas (RSI, MACD, etc.)
   - Features de mercado (sentimento, etc.)

3. **Feature engineering mais sofisticado:**
   - Transformações logarítmicas
   - Diferenciação de features não-estacionárias

### 4. Arquitetura Alternativa

**Considerar:**
- GRU ao invés de LSTM
- Attention mechanisms
- Ensemble de modelos
- Transformer para séries temporais

---

## 📝 Notas Técnicas

### Por que R² negativo?

R² negativo ocorre quando:
```
SS_res > SS_tot
```

Onde:
- `SS_res` = Soma dos quadrados dos resíduos (erros)
- `SS_tot` = Soma dos quadrados totais (variação dos dados)

Isso significa que o modelo está fazendo predições **piores** que simplesmente usar a média dos valores reais.

### Interpretação

Com R² = -8.81:
- O modelo está fazendo predições que são ~9x piores que usar a média
- Isso é extremamente raro e indica problema grave

### Possível Causa Raiz

O modelo pode estar:
1. Prevendo valores completamente fora da escala
2. Não aprendendo padrões temporais
3. Apenas memorizando ruído dos dados de treino

---

## 🔧 Scripts de Diagnóstico Criados

1. `scripts/diagnostico_inverse_transform.py` ✅
   - Verifica se inverse_transform funciona
   - **Resultado:** Funcionando perfeitamente

2. `scripts/diagnostico_predicoes.py` ⚠️
   - Verifica predições do modelo
   - **Status:** Precisa ser executado (problemas de encoding)

---

## 💡 Recomendações Imediatas

1. **Criar visualização das predições:**
   - Plot predições vs valores reais
   - Identificar padrões de erro

2. **Verificar escala das predições:**
   - Verificar se predições normalizadas estão em [0, 1]
   - Verificar se há outliers

3. **Testar modelo mais simples primeiro:**
   - Linear regression como baseline
   - Verificar se problema é específico do LSTM

4. **Considerar re-treinar com:**
   - Apenas features estacionárias
   - Arquitetura mais simples/complexa
   - Diferentes hiperparâmetros

---

## 📊 Recursos Disponíveis

**Máquina:** AMD 7900X com 32GB RAM
- Pode fazer treinamentos mais pesados
- Pode testar arquiteturas mais complexas
- Pode fazer hyperparameter tuning

---

## ✅ Conclusão

As correções de bugs foram aplicadas com sucesso, mas o modelo ainda apresenta performance muito ruim. O problema não está no `inverse_transform` ou no `target_idx`, mas provavelmente na capacidade do modelo de aprender padrões ou na qualidade/representatividade das features.

**Próximo passo crítico:** Analisar as predições diretamente para identificar o problema específico.

