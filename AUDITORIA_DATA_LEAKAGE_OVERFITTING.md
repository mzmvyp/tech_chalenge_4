# Auditoria de Data Leakage e Overfitting - Tech Challenge Fase 04
# Projeto: LSTM Stock Prediction

**Data da Auditoria:** 2025-11-16
**Auditor:** Cientista de Dados Especializado
**Objetivo:** Avaliar pipeline de dados para identificar data leakage, overfitting e outros problemas que causam alta performance no treino mas falha nos testes

---

## Sumário Executivo

### Resultado Geral: ✅ BOM (com ressalvas)

O projeto demonstra **excelentes práticas** em várias áreas críticas, especialmente no pipeline de preprocessamento e split temporal. No entanto, foram identificados **5 problemas de severidade ALTA** e **7 de severidade MÉDIA** que podem comprometer a performance em produção.

### Principais Achados:

✅ **Pontos Fortes:**
- Split temporal correto (sem shuffle)
- Normalização implementada corretamente
- Features temporais sem vazamento de futuro
- Módulo dedicado de testes anti-leakage
- Validações automáticas implementadas

⚠️ **Problemas Críticos:**
1. Forward fill no VIX pode causar leakage
2. Multicolinearidade excessiva nas features
3. Arquitetura muito complexa (risco de overfitting)
4. Ausência de walk-forward validation
5. Falta de regularização L1/L2

---

## 1. Análise de Data Leakage

### 1.1 Split Temporal ✅ CORRETO

**Arquivo:** `src/data/preprocessor.py:87-150`

```python
def temporal_split(self, df: pd.DataFrame, verbose: bool = True):
    # Split temporal sem shuffle
    df_train = df.iloc[:train_end_idx].copy()
    df_val = df.iloc[train_end_idx:val_end_idx].copy()
    df_test = df.iloc[val_end_idx:].copy()
```

**Avaliação:** ✅ EXCELENTE
- Split temporal respeitando ordem cronológica
- Validação automática de não-sobreposição (linhas 146-148)
- Ratios: 70% treino / 15% validação / 15% teste

**Evidências:**
```python
assert df_train.index[-1] < df_val.index[0]
assert df_val.index[-1] < df_test.index[0]
```

---

### 1.2 Normalização ✅ CORRETO

**Arquivo:** `src/data/preprocessor.py:152-205`

```python
# Fit scaler APENAS no treino
train_scaled = self.fit_scaler(df_train)

# Transform (sem fit!) em validação e teste
val_scaled = self.transform(df_val)
test_scaled = self.transform(df_test)
```

**Avaliação:** ✅ EXCELENTE
- Scaler fitado APENAS no conjunto de treino
- Validação e teste usam apenas `transform()`
- Implementação seguindo best practices

**Risco de Leakage:** ZERO

---

### 1.3 Features Temporais ✅ CORRETO (com 1 ressalva)

**Arquivo:** `src/data/feature_engineering.py`

#### Médias Móveis - CORRETO ✅
```python
df[col_name] = df[price_column].rolling(
    window=window,
    min_periods=window  # Garante que não usa futuro
).mean()
```

#### Momentum - CORRETO ✅
```python
df[col_name] = (df[price_column] / df[price_column].shift(window)) - 1
```

#### Retornos - CORRETO ✅
```python
df['Return'] = df[price_column].pct_change()
```

#### ⚠️ VIX Merge - PROBLEMA IDENTIFICADO

**Arquivo:** `src/data/feature_engineering.py:218-227`

```python
# Merge por índice (data)
df = df.join(df_vix, how='left')

# Forward fill para preencher gaps
df['VIX'] = df['VIX'].fillna(method='ffill')  # ⚠️ PROBLEMA!
df['VIX'] = df['VIX'].fillna(method='bfill')  # ⚠️ PROBLEMA!
```

**Severidade:** 🔴 ALTA

**Problema:**
- `fillna(method='ffill')` propaga valores futuros para o passado quando usado ANTES do split temporal
- Se VIX tem gaps, o valor do dia seguinte pode "vazar" para o dia anterior
- `bfill()` é ainda pior - usa valores futuros explicitamente

**Cenário de Leakage:**
```
Data        | VIX Real | VIX após ffill/bfill
2024-01-01  | 20.5     | 20.5
2024-01-02  | NaN      | 22.3  ← LEAKAGE! Usando valor de 03/01
2024-01-03  | 22.3     | 22.3
```

**Impacto:**
- Médio a Alto, dependendo da frequência de gaps no VIX
- Pode inflar métricas artificialmente

**Recomendação:**
```python
# SOLUÇÃO 1: Usar apenas o valor do dia anterior (sem leakage)
df['VIX'] = df['VIX'].fillna(method='ffill').shift(1)

# SOLUÇÃO 2: Remover dias com VIX faltante
df = df.dropna(subset=['VIX'])

# SOLUÇÃO 3: Usar média dos últimos N dias conhecidos
df['VIX'] = df['VIX'].fillna(df['VIX'].rolling(5, min_periods=1).mean().shift(1))
```

---

### 1.4 Sequências LSTM ✅ CORRETO

**Arquivo:** `src/data/preprocessor.py:207-242`

```python
for i in range(self.sequence_length, len(data)):
    # Sequência de entrada: últimos 'sequence_length' dias
    X.append(data[i - self.sequence_length:i])
    # Alvo: valor de fechamento do próximo dia
    y.append(data[i, target_column_idx])
```

**Avaliação:** ✅ EXCELENTE
- Sequências respeitam ordem temporal
- Cada sequência usa apenas dados históricos
- Target é o próximo dia (prediction horizon = 1)

**Risco de Leakage:** ZERO

---

### 1.5 Pipeline de Treinamento ✅ CORRETO

**Arquivo:** `scripts/train_model.py:104-115`

```python
# Ordem correta:
# 1. Split temporal PRIMEIRO
df_train, df_val, df_test = preprocessor.temporal_split(df)

# 2. Fit scaler APENAS no treino
train_scaled = preprocessor.fit_scaler(df_train)

# 3. Transform em val e test
val_scaled = preprocessor.transform(df_val)
test_scaled = preprocessor.transform(df_test)
```

**Avaliação:** ✅ EXCELENTE
- Ordem de operações perfeita
- Pipeline implementado corretamente

---

## 2. Análise de Overfitting

### 2.1 Multicolinearidade de Features 🔴 PROBLEMA CRÍTICO

**Arquivo:** `src/data/feature_engineering.py:229-298`

**Problema Identificado:**

O código cria múltiplas features **altamente correlacionadas** derivadas da mesma fonte (Close price):

```python
# Todas baseadas em 'Close':
- Return_1d
- Momentum_5d
- Momentum_10d
- Volatility_10d  (baseado em Return)
- Volatility_30d  (baseado em Return)
- Volume_Relative_5
- Volume_Relative_20
```

**Severidade:** 🔴 ALTA

**Por que isso causa overfitting:**
1. **Multicolinearidade:** Features correlacionadas confundem o modelo
2. **Informação redundante:** Modelo aprende o mesmo padrão várias vezes
3. **Overfitting nos dados de treino:** Modelo "decora" ruído
4. **Baixa generalização:** Falha em dados novos

**Evidência Teórica:**
- Retorno e Momentum são matematicamente relacionados
- Volatility_10d e Volatility_30d medem a mesma coisa em janelas diferentes
- Volume_Relative é derivado de Volume

**Matriz de Correlação Esperada:**
```
              Return  Mom_5d  Mom_10d  Vol_10d  Vol_30d
Return        1.00    0.85    0.75     0.60     0.55
Momentum_5d   0.85    1.00    0.90     0.50     0.45
Momentum_10d  0.75    0.90    1.00     0.45     0.40
Volatility_10 0.60    0.50    0.45     1.00     0.85
Volatility_30 0.55    0.45    0.40     0.85     1.00
```

**Recomendações:**

**SOLUÇÃO 1: Seleção de Features (Feature Selection)**
```python
# Manter apenas features com correlação < 0.7
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# Ou usar PCA para reduzir dimensionalidade
from sklearn.decomposition import PCA
```

**SOLUÇÃO 2: Regularização L1 (Lasso)**
```python
# No modelo LSTM, adicionar kernel_regularizer
from tensorflow.keras.regularizers import l1_l2

LSTM(units=128, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))
```

**SOLUÇÃO 3: Reduzir Features Manualmente**
```python
# Manter apenas:
- OHLCV básico (5 features)
- VIX (1 feature)
- Return (1 feature)
- Momentum_10d (1 feature - escolher APENAS 1)
- Volatility_30d (1 feature - escolher APENAS 1)
Total: 9 features vs. 15+ atuais
```

---

### 2.2 Arquitetura do Modelo 🔴 MUITO COMPLEXA

**Arquivo:** `config.yaml:46-65`

```yaml
lstm_layers:
  - units: 128
    return_sequences: true
    dropout: 0.2
  - units: 64
    return_sequences: true
    dropout: 0.2
  - units: 32
    return_sequences: false
    dropout: 0.2

dense_layers:
  - units: 16
    activation: "relu"
  - units: 1
```

**Análise de Complexidade:**

```
Camadas LSTM:     128 → 64 → 32 unidades
Camada Dense:     16 unidades
Total Parâmetros: ~150,000 - 200,000 (estimado)

Dados de Treino:  ~1,000 - 1,500 amostras (estimado)
Ratio:            100-200 parâmetros por amostra
```

**Severidade:** 🔴 ALTA

**Problema:**
- **Overfitting garantido** com 100-200 parâmetros por amostra
- Regra de ouro: 10-30 amostras por parâmetro
- Modelo vai "decorar" os dados de treino

**Evidências:**
1. Três camadas LSTM são excessivas para predição de 1 dia
2. 128 unidades na primeira camada é muito para ~15 features
3. Dropout de 0.2 (20%) é insuficiente

**Comparação com Literatura:**

| Paper | Dataset | Features | LSTM Units | Layers |
|-------|---------|----------|------------|--------|
| Fischer & Krauss (2018) | S&P 500 | 240 | 64 | 1 |
| Nelson et al. (2017) | Stock Market | 82 | 100 | 1 |
| **Seu Projeto** | **S&P 500** | **~15** | **128+64+32** | **3** |

**Recomendação:**

**ARQUITETURA RECOMENDADA:**
```yaml
# Opção 1: Simples e Efetiva
lstm_layers:
  - units: 32
    return_sequences: false
    dropout: 0.3

dense_layers:
  - units: 1

# Opção 2: Média Complexidade
lstm_layers:
  - units: 64
    return_sequences: true
    dropout: 0.3
  - units: 32
    return_sequences: false
    dropout: 0.3

dense_layers:
  - units: 8
    activation: "relu"
  - units: 1

# NUNCA usar 3 camadas LSTM para este problema!
```

**Justificativa:**
- Dados financeiros têm muito ruído
- Modelos simples generalizam melhor
- LSTM capta padrões temporais, não precisa ser profundo

---

### 2.3 Dropout Insuficiente ⚠️ MÉDIA

**Arquivo:** `config.yaml:53-59`

```yaml
dropout: 0.2  # 20% - muito baixo!
```

**Severidade:** 🟡 MÉDIA

**Problema:**
- Dropout de 20% é considerado baixo para prevenção de overfitting
- Literatura recomenda 30-50% para LSTMs em séries financeiras

**Recomendação:**
```yaml
# Aumentar dropout para:
dropout: 0.3  # Mínimo recomendado
# ou
dropout: 0.4  # Ideal para dados financeiros com ruído
```

---

### 2.4 Falta de Regularização L1/L2 ⚠️ MÉDIA

**Arquivo:** `src/models/lstm_model.py:93-104`

**Problema Atual:**
```python
model.add(LSTM(
    units=units,
    return_sequences=return_sequences,
    # ❌ SEM kernel_regularizer!
))
```

**Severidade:** 🟡 MÉDIA

**Recomendação:**
```python
from tensorflow.keras.regularizers import l1_l2

model.add(LSTM(
    units=units,
    return_sequences=return_sequences,
    kernel_regularizer=l1_l2(l1=0.01, l2=0.01),  # ✅ Adicionar!
    recurrent_regularizer=l1_l2(l1=0.01, l2=0.01)  # ✅ Adicionar!
))
```

**Benefícios:**
- L1: Feature selection automática (zera pesos irrelevantes)
- L2: Previne pesos muito grandes
- Reduz overfitting significativamente

---

### 2.5 Early Stopping - Patience Muito Alto ⚠️ MÉDIA

**Arquivo:** `config.yaml:80-84`

```yaml
early_stopping:
  monitor: "val_loss"
  patience: 15  # ⚠️ MUITO ALTO!
  restore_best_weights: true
  min_delta: 0.0001
```

**Severidade:** 🟡 MÉDIA

**Problema:**
- Patience de 15 épocas permite que o modelo continue treinando por muito tempo após parar de melhorar
- Isso pode resultar em overfitting

**Evidência:**
```
Época 20: val_loss = 0.001000 (melhor)
Época 21: val_loss = 0.001001
Época 22: val_loss = 0.001002
...
Época 35: val_loss = 0.001500  ← Parou aqui, mas já estava overfitting desde época 20!
```

**Recomendação:**
```yaml
early_stopping:
  monitor: "val_loss"
  patience: 7  # Reduzir para 7-10
  restore_best_weights: true  # ✅ Manter!
  min_delta: 0.0001
```

---

### 2.6 Ausência de Walk-Forward Validation 🔴 CRÍTICO

**Arquivo:** Não implementado

**Severidade:** 🔴 ALTA

**Problema:**
O projeto usa um **split único** (70/15/15), que pode não ser representativo:

```
|--------TRAIN--------|--VAL--|--TEST--|
2019      2021      2023  2024   2024
```

**Problemas:**
1. **Regime shifts não detectados:** Mercado em 2024 pode ser diferente de 2019-2021
2. **Overfitting temporal:** Modelo aprende padrões específicos de 2019-2021
3. **Falta de robustez:** Um split pode ser lucky/unlucky

**Recomendação: Walk-Forward Validation**

```python
# Implementar validação rolling window
def walk_forward_validation(df, n_splits=5):
    """
    Validação walk-forward para séries temporais.

    Split 1: |----TRAIN----|TEST|
    Split 2:   |----TRAIN----|TEST|
    Split 3:     |----TRAIN----|TEST|
    ...
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)

    scores = []
    for train_idx, test_idx in tscv.split(df):
        # Treinar modelo
        # Avaliar no test
        # Guardar métricas
        scores.append(score)

    return np.mean(scores), np.std(scores)
```

**Benefícios:**
- **Robustez:** Múltiplos splits reduzem variância
- **Confiabilidade:** Média de métricas é mais confiável
- **Detecção de overfitting:** Se std é alto, modelo não generaliza

**Custo:**
- Mais tempo de treinamento (5x mais lento)
- Maior complexidade de código

**Recomendação Final:** IMPLEMENTAR (essencial para produção)

---

### 2.7 Batch Size Pequeno ⚠️ BAIXA

**Arquivo:** `config.yaml:75`

```yaml
batch_size: 32
```

**Severidade:** 🟢 BAIXA

**Análise:**
- Batch size de 32 é razoável
- Mas pode causar ruído no gradiente
- Batch size maior = gradiente mais estável = menos overfitting

**Recomendação:**
```yaml
batch_size: 64  # ou 128
```

**Trade-off:**
- Batch maior: Menos overfitting, convergência mais estável
- Batch menor: Mais ruído, pode escapar de mínimos locais

Para dados financeiros: **Batch size maior é melhor**

---

## 3. Outros Problemas Identificados

### 3.1 Ausência de Validação de Estacionariedade

**Problema:**
- Séries temporais financeiras são geralmente não-estacionárias
- Modelo LSTM assume dados estacionários

**Recomendação:**
```python
from statsmodels.tsa.stattools import adfuller

# Testar estacionariedade
def test_stationarity(series):
    result = adfuller(series)
    p_value = result[1]

    if p_value > 0.05:
        print(f"⚠️ Série NÃO estacionária (p={p_value:.4f})")
        print("   Considere diferenciação ou transformação")
    else:
        print(f"✅ Série estacionária (p={p_value:.4f})")

    return p_value < 0.05

# Aplicar no pipeline
test_stationarity(df['Close'])
```

---

### 3.2 Falta de Análise de Resíduos

**Problema:**
- Não há análise dos resíduos (erros) do modelo
- Resíduos podem revelar padrões não capturados

**Recomendação:**
```python
# Após treinamento
residuals = y_test_true - y_test_pred

# Testes nos resíduos
plt.figure(figsize=(12, 4))

# 1. Distribuição
plt.subplot(131)
plt.hist(residuals, bins=50)
plt.title('Distribuição dos Resíduos')

# 2. Q-Q Plot
plt.subplot(132)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')

# 3. Autocorrelação
plt.subplot(133)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, lags=40)
plt.title('Autocorrelação dos Resíduos')

plt.tight_layout()
```

**O que procurar:**
- Resíduos devem ser normalmente distribuídos
- Sem autocorrelação (indica padrão não capturado)
- Sem heteroscedasticidade

---

### 3.3 Sequence Length Fixo

**Arquivo:** `config.yaml:48`

```yaml
sequence_length: 60  # Fixo!
```

**Problema:**
- Não há experimentação com diferentes janelas temporais
- 60 dias pode não ser ideal

**Recomendação:**
```python
# Experimentar múltiplas janelas
sequence_lengths = [30, 45, 60, 90, 120]

for seq_len in sequence_lengths:
    # Treinar modelo
    # Avaliar performance
    # Guardar métricas

# Escolher baseado em validação cruzada
```

---

### 3.4 Ausência de Feature Importance

**Problema:**
- Não há análise de quais features são importantes
- Pode haver features irrelevantes

**Recomendação:**
```python
# Após treinamento, calcular importância
import shap

# SHAP values para LSTM
explainer = shap.DeepExplainer(model, X_train[:100])
shap_values = explainer.shap_values(X_test[:100])

# Visualizar
shap.summary_plot(shap_values, X_test[:100],
                  feature_names=feature_names)
```

**Benefícios:**
- Identificar features irrelevantes
- Remover features que adicionam ruído
- Entender o modelo

---

## 4. Resumo das Recomendações Priorizadas

### 🔴 PRIORIDADE CRÍTICA (Implementar IMEDIATAMENTE)

1. **Corrigir Forward Fill do VIX** (feature_engineering.py:221-224)
   - Substituir `ffill()` por `ffill().shift(1)`
   - Estimativa: 30 minutos
   - Impacto: ALTO

2. **Reduzir Complexidade do Modelo** (config.yaml:50-65)
   - Usar 1-2 camadas LSTM ao invés de 3
   - Reduzir unidades: 64→32 ou 32
   - Estimativa: 15 minutos
   - Impacto: MUITO ALTO

3. **Implementar Walk-Forward Validation**
   - Criar novo módulo de validação cruzada temporal
   - Estimativa: 4-6 horas
   - Impacto: MUITO ALTO

4. **Reduzir Multicolinearidade**
   - Análise de correlação + seleção de features
   - Manter 8-10 features ao invés de 15+
   - Estimativa: 2-3 horas
   - Impacto: ALTO

### 🟡 PRIORIDADE MÉDIA (Implementar em seguida)

5. **Adicionar Regularização L1/L2** (lstm_model.py)
   - Adicionar kernel_regularizer nas camadas LSTM
   - Estimativa: 1 hora
   - Impacto: MÉDIO

6. **Aumentar Dropout** (config.yaml)
   - Mudar de 0.2 para 0.3-0.4
   - Estimativa: 5 minutos
   - Impacto: MÉDIO

7. **Reduzir Early Stopping Patience** (config.yaml)
   - Mudar de 15 para 7-10
   - Estimativa: 5 minutos
   - Impacto: MÉDIO

8. **Implementar Testes de Estacionariedade**
   - Adicionar ao pipeline de validação
   - Estimativa: 1-2 horas
   - Impacto: MÉDIO

### 🟢 PRIORIDADE BAIXA (Melhorias futuras)

9. **Análise de Resíduos**
   - Adicionar ao módulo de visualização
   - Estimativa: 2-3 horas
   - Impacto: BAIXO

10. **Feature Importance (SHAP)**
    - Implementar análise de importância
    - Estimativa: 3-4 horas
    - Impacto: BAIXO

11. **Experimentar Sequence Lengths**
    - Grid search para janela temporal ótima
    - Estimativa: 4-6 horas
    - Impacto: BAIXO

---

## 5. Implementações Sugeridas

### 5.1 Correção do VIX Forward Fill

**Arquivo:** `src/data/feature_engineering.py`

```python
def merge_vix_data(
    self,
    df_main: pd.DataFrame,
    df_vix: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Merge dos dados principais com VIX.

    ✅ ANTI-LEAKAGE: Merge temporal alinhado por data
    """
    if df_vix is None:
        print("⚠️  Sem dados do VIX para mergear")
        return df_main

    df = df_main.copy()

    # Merge por índice (data)
    df = df.join(df_vix, how='left')

    # ❌ ANTES (COM LEAKAGE):
    # df['VIX'] = df['VIX'].fillna(method='ffill')
    # df['VIX'] = df['VIX'].fillna(method='bfill')

    # ✅ DEPOIS (SEM LEAKAGE):
    # Opção 1: Forward fill com shift
    df['VIX'] = df['VIX'].fillna(method='ffill').shift(1)

    # Opção 2: Usar média dos últimos N dias
    # df['VIX'] = df['VIX'].fillna(
    #     df['VIX'].rolling(5, min_periods=1).mean().shift(1)
    # )

    # Para primeiros valores NaN, usar média geral
    df['VIX'] = df['VIX'].fillna(df['VIX'].mean())

    print(f"✓ Dados VIX mergeados: {df['VIX'].notna().sum()} valores")
    return df
```

---

### 5.2 Arquitetura Simplificada

**Arquivo:** `config.yaml`

```yaml
# ❌ ANTES (OVERFITTING):
model:
  lstm_layers:
    - units: 128
      return_sequences: true
      dropout: 0.2
    - units: 64
      return_sequences: true
      dropout: 0.2
    - units: 32
      return_sequences: false
      dropout: 0.2
  dense_layers:
    - units: 16
      activation: "relu"
    - units: 1

# ✅ DEPOIS (MAIS SIMPLES):
model:
  sequence_length: 60

  lstm_layers:
    - units: 64
      return_sequences: true
      dropout: 0.4  # Aumentado!
      kernel_regularizer:
        l1: 0.01
        l2: 0.01
    - units: 32
      return_sequences: false
      dropout: 0.4  # Aumentado!
      kernel_regularizer:
        l1: 0.01
        l2: 0.01

  dense_layers:
    - units: 1
      activation: null

  # Compilação
  optimizer: "adam"
  learning_rate: 0.001
  loss: "mse"
  metrics: ["mae"]
```

**Mudanças:**
- 3 camadas LSTM → 2 camadas LSTM
- 128 unidades → 64 unidades
- Dropout 0.2 → 0.4
- Adicionado regularização L1/L2
- Removida camada Dense intermediária

---

### 5.3 Seleção de Features

**Novo arquivo:** `src/data/feature_selector.py`

```python
"""
Módulo de Seleção de Features
==============================

Remove features redundantes e correlacionadas para prevenir overfitting.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


class FeatureSelector:
    """Seleciona features não-redundantes."""

    def __init__(self, correlation_threshold: float = 0.7):
        """
        Args:
            correlation_threshold: Limiar de correlação (padrão: 0.7)
        """
        self.correlation_threshold = correlation_threshold
        self.selected_features = None

    def remove_correlated_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features altamente correlacionadas.

        Args:
            df: DataFrame com features

        Returns:
            DataFrame com features selecionadas
        """
        # Calcular matriz de correlação
        corr_matrix = df.corr().abs()

        # Triângulo superior da matriz
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Features para remover
        to_drop = [
            column for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
        ]

        print(f"\n🔍 Análise de Correlação:")
        print(f"   Features originais: {len(df.columns)}")
        print(f"   Features removidas: {len(to_drop)}")
        print(f"   Features mantidas: {len(df.columns) - len(to_drop)}")

        if to_drop:
            print(f"\n❌ Features removidas (correlação > {self.correlation_threshold}):")
            for feat in to_drop:
                print(f"     - {feat}")

        # Remover features
        df_selected = df.drop(columns=to_drop)
        self.selected_features = df_selected.columns.tolist()

        return df_selected, to_drop

    def select_top_k_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        k: int = 10
    ) -> List[str]:
        """
        Seleciona top-k features usando mutual information.

        Args:
            X: Array de features
            y: Target
            feature_names: Nomes das features
            k: Número de features a selecionar

        Returns:
            Lista de features selecionadas
        """
        from sklearn.feature_selection import mutual_info_regression

        # Calcular mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)

        # Ordenar por importância
        mi_scores_series = pd.Series(mi_scores, index=feature_names)
        mi_scores_series = mi_scores_series.sort_values(ascending=False)

        # Top-k
        top_features = mi_scores_series.head(k).index.tolist()

        print(f"\n🎯 Top-{k} Features (Mutual Information):")
        for i, (feat, score) in enumerate(mi_scores_series.head(k).items(), 1):
            print(f"   {i}. {feat}: {score:.4f}")

        return top_features


# Integrar no pipeline
if __name__ == "__main__":
    # Exemplo de uso
    selector = FeatureSelector(correlation_threshold=0.7)

    # Após criar features
    df_selected, dropped = selector.remove_correlated_features(df_features)
```

**Uso no pipeline:**

```python
# Em scripts/train_model.py, após feature engineering

# Selecionar features
from src.data.feature_selector import FeatureSelector

selector = FeatureSelector(correlation_threshold=0.7)
df_features_selected, dropped_features = selector.remove_correlated_features(df_features)

# Continuar com preprocessing
data = preprocessor.prepare_data(df_features_selected, target_column='Close')
```

---

### 5.4 Walk-Forward Validation

**Novo arquivo:** `src/validation/walk_forward.py`

```python
"""
Walk-Forward Validation para Séries Temporais
==============================================

Implementa validação cruzada temporal (time series cross-validation).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable
from sklearn.model_selection import TimeSeriesSplit


class WalkForwardValidator:
    """Validação walk-forward para séries temporais."""

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = None
    ):
        """
        Args:
            n_splits: Número de splits
            test_size: Tamanho do conjunto de teste (em amostras)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.results = []

    def split(self, df: pd.DataFrame) -> List[Tuple[pd.Index, pd.Index]]:
        """
        Gera splits temporais.

        Args:
            df: DataFrame com índice temporal

        Returns:
            Lista de tuplas (train_index, test_index)
        """
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self.test_size
        )

        splits = []
        for train_idx, test_idx in tscv.split(df):
            train_index = df.index[train_idx]
            test_index = df.index[test_idx]
            splits.append((train_index, test_index))

            print(f"\nSplit {len(splits)}:")
            print(f"  Treino: {train_index[0]} a {train_index[-1]} ({len(train_index)} amostras)")
            print(f"  Teste:  {test_index[0]} a {test_index[-1]} ({len(test_index)} amostras)")

        return splits

    def validate(
        self,
        df: pd.DataFrame,
        train_func: Callable,
        eval_func: Callable
    ) -> Dict[str, np.ndarray]:
        """
        Executa validação walk-forward.

        Args:
            df: DataFrame com features
            train_func: Função de treinamento
            eval_func: Função de avaliação

        Returns:
            Dicionário com métricas de cada split
        """
        splits = self.split(df)

        all_metrics = []

        for i, (train_idx, test_idx) in enumerate(splits, 1):
            print(f"\n{'='*60}")
            print(f"EXECUTANDO SPLIT {i}/{self.n_splits}")
            print(f"{'='*60}")

            # Separar dados
            df_train = df.loc[train_idx]
            df_test = df.loc[test_idx]

            # Treinar modelo
            model, preprocessor = train_func(df_train)

            # Avaliar
            metrics = eval_func(model, preprocessor, df_test)
            all_metrics.append(metrics)

            print(f"\nMétricas do Split {i}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        # Agregar resultados
        aggregated = {}
        metric_names = all_metrics[0].keys()

        for metric in metric_names:
            values = [m[metric] for m in all_metrics]
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
            aggregated[f"{metric}_values"] = values

        print(f"\n{'='*60}")
        print("RESULTADOS AGREGADOS")
        print(f"{'='*60}")

        for metric in metric_names:
            mean = aggregated[f"{metric}_mean"]
            std = aggregated[f"{metric}_std"]
            print(f"{metric}: {mean:.4f} ± {std:.4f}")

        return aggregated


# Exemplo de uso
if __name__ == "__main__":

    def train_model(df_train):
        """Função de treinamento."""
        # Implementar treinamento
        # Retornar model, preprocessor
        pass

    def evaluate_model(model, preprocessor, df_test):
        """Função de avaliação."""
        # Implementar avaliação
        # Retornar dict de métricas
        pass

    # Executar validação
    validator = WalkForwardValidator(n_splits=5)
    results = validator.validate(df, train_model, evaluate_model)
```

---

## 6. Checklist de Implementação

### Fase 1: Correções Críticas (1-2 dias)

- [ ] **Corrigir VIX forward fill** (30 min)
  - Modificar `feature_engineering.py:221-227`
  - Adicionar testes unitários
  - Validar que não há leakage

- [ ] **Reduzir complexidade do modelo** (1 hora)
  - Modificar `config.yaml:50-65`
  - 3 camadas → 2 camadas
  - 128 unidades → 64 unidades
  - Treinar novo modelo e comparar

- [ ] **Aumentar dropout e adicionar regularização** (2 horas)
  - Dropout: 0.2 → 0.4
  - Adicionar L1/L2 regularization
  - Modificar `lstm_model.py:93-104`
  - Atualizar config.yaml

- [ ] **Implementar seleção de features** (3 horas)
  - Criar `feature_selector.py`
  - Análise de correlação
  - Integrar no pipeline
  - Reduzir de 15+ → 8-10 features

### Fase 2: Validação Robusta (2-3 dias)

- [ ] **Implementar walk-forward validation** (6 horas)
  - Criar `walk_forward.py`
  - Integrar no pipeline de treinamento
  - Executar com 5 splits
  - Comparar com split único

- [ ] **Análise de estacionariedade** (2 horas)
  - Adicionar testes ADF
  - Implementar diferenciação se necessário
  - Validar impacto nas métricas

- [ ] **Ajustar early stopping** (15 min)
  - Patience: 15 → 7
  - Testar impacto

### Fase 3: Análises Adicionais (1-2 dias)

- [ ] **Análise de resíduos** (3 horas)
  - Implementar visualizações
  - Testes de normalidade
  - Autocorrelação

- [ ] **Feature importance** (4 horas)
  - Implementar SHAP
  - Visualizações
  - Remover features irrelevantes

- [ ] **Experimentação com sequence_length** (4 horas)
  - Testar [30, 45, 60, 90, 120]
  - Validação cruzada
  - Escolher ótimo

### Fase 4: Documentação e Testes (1 dia)

- [ ] **Testes unitários**
  - Testar pipeline completo
  - Validar anti-leakage
  - Cobertura > 80%

- [ ] **Atualizar documentação**
  - README com mudanças
  - Documentar decisões
  - Guia de uso

- [ ] **Relatório final**
  - Comparar antes vs. depois
  - Métricas em produção
  - Lições aprendidas

---

## 7. Métricas Esperadas

### Antes das Correções (Estimativa)

```
Conjunto de VALIDAÇÃO:
- R²: 0.85-0.95 (suspeito de overfitting)
- MAPE: 0.5-1.5% (muito bom)
- Direction Accuracy: 65-75%

Conjunto de TESTE:
- R²: 0.45-0.65 (drop significativo!)
- MAPE: 3-5% (piora considerável)
- Direction Accuracy: 50-55% (quase aleatório)

⚠️ Gap entre validação e teste indica OVERFITTING
```

### Depois das Correções (Esperado)

```
Conjunto de VALIDAÇÃO:
- R²: 0.55-0.70 (mais realista)
- MAPE: 2-3%
- Direction Accuracy: 58-63%

Conjunto de TESTE:
- R²: 0.50-0.68 (gap reduzido!)
- MAPE: 2.5-3.5% (consistente)
- Direction Accuracy: 56-62% (melhor generalização)

✅ Menor gap = Melhor generalização
```

**Indicadores de Sucesso:**
1. **Gap Reduzido:** Diferença val-test < 10%
2. **Consistência:** Std baixo no walk-forward (< 15%)
3. **Melhor que Baseline:** Supera naive em 10%+
4. **Direction Accuracy:** > 55% (melhor que aleatório)

---

## 8. Conclusões e Próximos Passos

### 8.1 Pontos Fortes do Projeto ✅

1. **Excelente estrutura de código**
   - Modular e bem organizado
   - Separação clara de responsabilidades
   - Documentação adequada

2. **Pipeline anti-leakage bem implementado**
   - Split temporal correto
   - Normalização adequada
   - Sequências LSTM corretas

3. **Testes automáticos**
   - Módulo de validação anti-leakage
   - Comparação com baselines

4. **Boas práticas de ML**
   - Seeds para reprodutibilidade
   - Early stopping
   - Model checkpoint

### 8.2 Áreas Críticas que Precisam de Atenção 🔴

1. **Forward fill do VIX** - pode causar leakage
2. **Modelo muito complexo** - garantia de overfitting
3. **Features redundantes** - multicolinearidade alta
4. **Falta de validação cruzada temporal** - robustez questionável
5. **Regularização insuficiente** - dropout e L1/L2

### 8.3 Roadmap Recomendado

**Semana 1: Correções Críticas**
- Corrigir VIX forward fill
- Reduzir complexidade do modelo
- Adicionar regularização L1/L2
- Implementar seleção de features

**Semana 2: Validação Robusta**
- Implementar walk-forward validation
- Análise de estacionariedade
- Testes de resíduos

**Semana 3: Otimização**
- Feature importance
- Experimentação com hiperparâmetros
- Grid search para architecture

**Semana 4: Produção**
- Testes finais
- Documentação
- Deploy

### 8.4 Estimativa de Impacto

**Impacto Esperado nas Métricas:**

| Métrica | Antes (estimado) | Depois (esperado) | Melhoria |
|---------|-----------------|-------------------|----------|
| R² (teste) | 0.50-0.65 | 0.55-0.70 | +5-10% |
| MAPE (teste) | 3-5% | 2.5-3.5% | -15-30% |
| Direction Acc | 50-55% | 56-62% | +6-12% |
| Val-Test Gap | 20-40% | 5-15% | -60% ✅ |

**Mais importante:** Redução do gap val-test indica melhor generalização!

### 8.5 Recomendação Final

**Nível de Confiança Atual:** ⚠️ MÉDIO (60%)

**Recomendação:**
1. **NÃO** colocar em produção sem correções críticas
2. **IMPLEMENTAR** todas as correções de prioridade crítica
3. **VALIDAR** com walk-forward antes de deploy
4. **MONITORAR** performance em produção continuamente

**Com as correções implementadas:**
- Nível de Confiança: ✅ ALTO (85-90%)
- Pronto para produção: SIM (com monitoramento)

---

## 9. Recursos Adicionais

### Papers Relevantes

1. **Fischer & Krauss (2018)**
   - "Deep learning with long short-term memory networks for financial market predictions"
   - DOI: 10.1016/j.ejor.2017.11.054

2. **Nelson et al. (2017)**
   - "Stock market's price movement prediction with LSTM neural networks"
   - IEEE IJCNN 2017

3. **Bao et al. (2017)**
   - "A deep learning framework for financial time series using stacked autoencoders and long-short term memory"
   - PLOS ONE

### Livros Recomendados

1. **"Advances in Financial Machine Learning"** - Marcos López de Prado
   - Capítulo sobre data leakage
   - Validação cruzada temporal

2. **"Machine Learning for Asset Managers"** - Marcos López de Prado
   - Feature selection
   - Overfitting em finanças

### Ferramentas

1. **SHAP:** Interpretabilidade de modelos
2. **MLflow:** Tracking de experimentos
3. **Optuna:** Otimização de hiperparâmetros
4. **TensorBoard:** Monitoramento de treinamento

---

## 10. Contato e Suporte

Para dúvidas ou discussões sobre esta auditoria:

- **Issues:** Abra uma issue no repositório
- **Discussões:** Use a aba de Discussions
- **Code Review:** Pull requests são bem-vindos

---

**Última Atualização:** 2025-11-16
**Versão:** 1.0
**Status:** Auditoria Completa ✅
