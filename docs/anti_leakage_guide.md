# 🔒 Guia Completo Anti-Data-Leakage

## O que é Data Leakage?

**Data leakage** (vazamento de dados) ocorre quando informação do **futuro** "vaza" para o **passado** durante o treinamento de um modelo de Machine Learning. Isso resulta em métricas artificialmente boas que **não se reproduzem em produção**.

### Exemplo Simples

```python
# ❌ ERRADO - Data Leakage!
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data)  # Fitou em TODOS os dados
X_train, X_test = train_test_split(X_scaled)

# Por que é ruim?
# O scaler "viu" os dados de teste durante o fit!
# O modelo terá vantagem injusta.
```

```python
# ✅ CORRETO - Sem Leakage
X_train, X_test = train_test_split(data)  # Split PRIMEIRO
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit apenas no treino
X_test_scaled = scaler.transform(X_test)  # Transform no teste
```

---

## Por que Data Leakage é Especialmente Perigoso em Séries Temporais?

Em séries temporais, temos uma dimensão adicional: **tempo**.

### Problemas Comuns

1. **Split aleatório (shuffle)**
   - Mistura passado e futuro
   - Modelo "vê" o futuro durante treinamento

2. **Features com informação futura**
   - Média móvel calculada incorretamente
   - Indicadores que usam dados futuros

3. **Normalização incorreta**
   - Scaler fitado em todo o dataset
   - Min/max incluem valores futuros

4. **Sequências temporais erradas**
   - Sequências que cruzam períodos de train/test
   - Sobreposição temporal

---

## 🛡️ Proteções Implementadas Neste Projeto

### 1. Split Temporal PRIMEIRO

```python
# ⚠️ ORDEM CRÍTICA (NUNCA MUDAR):
# 1. Split temporal
# 2. Feature engineering (se necessário)
# 3. Normalização
# 4. Criação de sequências

# ✅ Implementação correta
def temporal_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Split temporal sem shuffle.
    """
    # Garantir ordem temporal
    df = df.sort_index()

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Split temporal (sem shuffle!)
    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    # ✅ Validação: sem sobreposição temporal
    assert df_train.index[-1] < df_val.index[0]
    assert df_val.index[-1] < df_test.index[0]

    return df_train, df_val, df_test
```

### 2. Normalização Correta

```python
# ✅ Scaler fit APENAS no treino
scaler = MinMaxScaler()

# FIT apenas no treino
train_scaled = scaler.fit_transform(df_train)

# TRANSFORM (sem fit!) em val e test
val_scaled = scaler.transform(df_val)
test_scaled = scaler.transform(df_test)

# ❌ NUNCA faça:
# val_scaled = scaler.fit_transform(df_val)  # ERRADO!
# test_scaled = scaler.fit_transform(df_test)  # ERRADO!
```

### 3. Features Sem Informação Futura

```python
# ✅ Média móvel correta
df['MA_20'] = df['Close'].rolling(
    window=20,
    min_periods=20  # Não usa futuro
).mean()

# ❌ ERRADO - usa futuro
df['MA_20'] = df['Close'].rolling(
    window=20,
    center=True  # ERRADO! Usa dados futuros
).mean()

# ✅ Retornos corretos (olha para trás)
df['Return'] = df['Close'].pct_change(periods=1)

# ❌ ERRADO - olha para frente
df['Future_Return'] = df['Close'].shift(-1)  # ERRADO!
```

### 4. Sequências Temporais Corretas

```python
# ✅ Sequências respeitam ordem temporal
def create_sequences(data, sequence_length=60):
    """
    Cria sequências temporais corretas.
    """
    X, y = [], []

    for i in range(sequence_length, len(data)):
        # Sequência: últimos 60 dias
        X.append(data[i - sequence_length:i])

        # Alvo: próximo dia (futuro APÓS a sequência)
        y.append(data[i, target_idx])

    return np.array(X), np.array(y)

# ❌ ERRADO - sequências que misturam períodos
# Criar sequências ANTES do split
# ou sequências que cruzam train/val/test
```

---

## 🧪 Testes Automatizados

### Teste 1: Ordem Temporal

```python
def test_temporal_order(train_dates, val_dates, test_dates):
    """
    Verifica se train < val < test temporalmente.
    """
    train_max = train_dates.max()
    val_min = val_dates.min()
    val_max = val_dates.max()
    test_min = test_dates.min()

    # Train termina antes de val começar
    assert train_max < val_min, "Treino sobrepõe validação!"

    # Val termina antes de test começar
    assert val_max < test_min, "Validação sobrepõe teste!"

    # Dados estão ordenados
    assert train_dates.is_monotonic_increasing
    assert val_dates.is_monotonic_increasing
    assert test_dates.is_monotonic_increasing
```

### Teste 2: Sem Sobreposição

```python
def test_no_overlap(train_idx, val_idx, test_idx):
    """
    Verifica que não há amostras compartilhadas.
    """
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    # Nenhuma interseção
    assert len(train_set & val_set) == 0, "Treino e val compartilham dados!"
    assert len(train_set & test_set) == 0, "Treino e teste compartilham dados!"
    assert len(val_set & test_set) == 0, "Val e teste compartilham dados!"
```

### Teste 3: Performance Suspeita

```python
def test_suspicious_performance(r2_score, mape):
    """
    Detecta performance suspeita (muito boa = possível leakage).
    """
    # R² > 0.95 em finanças é MUITO suspeito
    if r2_score > 0.95:
        warnings.warn(
            f"R² muito alto ({r2_score:.4f})! "
            "Possível data leakage. "
            "Séries temporais financeiras raramente têm R² > 0.95."
        )

    # MAPE < 0.1% é suspeito
    if mape < 0.1:
        warnings.warn(
            f"MAPE muito baixo ({mape:.4f}%)! "
            "Possível data leakage."
        )
```

### Teste 4: Melhor que Baseline

```python
def test_better_than_baseline(model_rmse, baseline_rmse):
    """
    Modelo DEVE ser melhor que naive forecast.
    """
    improvement = (baseline_rmse - model_rmse) / baseline_rmse

    if improvement < 0:
        warnings.warn(
            "Modelo PIOR que naive forecast! "
            "Pode indicar problemas."
        )

    if improvement < 0.05:
        warnings.warn(
            f"Melhoria muito pequena ({improvement*100:.2f}%). "
            "Modelo pode não estar aprendendo."
        )
```

### Teste 5: Scaler Validation

```python
def test_scaler_fit_train_only(train_data, val_data, scaler):
    """
    Verifica que scaler foi fitado apenas no treino.
    """
    train_min = train_data.min(axis=0)
    train_max = train_data.max(axis=0)

    scaler_min = scaler.data_min_
    scaler_max = scaler.data_max_

    # Scaler min/max devem corresponder ao treino
    # (com pequena tolerância para erros numéricos)
    np.testing.assert_allclose(
        scaler_min, train_min,
        rtol=1e-3,
        err_msg="Scaler min não corresponde ao treino!"
    )

    np.testing.assert_allclose(
        scaler_max, train_max,
        rtol=1e-3,
        err_msg="Scaler max não corresponde ao treino!"
    )
```

---

## 📋 Checklist Anti-Leakage

Antes de colocar um modelo em produção, verifique:

### Dados
- [ ] Split temporal foi feito PRIMEIRO
- [ ] Sem shuffle nos dados
- [ ] Sem sobreposição entre train/val/test
- [ ] Dados ordenados temporalmente

### Preprocessamento
- [ ] Scaler fit() apenas no treino
- [ ] Transform() em val e test (sem fit!)
- [ ] Mesmas transformações em todos os conjuntos

### Features
- [ ] Nenhuma feature usa informação futura
- [ ] Médias móveis com min_periods correto
- [ ] Lags olham para trás (não para frente)
- [ ] Features calculadas APÓS split

### Modelo
- [ ] Sequências respeitam ordem temporal
- [ ] Nenhuma sequência cruza train/val/test
- [ ] Validação usa dados futuros ao treino

### Avaliação
- [ ] Métricas calculadas apenas em test
- [ ] Comparação com baselines
- [ ] Performance não é "boa demais"
- [ ] R² < 0.95 (para finanças)
- [ ] MAPE > 0.1% (para finanças)

### Testes
- [ ] Todos os testes anti-leakage passaram
- [ ] Sem warnings suspeitos
- [ ] Modelo supera baseline
- [ ] Validação temporal OK

---

## 🚨 Sinais de Alerta

### Performance Suspeitamente Boa

```
R² = 0.99    ⚠️  MUITO SUSPEITO!
MAPE = 0.05% ⚠️  MUITO SUSPEITO!

Ação: Revisar pipeline completo
```

### Modelo Pior que Baseline

```
RMSE Modelo: 25.5
RMSE Naive:  18.3  ⚠️  Modelo PIOR!

Ação: Modelo não está aprendendo
```

### Erro no Train << Erro no Test

```
Train MAE: 2.5
Test MAE:  45.8  ⚠️  Overfitting extremo!

Ação: Possível leakage ou overfitting
```

---

## 💡 Boas Práticas

### 1. Sempre Comece com Baseline

```python
# Naive forecast: amanhã = hoje
baseline_pred = y_true[:-1]
baseline_metrics = calculate_metrics(y_true[1:], baseline_pred)

# Seu modelo DEVE superar isso
if model_rmse >= baseline_rmse:
    raise ValueError("Modelo não supera baseline!")
```

### 2. Walk-Forward Validation

Para validação extra-rigorosa:

```python
# Treinar em dados até 2021
# Testar em 2022
# Re-treinar incluindo 2022
# Testar em 2023
# etc...
```

### 3. Documentation

Documente TODAS as decisões:
- Por que feature X foi criada assim
- Por que janela de Y dias
- Por que esse split ratio

### 4. Code Review

Peça para alguém revisar:
- Ordem das operações
- Fit vs transform
- Features temporais

---

## 📚 Recursos Adicionais

### Artigos Recomendados
- [Avoiding Look-Ahead Bias in Time Series](https://www.google.com)
- [Data Leakage in ML](https://www.kaggle.com)

### Livros
- "Hands-On Machine Learning" - Aurélien Géron (Capítulo sobre Time Series)
- "Forecasting: Principles and Practice" - Hyndman & Athanasopoulos

---

## 🎓 Exercícios para Praticar

### Exercício 1: Encontre o Leakage

```python
# Onde está o leakage?
data = pd.read_csv('stock_data.csv')
data['MA_20'] = data['Close'].rolling(20, center=True).mean()  # ???
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Close', 'MA_20']])  # ???
X_train, X_test = train_test_split(data_scaled, shuffle=True)  # ???
```

<details>
<summary>Resposta</summary>

Três problemas:
1. `center=True` usa dados futuros na média móvel
2. `fit_transform` em todo dataset antes do split
3. `shuffle=True` mistura passado e futuro
</details>

### Exercício 2: Corrija o Código

```python
# Versão correta
data = pd.read_csv('stock_data.csv')

# Split PRIMEIRO
train, test = temporal_split(data)

# Features (sem center!)
train['MA_20'] = train['Close'].rolling(20, min_periods=20).mean()
test['MA_20'] = test['Close'].rolling(20, min_periods=20).mean()

# Scaler fit apenas no treino
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train[['Close', 'MA_20']])
test_scaled = scaler.transform(test[['Close', 'MA_20']])

# Sem shuffle!
```

---

**Última atualização:** 2024-11-16

---

**⚠️ LEMBRE-SE:** Data leakage é uma das causas mais comuns de modelos que funcionam bem em desenvolvimento mas falham em produção. Seja rigoroso!
