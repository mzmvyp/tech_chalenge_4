# Guia de Online Learning (Auto Learning)

**Data:** 2025-11-25  
**Status:** ✅ Implementado

---

## 🎯 O que é Online Learning?

Online Learning permite que o modelo **aprenda continuamente** com novos dados e melhore ao longo do tempo, sem precisar retreinar do zero.

**Como funciona:**
1. Modelo faz predição
2. Quando valor real é conhecido, adiciona ao buffer
3. Quando buffer atinge threshold (ex: 50 exemplos), retreina automaticamente
4. Modelo melhora com o tempo!

---

## 🚀 Como Usar

### 1. Inicializar Online Learner

```python
from src.models.online_learner import OnlineLearner

learner = OnlineLearner(
    retrain_threshold=50,  # Retreinar quando tiver 50 novos exemplos
    fine_tune_epochs=5,   # Épocas para fine-tuning
    fine_tune_lr=0.0001   # Learning rate menor que treino inicial
)

learner.load_model_and_scaler()
```

### 2. Fazer Predição e Aprender

```python
# Fazer predição
result = learner.predict_with_learning(
    sequence=sequence,  # Sequência de entrada
    actual_return=actual_return  # Return real (quando disponível)
)

# Se tiver valor real, o sistema automaticamente:
# - Adiciona ao buffer
# - Retreina quando buffer está cheio
```

### 3. Adicionar Feedback Manualmente

```python
learner.add_prediction_feedback(
    features=features_df,
    actual_return=0.02,
    predicted_return=0.015,
    actual_close=4500.0,
    predicted_close=4480.0,
    date=datetime.now()
)
```

### 4. Retreinar Manualmente

```python
# Se tiver novos dados preparados
learner.retrain(
    X_new=X_new_sequences,
    y_new=y_new_targets
)
```

---

## ⚙️ Configuração

Edite `config.yaml`:

```yaml
online_learning:
  enabled: true
  retrain_threshold: 50      # Quantos exemplos antes de retreinar
  fine_tune_epochs: 5        # Épocas para fine-tuning
  fine_tune_lr: 0.0001       # Learning rate (menor que treino inicial)
  buffer_path: "data/processed/online_learning_buffer.csv"
```

---

## 💡 Exemplo Completo

```python
from src.models.online_learner import OnlineLearner
import numpy as np
from datetime import datetime

# Criar learner
learner = OnlineLearner(retrain_threshold=5)  # Baixo para demo
learner.load_model_and_scaler()

# Simular predições diárias
for day in range(10):
    # Fazer predição
    sequence = get_latest_sequence()  # Sua função
    result = learner.predict_with_learning(
        sequence=sequence,
        actual_return=get_actual_return(day)  # Quando disponível
    )
    
    print(f"Dia {day+1}: Return predito = {result['predicted_return']:.4f}")
    print(f"Buffer: {len(learner.new_data_buffer)}/{learner.retrain_threshold}")
    
    # Quando buffer atinge threshold, retreina automaticamente!
```

---

## 📊 Benefícios

1. **Melhoria Contínua**: Modelo aprende com seus próprios erros
2. **Adaptação**: Ajusta-se a mudanças no mercado
3. **Eficiência**: Fine-tuning é mais rápido que treinar do zero
4. **Automação**: Retreina automaticamente quando necessário

---

## ⚠️ Considerações

- **Fine-tuning**: Usa learning rate menor para ajustes sutis
- **Buffer**: Mantém histórico de predições e resultados
- **Validação**: Sempre valide performance após retreinar
- **Monitoramento**: Acompanhe métricas para detectar degradação

---

## 🔧 Próximos Passos

Para implementação completa em produção:
1. Integrar com pipeline de dados
2. Adicionar validação de performance após retreinar
3. Implementar rollback se performance degradar
4. Adicionar monitoramento e alertas

