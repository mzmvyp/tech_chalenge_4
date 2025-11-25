"""
Script de Diagnostico - Predicoes do Modelo
===========================================

Verifica se o modelo esta fazendo predicoes corretas ou se ha problema
na escala/processamento das predicoes.

Uso:
    python scripts/diagnostico_predicoes.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

print("="*60)
print("DIAGNOSTICO: PREDICOES DO MODELO")
print("="*60)

# Carregar modelo
model_path = "models/lstm_model.h5"
if not Path(model_path).exists():
    print(f"[ERRO] Modelo nao encontrado: {model_path}")
    sys.exit(1)

print(f"\nCarregando modelo de: {model_path}")
try:
    model = tf.keras.models.load_model(model_path, compile=False)
except:
    # Tentar com compile=True se falhar
    model = tf.keras.models.load_model(model_path)
print(f"[OK] Modelo carregado")

# Carregar scaler
scaler_path = "models/scaler.pkl"
scaler_data = joblib.load(scaler_path)
scaler = scaler_data['scaler']
target_idx = scaler_data.get('target_idx', 0)
feature_names = scaler_data.get('feature_names', [])

print(f"\nScaler carregado:")
print(f"   Target index: {target_idx}")
print(f"   Features: {len(feature_names)}")

# Carregar dados de teste
print(f"\n" + "="*60)
print("CARREGANDO DADOS DE TESTE")
print("="*60)

# Carregar feature_config para obter sequence_length
import json
config_path = Path("data/processed/feature_config.json")
if config_path.exists():
    with open(config_path, 'r') as f:
        feature_config = json.load(f)
    sequence_length = feature_config.get('sequence_length', 60)
else:
    sequence_length = 60

# Carregar dados processados
from src.data.preprocessor import TimeSeriesPreprocessor
preprocessor = TimeSeriesPreprocessor(
    sequence_length=sequence_length,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15
)

# Carregar features
features_path = Path("data/processed/features.csv")
df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)

# Preparar dados (usar o mesmo preprocessor que foi usado no treino)
preprocessor.load_scaler(scaler_path)
df_train, df_val, df_test = preprocessor.temporal_split(df_features, verbose=False)

# Transformar
test_scaled = preprocessor.transform(df_test)
X_test, y_test = preprocessor.create_sequences(test_scaled, target_idx)

print(f"\nDados de teste preparados:")
print(f"   X_test shape: {X_test.shape}")
print(f"   y_test shape: {y_test.shape}")

# Fazer predicoes
print(f"\n" + "="*60)
print("FAZENDO PREDICOES")
print("="*60)

y_pred_scaled = model.predict(X_test, verbose=0).flatten()

print(f"\nPredicoes normalizadas (primeiras 10):")
print(f"   {y_pred_scaled[:10]}")

print(f"\nEstatisticas das predicoes normalizadas:")
print(f"   Min: {y_pred_scaled.min():.6f}")
print(f"   Max: {y_pred_scaled.max():.6f}")
print(f"   Media: {y_pred_scaled.mean():.6f}")
print(f"   Std: {y_pred_scaled.std():.6f}")

# Verificar se estao na escala correta (0-1 para MinMaxScaler)
if y_pred_scaled.min() < -0.1 or y_pred_scaled.max() > 1.1:
    print(f"\n[AVISO] Predicoes fora da escala esperada (0-1)!")
    print(f"   Isso pode indicar problema no modelo.")

# Valores reais normalizados
print(f"\nValores reais normalizados (primeiros 10):")
print(f"   {y_test[:10]}")

print(f"\nEstatisticas dos valores reais normalizados:")
print(f"   Min: {y_test.min():.6f}")
print(f"   Max: {y_test.max():.6f}")
print(f"   Media: {y_test.mean():.6f}")
print(f"   Std: {y_test.std():.6f}")

# Fazer inverse transform
print(f"\n" + "="*60)
print("INVERSE TRANSFORM")
print("="*60)

y_pred = preprocessor.inverse_transform_target(y_pred_scaled, 'Close')
y_true = preprocessor.inverse_transform_target(y_test, 'Close')

print(f"\nPredicoes originais (primeiras 10):")
print(f"   {y_pred[:10]}")

print(f"\nValores reais originais (primeiros 10):")
print(f"   {y_true[:10]}")

# Calcular metricas
print(f"\n" + "="*60)
print("METRICAS")
print("="*60)

mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# R2
ss_res = np.sum((y_true - y_pred)**2)
ss_tot = np.sum((y_true - np.mean(y_true))**2)
r2 = 1 - (ss_res / ss_tot)

print(f"\nMAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R2:   {r2:.4f}")

# Comparar com naive
naive_pred = np.roll(y_true, 1)
naive_pred[0] = y_true[0]  # Primeiro valor usa ele mesmo
naive_mae = np.mean(np.abs(y_true - naive_pred))
naive_rmse = np.sqrt(np.mean((y_true - naive_pred)**2))

print(f"\nComparacao com Naive Forecast:")
print(f"   LSTM MAE:  {mae:.2f} vs Naive MAE:  {naive_mae:.2f}")
print(f"   LSTM RMSE: {rmse:.2f} vs Naive RMSE: {naive_rmse:.2f}")

# Analise de erros
print(f"\n" + "="*60)
print("ANALISE DE ERROS")
print("="*60)

errors = y_true - y_pred
print(f"\nErros (primeiros 10):")
print(f"   {errors[:10]}")

print(f"\nEstatisticas dos erros:")
print(f"   Media: {errors.mean():.2f}")
print(f"   Std:   {errors.std():.2f}")
print(f"   Min:   {errors.min():.2f}")
print(f"   Max:   {errors.max():.2f}")

# Verificar se ha padrao sistematico
if abs(errors.mean()) > 10:
    print(f"\n[AVISO] Erro medio muito alto ({errors.mean():.2f})")
    print(f"   O modelo pode estar com bias sistematico")

# Verificar se predicoes estao na escala correta
print(f"\n" + "="*60)
print("VERIFICACAO DE ESCALA")
print("="*60)

print(f"\nValores reais:")
print(f"   Min: {y_true.min():.2f}")
print(f"   Max: {y_true.max():.2f}")
print(f"   Media: {y_true.mean():.2f}")

print(f"\nPredicoes:")
print(f"   Min: {y_pred.min():.2f}")
print(f"   Max: {y_pred.max():.2f}")
print(f"   Media: {y_pred.mean():.2f}")

if y_pred.min() < y_true.min() * 0.5 or y_pred.max() > y_true.max() * 1.5:
    print(f"\n[ERRO CRITICO] Predicoes fora da escala dos valores reais!")
    print(f"   Isso indica problema grave no inverse_transform ou no modelo")

print("\n" + "="*60)
print("[OK] DIAGNOSTICO CONCLUIDO")
print("="*60)

