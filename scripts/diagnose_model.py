"""
Diagnóstico do Modelo
=====================

Analisa o modelo treinado para identificar problemas.

Uso:
    python scripts/diagnose_model.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json

from src.config import get_config
from src.data.data_loader import DataLoader
from src.data.feature_engineering_stationary import create_stationary_features
from src.data.preprocessor import TimeSeriesPreprocessor
from src.evaluation.metrics import calculate_all_metrics

def diagnose():
    """Diagnostica o modelo."""
    
    print("\n" + "="*60)
    print("DIAGNOSTICO DO MODELO")
    print("="*60)
    
    # Carregar modelo
    model = tf.keras.models.load_model("models/lstm_model.h5", compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Carregar dados
    config = get_config()
    data_config = config.get_data_config()
    
    loader = DataLoader(
        symbol=data_config['symbol'],
        start_date=data_config['start_date'],
        end_date=data_config['end_date'],
        interval=data_config['interval'],
        vix_symbol=data_config.get('vix_symbol')
    )
    
    df_main, df_vix = loader.load_all_data()
    close_series = df_main['Close'].copy()
    
    # Features
    df_features = create_stationary_features(df_main, df_vix)
    close_series = close_series.loc[df_features.index]
    
    # Scaler
    scaler_data = joblib.load("models/scaler.pkl")
    if isinstance(scaler_data, dict):
        scaler = scaler_data['scaler']
        target_idx = scaler_data.get('target_idx', 0)
    else:
        scaler = scaler_data
        target_idx = 0
    
    # Preprocessor
    model_config = config.get_model_config()
    preprocessor = TimeSeriesPreprocessor(
        sequence_length=model_config['sequence_length'],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Preparar dados
    data = preprocessor.prepare_data(df_features, target_column='Return', verbose=False)
    
    # Fazer algumas predições
    print("\nAnalisando predicoes...")
    
    # Test set
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Predições
    y_pred_scaled = model.predict(X_test[:10], verbose=0).flatten()
    
    # Inverse transform
    y_pred_return = []
    y_true_return = []
    
    for i in range(10):
        # Predito
        dummy = np.zeros((1, scaler.n_features_in_))
        dummy[0, target_idx] = y_pred_scaled[i]
        pred_return = scaler.inverse_transform(dummy)[0, target_idx]
        y_pred_return.append(pred_return)
        
        # Real
        dummy[0, target_idx] = y_test[i]
        true_return = scaler.inverse_transform(dummy)[0, target_idx]
        y_true_return.append(true_return)
    
    print("\nPrimeiras 10 predicoes (Return):")
    print(f"{'Real':<15} {'Predito':<15} {'Erro':<15} {'Erro %':<15}")
    print("-" * 60)
    
    for i in range(10):
        real = y_true_return[i]
        pred = y_pred_return[i]
        error = abs(real - pred)
        error_pct = (error / abs(real) * 100) if real != 0 else 0
        print(f"{real:<15.6f} {pred:<15.6f} {error:<15.6f} {error_pct:<15.2f}%")
    
    # Estatísticas dos Returns
    print("\nEstatisticas dos Returns:")
    print(f"   Real - Media: {np.mean(y_true_return):.6f}, Std: {np.std(y_true_return):.6f}")
    print(f"   Predito - Media: {np.mean(y_pred_return):.6f}, Std: {np.std(y_pred_return):.6f}")
    print(f"   Range Real: [{np.min(y_true_return):.6f}, {np.max(y_true_return):.6f}]")
    print(f"   Range Predito: [{np.min(y_pred_return):.6f}, {np.max(y_pred_return):.6f}]")
    
    # Verificar se predições estão na escala correta
    print("\nVerificacoes:")
    
    # 1. Returns devem estar entre -1 e 1 (geralmente)
    if np.any(np.abs(y_pred_return) > 1):
        print("   ALERTA: Algumas predicoes de Return estao fora do range [-1, 1]")
        print(f"      Max abs: {np.max(np.abs(y_pred_return)):.6f}")
    else:
        print("   OK: Returns preditos estao em range razoavel")
    
    # 2. Verificar se há NaN ou Inf
    if np.any(np.isnan(y_pred_return)) or np.any(np.isinf(y_pred_return)):
        print("   ALERTA: Predicoes contem NaN ou Inf!")
    else:
        print("   OK: Sem NaN ou Inf nas predicoes")
    
    # 3. Verificar variância
    if np.std(y_pred_return) < 0.0001:
        print("   ALERTA: Predicoes tem variancia muito baixa (modelo pode estar colapsando)")
    else:
        print(f"   OK: Variancia das predicoes: {np.std(y_pred_return):.6f}")
    
    print("\n" + "="*60)
    print("DIAGNOSTICO CONCLUIDO")
    print("="*60)


if __name__ == "__main__":
    diagnose()

