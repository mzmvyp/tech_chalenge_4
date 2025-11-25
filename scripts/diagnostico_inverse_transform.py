"""
Script de Diagnóstico - Inverse Transform
==========================================

Este script diagnostica problemas com o inverse_transform que podem
estar causando o R² negativo.

Uso:
    python scripts/diagnostico_inverse_transform.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from src.data.preprocessor import TimeSeriesPreprocessor

print("="*60)
print("DIAGNOSTICO: INVERSE TRANSFORM")
print("="*60)

# Carregar scaler
scaler_path = "models/scaler.pkl"
if not Path(scaler_path).exists():
    print(f"[ERRO] Scaler nao encontrado: {scaler_path}")
    print("   Execute o treinamento primeiro!")
    sys.exit(1)

print(f"\nCarregando scaler de: {scaler_path}")
scaler_data = joblib.load(scaler_path)

if isinstance(scaler_data, dict):
    scaler = scaler_data['scaler']
    feature_names = scaler_data.get('feature_names', [])
    target_column = scaler_data.get('target_column', 'Close')
    target_idx = scaler_data.get('target_idx')
    
    print(f"[OK] Scaler carregado (formato novo)")
    print(f"   Features: {len(feature_names)}")
    print(f"   Feature names: {feature_names}")
    print(f"   Target column: {target_column}")
    print(f"   Target index: {target_idx}")
    
    # Verificar se target_idx corresponde à coluna
    if feature_names and target_idx is not None:
        if target_idx < len(feature_names):
            coluna_no_indice = feature_names[target_idx]
            print(f"\nVERIFICACAO:")
            print(f"   Indice {target_idx} -> Coluna: '{coluna_no_indice}'")
            if coluna_no_indice == target_column:
                print(f"   [OK] CORRETO: Indice corresponde a coluna target")
            else:
                print(f"   [ERRO] Indice {target_idx} aponta para '{coluna_no_indice}', nao '{target_column}'!")
        else:
            print(f"   [ERRO] target_idx ({target_idx}) >= numero de features ({len(feature_names)})")
else:
    print("[ERRO] Scaler em formato antigo (sem metadados)")
    sys.exit(1)

# Testar inverse transform
print(f"\n" + "="*60)
print("TESTE DE INVERSE TRANSFORM")
print("="*60)

# Criar valores de teste normalizados (0-1)
test_values_scaled = np.array([0.1, 0.5, 0.9])

print(f"\nValores normalizados (entrada): {test_values_scaled}")

# Método 1: Usando target_idx salvo
n_features = scaler.n_features_in_
dummy = np.zeros((len(test_values_scaled), n_features))
dummy[:, target_idx] = test_values_scaled

inversed = scaler.inverse_transform(dummy)
result_method1 = inversed[:, target_idx]

print(f"\n[OK] Metodo 1 (usando target_idx={target_idx}):")
print(f"   Resultado: {result_method1}")

# Método 2: Tentar encontrar Close manualmente
if 'Close' in feature_names:
    close_idx_manual = feature_names.index('Close')
    dummy2 = np.zeros((len(test_values_scaled), n_features))
    dummy2[:, close_idx_manual] = test_values_scaled
    inversed2 = scaler.inverse_transform(dummy2)
    result_method2 = inversed2[:, close_idx_manual]
    
    print(f"\n[OK] Metodo 2 (buscando 'Close' manualmente, indice={close_idx_manual}):")
    print(f"   Resultado: {result_method2}")
    
    if close_idx_manual != target_idx:
        print(f"\n[AVISO] DIFERENCA: target_idx salvo ({target_idx}) != indice encontrado ({close_idx_manual})")
        print(f"   Diferenca nos resultados: {np.abs(result_method1 - result_method2).mean():.2f}")

# Verificar valores mínimos e máximos do scaler para Close
print(f"\n" + "="*60)
print("ESTATISTICAS DO SCALER")
print("="*60)

# Obter min e max do scaler para a coluna Close
if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
    min_val = scaler.data_min_[target_idx]
    max_val = scaler.data_max_[target_idx]
    print(f"\nColuna '{target_column}' (indice {target_idx}):")
    print(f"   Min (normalizado): {min_val:.6f}")
    print(f"   Max (normalizado): {max_val:.6f}")
    print(f"   Range original: {min_val:.2f} a {max_val:.2f}")

# Testar com valores reais do dataset
print(f"\n" + "="*60)
print("TESTE COM VALORES REAIS")
print("="*60)

# Carregar features processadas
features_path = Path("data/processed/features.csv")
if features_path.exists():
    df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
    print(f"\nDataset carregado: {df_features.shape}")
    print(f"   Colunas: {list(df_features.columns)}")
    
    # Pegar alguns valores reais de Close
    close_values = df_features['Close'].iloc[:5].values
    print(f"\nValores reais de Close (primeiros 5):")
    print(f"   {close_values}")
    
    # Normalizar esses valores
    close_scaled = scaler.transform(df_features.iloc[:5].values)[:, target_idx]
    print(f"\nValores normalizados de Close:")
    print(f"   {close_scaled}")
    
    # Fazer inverse transform
    dummy_real = np.zeros((5, n_features))
    dummy_real[:, target_idx] = close_scaled
    inversed_real = scaler.inverse_transform(dummy_real)[:, target_idx]
    
    print(f"\nValores apos inverse transform:")
    print(f"   {inversed_real}")
    
    # Comparar
    diff = np.abs(close_values - inversed_real)
    print(f"\nDiferenca (erro):")
    print(f"   {diff}")
    print(f"   Erro medio: {diff.mean():.2f}")
    print(f"   Erro maximo: {diff.max():.2f}")
    
    if diff.max() > 1.0:
        print(f"\n[ERRO CRITICO] Inverse transform nao esta funcionando corretamente!")
        print(f"   O erro maximo ({diff.max():.2f}) e muito alto.")
    else:
        print(f"\n[OK] Inverse transform funcionando corretamente!")
else:
    print(f"[AVISO] Features nao encontradas em: {features_path}")

print("\n" + "="*60)
print("[OK] DIAGNOSTICO CONCLUIDO")
print("="*60)

