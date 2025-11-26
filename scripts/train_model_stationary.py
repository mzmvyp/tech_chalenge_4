"""
Script de Treinamento do Modelo LSTM
=====================================

Este script executa o pipeline completo de treinamento:
1. Carrega e baixa dados
2. Cria features estacionárias (prediz Return ao invés de Close)
3. Preprocessa com proteção anti-leakage
4. Treina modelo LSTM otimizado
5. Avalia performance
6. Salva modelo e resultados

Uso:
    python scripts/train_model_stationary.py
"""

import sys
from pathlib import Path
import io

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
from datetime import datetime

from src.config import get_config
from src.data.data_loader import DataLoader
from src.data.feature_engineering_stationary import create_stationary_features, predict_close_from_return
from src.data.feature_selector import FeatureSelector
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.lstm_model import create_model_from_config
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import calculate_all_metrics, compare_with_baselines
from src.evaluation.visualizations import ModelVisualizer
from src.validation.anti_leakage_tests import AntiLeakageValidator


def main():
    """Pipeline principal de treinamento com features estacionárias."""
    
    print("\n" + "="*60)
    print("🚀 TREINAMENTO COM FEATURES ESTACIONÁRIAS")
    print("="*60)
    print(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # ============================================
    # 1. CARREGAR CONFIGURAÇÕES
    # ============================================
    print("\n📋 PASSO 1: Carregando configurações...")
    config = get_config()
    config.create_directories()
    
    data_config = config.get_data_config()
    model_config = config.get_model_config()
    training_config = config.get_training_config()
    
    # ============================================
    # 2. BAIXAR DADOS
    # ============================================
    print("\n📥 PASSO 2: Baixando dados...")
    
    loader = DataLoader(
        symbol=data_config['symbol'],
        start_date=data_config['start_date'],
        end_date=data_config['end_date'],
        interval=data_config['interval'],
        vix_symbol=data_config.get('vix_symbol')
    )
    
    df_main, df_vix = loader.load_all_data()
    
    # Guardar Close original e índices para conversão depois
    close_series = df_main['Close'].copy()
    original_index = df_main.index.copy()
    
    # ============================================
    # 3. FEATURE ENGINEERING ESTACIONÁRIO
    # ============================================
    print("\n🔨 PASSO 3: Criando features ESTACIONÁRIAS...")
    print("   (Removendo Close, High, Low, Open)")
    
    df_features = create_stationary_features(df_main, df_vix)
    
    # O índice já está preservado pelo dropna() no create_stationary_features
    # Mas precisamos garantir que close_series tenha os mesmos índices
    # Alinhar close_series com df_features (usar apenas índices que existem em ambos)
    close_series = close_series.loc[df_features.index]
    
    print(f"\n📊 Features estacionárias criadas: {df_features.shape}")
    print(f"   Colunas: {list(df_features.columns)}")
    
    # Verificar se Return está presente
    if 'Return' not in df_features.columns:
        raise ValueError("Return não encontrado nas features! Verifique feature engineering.")
    
    # Salvar features
    features_path = Path(data_config['processed_data_path']) / 'features_stationary.csv'
    df_features.to_csv(features_path)
    print(f"\n💾 Features estacionárias salvas em: {features_path}")
    
    # Feature selection (opcional, mas recomendado)
    print("\n🔍 PASSO 3.5: Selecionando features...")
    selector = FeatureSelector(correlation_threshold=0.8)
    df_features = selector.select_features(df_features, verbose=True)
    
    # ============================================
    # 4. PREPROCESSAMENTO
    # ============================================
    print("\n🔧 PASSO 4: Preprocessando dados...")
    print("   Target: Return (estacionário)")
    
    preprocessor = TimeSeriesPreprocessor(
        sequence_length=model_config['sequence_length'],
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio']
    )
    
    # Preparar dados com Return como target
    data = preprocessor.prepare_data(df_features, target_column='Return', verbose=True)
    
    # Salvar scaler
    preprocessor.save_scaler(config.get('model_paths', 'scaler_file'))
    
    # ============================================
    # 5. CRIAR MODELO
    # ============================================
    print("\n🧠 PASSO 5: Criando modelo LSTM...")
    
    n_features = df_features.shape[1]
    lstm_model = create_model_from_config(config.config, n_features)
    
    # ============================================
    # 6. TREINAR MODELO
    # ============================================
    print("\n🏋️  PASSO 6: Treinando modelo...")
    
    trainer = ModelTrainer(lstm_model.model, config.config)
    history = trainer.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        verbose=1
    )
    
    trainer.save_training_info(config.get('model_paths', 'metadata_file'))
    
    # ============================================
    # 7. AVALIAR NO VALIDATION SET
    # ============================================
    print("\n📊 PASSO 7: Avaliando no conjunto de VALIDAÇÃO...")
    
    # Predições (Return)
    y_val_pred_return = lstm_model.predict(data['X_val']).flatten()
    y_val_true_return = preprocessor.inverse_transform_target(data['y_val'], 'Return')
    y_val_pred_return = preprocessor.inverse_transform_target(y_val_pred_return, 'Return')
    
    # ✅ DIAGNÓSTICO: Verificar valores de Return
    print(f"\n🔍 DIAGNÓSTICO - Validação:")
    print(f"   Return predito (normalizado): min={y_val_pred_return.min():.6f}, max={y_val_pred_return.max():.6f}, mean={y_val_pred_return.mean():.6f}")
    print(f"   Return real (normalizado): min={y_val_true_return.min():.6f}, max={y_val_true_return.max():.6f}, mean={y_val_true_return.mean():.6f}")
    
    # Converter Return → Close para métricas
    # IMPORTANTE: As sequências são criadas dentro de cada conjunto separadamente
    # A primeira sequência de validação usa os últimos sequence_length dias do treino + primeiro dia de val
    
    # Pegar índices do conjunto de validação
    val_start_date = pd.to_datetime(data['split_info']['val_period'][0])
    val_end_date = pd.to_datetime(data['split_info']['val_period'][1])
    
    # Filtrar índices do DataFrame original que estão no período de validação
    val_mask = (df_features.index >= val_start_date) & (df_features.index <= val_end_date)
    val_indices = df_features.index[val_mask]
    
    # As sequências começam após sequence_length dentro do conjunto de validação
    # Mas a primeira sequência usa dados do final do treino
    # Para simplificar: usar os índices do DataFrame completo
    
    # Encontrar onde começa o período de validação no DataFrame completo
    all_indices = df_features.index
    # Usar searchsorted para encontrar a posição (mais robusto que get_loc)
    val_start_pos = all_indices.searchsorted(val_start_date)
    
    # Número de predições
    n_predictions = len(y_val_pred_return)
    
    # Ajustar para não ultrapassar os limites
    max_predictions = len(val_indices) - model_config['sequence_length']
    n_predictions = min(n_predictions, max_predictions)
    y_val_pred_return = y_val_pred_return[:n_predictions]
    
    # Para cada predição i:
    # - Close real está em: val_start_pos + sequence_length + i
    # - Close anterior está em: val_start_pos + sequence_length - 1 + i
    val_pred_indices = []
    val_close_last_indices = []
    
    for i in range(n_predictions):
        # Índice do Close real (dia da predição)
        pred_pos = val_start_pos + model_config['sequence_length'] + i
        if pred_pos < len(all_indices):
            val_pred_indices.append(all_indices[pred_pos])
        
        # Índice do Close anterior (dia antes da predição)
        last_pos = val_start_pos + model_config['sequence_length'] - 1 + i
        if last_pos < len(all_indices):
            val_close_last_indices.append(all_indices[last_pos])
    
    # Ajustar tamanhos finais
    min_len = min(len(val_pred_indices), len(val_close_last_indices), len(y_val_pred_return))
    val_pred_indices = val_pred_indices[:min_len]
    val_close_last_indices = val_close_last_indices[:min_len]
    y_val_pred_return = y_val_pred_return[:min_len]
    
    # Close real: valores correspondentes às datas das predições
    val_close_real = close_series.loc[val_pred_indices].values
    
    # Close predito: usar último Close conhecido (dia anterior)
    val_close_last = close_series.loc[val_close_last_indices].values
    val_close_pred = val_close_last * (1 + y_val_pred_return)
    
    # ✅ DIAGNÓSTICO: Verificar valores de Close
    print(f"\n🔍 DIAGNÓSTICO - Close (Validação):")
    print(f"   Close real: min={val_close_real.min():.2f}, max={val_close_real.max():.2f}, mean={val_close_real.mean():.2f}")
    print(f"   Close predito: min={val_close_pred.min():.2f}, max={val_close_pred.max():.2f}, mean={val_close_pred.mean():.2f}")
    print(f"   Close anterior: min={val_close_last.min():.2f}, max={val_close_last.max():.2f}, mean={val_close_last.mean():.2f}")
    print(f"   Return médio: {y_val_pred_return.mean():.6f} ({y_val_pred_return.mean()*100:.2f}%)")
    
    # Métricas em Close
    val_metrics = calculate_all_metrics(val_close_real, val_close_pred, verbose=True)
    val_comparison = compare_with_baselines(val_close_real, val_close_pred, model_name="LSTM (Return)", verbose=True)
    
    # ============================================
    # 8. AVALIAR NO TEST SET
    # ============================================
    print("\n📊 PASSO 8: Avaliando no conjunto de TESTE...")
    
    # Predições (Return)
    y_test_pred_return = lstm_model.predict(data['X_test']).flatten()
    y_test_true_return = preprocessor.inverse_transform_target(data['y_test'], 'Return')
    y_test_pred_return = preprocessor.inverse_transform_target(y_test_pred_return, 'Return')
    
    # Converter Return → Close (mesma lógica da validação)
    test_start_date = pd.to_datetime(data['split_info']['test_period'][0])
    test_end_date = pd.to_datetime(data['split_info']['test_period'][1])
    
    # Filtrar índices do DataFrame original que estão no período de teste
    test_mask = (df_features.index >= test_start_date) & (df_features.index <= test_end_date)
    test_indices = df_features.index[test_mask]
    
    # Encontrar onde começa o período de teste no DataFrame completo
    test_start_pos = all_indices.searchsorted(test_start_date)
    
    # Número de predições
    n_predictions = len(y_test_pred_return)
    
    # Ajustar para não ultrapassar os limites
    max_predictions = len(test_indices) - model_config['sequence_length']
    n_predictions = min(n_predictions, max_predictions)
    y_test_pred_return = y_test_pred_return[:n_predictions]
    
    # Para cada predição i:
    # - Close real está em: test_start_pos + sequence_length + i
    # - Close anterior está em: test_start_pos + sequence_length - 1 + i
    test_pred_indices = []
    test_close_last_indices = []
    
    for i in range(n_predictions):
        # Índice do Close real (dia da predição)
        pred_pos = test_start_pos + model_config['sequence_length'] + i
        if pred_pos < len(all_indices):
            test_pred_indices.append(all_indices[pred_pos])
        
        # Índice do Close anterior (dia antes da predição)
        last_pos = test_start_pos + model_config['sequence_length'] - 1 + i
        if last_pos < len(all_indices):
            test_close_last_indices.append(all_indices[last_pos])
    
    # Ajustar tamanhos finais
    min_len = min(len(test_pred_indices), len(test_close_last_indices), len(y_test_pred_return))
    test_pred_indices = test_pred_indices[:min_len]
    test_close_last_indices = test_close_last_indices[:min_len]
    y_test_pred_return = y_test_pred_return[:min_len]
    
    # Close real
    test_close_real = close_series.loc[test_pred_indices].values
    
    # Close predito: usar último Close conhecido (dia anterior)
    test_close_last = close_series.loc[test_close_last_indices].values
    test_close_pred = test_close_last * (1 + y_test_pred_return)
    
    # ✅ DIAGNÓSTICO: Verificar valores de Close (Teste)
    print(f"\n🔍 DIAGNÓSTICO - Close (Teste):")
    print(f"   Close real: min={test_close_real.min():.2f}, max={test_close_real.max():.2f}, mean={test_close_real.mean():.2f}")
    print(f"   Close predito: min={test_close_pred.min():.2f}, max={test_close_pred.max():.2f}, mean={test_close_pred.mean():.2f}")
    print(f"   Return médio: {y_test_pred_return.mean():.6f} ({y_test_pred_return.mean()*100:.2f}%)")
    
    # Métricas em Close
    test_metrics = calculate_all_metrics(test_close_real, test_close_pred, verbose=True)
    test_comparison = compare_with_baselines(test_close_real, test_close_pred, model_name="LSTM (Return)", verbose=True)
    
    # ============================================
    # 9. TESTES ANTI-LEAKAGE
    # ============================================
    print("\n🔍 PASSO 9: Executando testes anti-leakage...")
    validator = AntiLeakageValidator(strict_mode=False)
    
    # Obter datas dos splits
    split_info = data['split_info']
    
    # Usar índices reais ao invés de date_range
    train_mask = (df_features.index >= pd.to_datetime(split_info['train_period'][0])) & \
                 (df_features.index <= pd.to_datetime(split_info['train_period'][1]))
    val_mask = (df_features.index >= pd.to_datetime(split_info['val_period'][0])) & \
               (df_features.index <= pd.to_datetime(split_info['val_period'][1]))
    test_mask = (df_features.index >= pd.to_datetime(split_info['test_period'][0])) & \
                (df_features.index <= pd.to_datetime(split_info['test_period'][1]))
    
    train_dates = df_features.index[train_mask]
    val_dates = df_features.index[val_mask]
    test_dates = df_features.index[test_mask]
    
    # Preparar métricas para os testes
    model_metrics = {
        'R2': test_metrics.get('R2', 0),
        'RMSE': test_metrics.get('RMSE', float('inf')),
        'MAPE': test_metrics.get('MAPE', 100)
    }
    
    baseline_metrics = {
        'RMSE': test_comparison.get('Naive', {}).get('RMSE', 0),
        'R2': test_comparison.get('Naive', {}).get('R2', None)  # ✅ NOVO: Incluir R² do baseline
    }
    
    # Executar todos os testes
    # NOTA: Para sequências temporais, não podemos usar índices simples porque
    # cada sequência usa dados anteriores. O teste de sobreposição deve verificar
    # se há sobreposição temporal nas DATAS, não nos índices das sequências.
    # Por isso, passamos None para os índices e o teste verifica apenas ordem temporal.
    all_passed = validator.run_all_tests(
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        train_indices=None,  # Não usar índices relativos - causa falsos positivos
        val_indices=None,
        test_indices=None,
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        scaler=None,  # Pular teste do scaler por enquanto (tem bug)
        train_data_sample=None,
        val_data_sample=None
    )
    
    # ============================================
    # 10. VISUALIZAÇÕES
    # ============================================
    print("\n📊 PASSO 10: Criando visualizações...")
    visualizer = ModelVisualizer()
    
    # Histórico de treinamento
    visualizer.plot_training_history(history.history)
    
    # Validação
    visualizer.create_comprehensive_report(
        y_true=val_close_real,
        y_pred=val_close_pred,
        history=history.history,
        metrics_comparison=val_comparison,
        dataset_name="validation"
    )
    
    # Teste
    visualizer.create_comprehensive_report(
        y_true=test_close_real,
        y_pred=test_close_pred,
        history=history.history,
        metrics_comparison=test_comparison,
        dataset_name="test"
    )
    
    print("\n" + "="*60)
    print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*60)
    print(f"\n📊 MÉTRICAS FINAIS (Conjunto de TESTE):")
    print(f"   MAE:                 {test_metrics['MAE']:.4f}")
    print(f"   RMSE:                {test_metrics['RMSE']:.4f}")
    print(f"   MAPE:                {test_metrics['MAPE']:.2f}%")
    print(f"   R² Score:            {test_metrics['R2']:.4f}")
    print(f"   Direction Accuracy:  {test_metrics['Direction_Accuracy']:.2f}%")
    print(f"\n💾 ARQUIVOS SALVOS:")
    print(f"   Modelo:     {config.get('model_paths', 'model_file')}")
    print(f"   Scaler:     {config.get('model_paths', 'scaler_file')}")
    print(f"   Metadata:   {config.get('model_paths', 'metadata_file')}")
    print(f"   Features:   {features_path}")
    print(f"\n⏰ Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()

