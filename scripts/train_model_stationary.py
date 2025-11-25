"""
Script de Treinamento - Versão com Features Estacionárias
==========================================================

Esta versão:
1. Remove features não-estacionárias (Close, High, Low, Open)
2. Prediz Return (estacionário) ao invés de Close
3. Converte Return predito de volta para Close

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
    
    # Manter índice original para alinhamento (após remoção de NaN)
    # O create_stationary_features remove linhas, então precisamos alinhar índices
    if len(df_features) < len(original_index):
        # Pegar índices correspondentes aos dados que sobraram
        # Assumir que as primeiras linhas foram removidas (NaN de rolling windows)
        df_features.index = original_index[len(original_index) - len(df_features):]
    else:
        df_features.index = original_index[:len(df_features)]
    
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
    
    # Converter Return → Close para métricas
    # Usar índices reais do DataFrame (não date_range, pois dados financeiros não têm todos os dias)
    
    # Pegar índices do conjunto de validação
    val_start_date = pd.to_datetime(data['split_info']['val_period'][0])
    val_end_date = pd.to_datetime(data['split_info']['val_period'][1])
    
    # Filtrar índices do DataFrame original que estão no período de validação
    val_mask = (df_features.index >= val_start_date) & (df_features.index <= val_end_date)
    val_indices = df_features.index[val_mask]
    
    # As sequências começam após sequence_length, então pular os primeiros sequence_length índices
    val_pred_indices = val_indices[model_config['sequence_length']:]
    
    # Ajustar tamanho se necessário
    min_len = min(len(val_pred_indices), len(y_val_pred_return))
    val_pred_indices = val_pred_indices[:min_len]
    y_val_pred_return = y_val_pred_return[:min_len]
    
    # Close real: valores correspondentes às datas das predições
    val_close_real = close_series.loc[val_pred_indices].values
    
    # Close predito: usar último Close conhecido (índice anterior a cada predição)
    # Encontrar índices anteriores no DataFrame original
    val_close_last = []
    for idx in val_pred_indices:
        # Encontrar índice anterior no close_series
        prev_idx = close_series.index[close_series.index < idx]
        if len(prev_idx) > 0:
            val_close_last.append(close_series.loc[prev_idx[-1]])
        else:
            # Fallback: usar primeiro valor disponível
            val_close_last.append(close_series.iloc[0])
    
    val_close_last = np.array(val_close_last)
    val_close_pred = val_close_last * (1 + y_val_pred_return)
    
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
    
    # Converter Return → Close
    test_start_date = pd.to_datetime(data['split_info']['test_period'][0])
    test_end_date = pd.to_datetime(data['split_info']['test_period'][1])
    
    # Filtrar índices do DataFrame original que estão no período de teste
    test_mask = (df_features.index >= test_start_date) & (df_features.index <= test_end_date)
    test_indices = df_features.index[test_mask]
    
    # As sequências começam após sequence_length
    test_pred_indices = test_indices[model_config['sequence_length']:]
    
    # Ajustar tamanho se necessário
    min_len = min(len(test_pred_indices), len(y_test_pred_return))
    test_pred_indices = test_pred_indices[:min_len]
    y_test_pred_return = y_test_pred_return[:min_len]
    
    # Close real
    test_close_real = close_series.loc[test_pred_indices].values
    
    # Close predito: usar último Close conhecido
    test_close_last = []
    for idx in test_pred_indices:
        prev_idx = close_series.index[close_series.index < idx]
        if len(prev_idx) > 0:
            test_close_last.append(close_series.loc[prev_idx[-1]])
        else:
            test_close_last.append(close_series.iloc[0])
    
    test_close_last = np.array(test_close_last)
    test_close_pred = test_close_last * (1 + y_test_pred_return)
    
    # Métricas em Close
    test_metrics = calculate_all_metrics(test_close_real, test_close_pred, verbose=True)
    test_comparison = compare_with_baselines(test_close_real, test_close_pred, model_name="LSTM (Return)", verbose=True)
    
    # ============================================
    # 9. TESTES ANTI-LEAKAGE
    # ============================================
    print("\n🔍 PASSO 9: Executando testes anti-leakage...")
    validator = AntiLeakageValidator()
    validator.validate_all(
        df_train=df_features.iloc[:data['split_info']['train_samples']],
        df_val=df_features.iloc[data['split_info']['train_samples']:data['split_info']['train_samples']+data['split_info']['val_samples']],
        df_test=df_features.iloc[data['split_info']['train_samples']+data['split_info']['val_samples']:],
        y_train_pred=None,  # Não necessário para validação básica
        y_val_pred=val_close_pred,
        y_test_pred=test_close_pred,
        y_train_true=None,
        y_val_true=val_close_real,
        y_test_true=test_close_real
    )
    
    # ============================================
    # 10. VISUALIZAÇÕES
    # ============================================
    print("\n📊 PASSO 10: Criando visualizações...")
    visualizer = ModelVisualizer(output_dir="outputs/figures")
    
    # Histórico de treinamento
    visualizer.plot_training_history(history.history)
    
    # Validação
    visualizer.create_evaluation_report(
        y_true=val_close_real,
        y_pred=val_close_pred,
        dataset_name="validation"
    )
    
    # Teste
    visualizer.create_evaluation_report(
        y_true=test_close_real,
        y_pred=test_close_pred,
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

