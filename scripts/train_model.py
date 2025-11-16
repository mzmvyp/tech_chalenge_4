"""
Script de Treinamento do Modelo LSTM
=====================================

Este script executa todo o pipeline de treinamento:
1. Carrega configurações
2. Baixa dados
3. Cria features
4. Preprocessa com proteção anti-leakage
5. Treina modelo LSTM
6. Avalia performance
7. Salva modelo e resultados

Uso:
    python scripts/train_model.py

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import sys
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime

# Imports do projeto
from src.config import get_config
from src.data.data_loader import DataLoader
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.lstm_model import create_model_from_config
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import calculate_all_metrics, compare_with_baselines
from src.evaluation.visualizations import ModelVisualizer
from src.validation.anti_leakage_tests import AntiLeakageValidator


def main():
    """Pipeline principal de treinamento."""

    print("\n" + "="*60)
    print("🚀 INICIANDO PIPELINE DE TREINAMENTO")
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
    loader.save_raw_data(df_main, df_vix, output_dir=data_config['raw_data_path'])

    # ============================================
    # 3. FEATURE ENGINEERING
    # ============================================
    print("\n🔨 PASSO 3: Criando features...")

    fe = FeatureEngineer()
    features_config = config.get('features')

    df_features = fe.create_all_features(
        df_main=df_main,
        df_vix=df_vix,
        use_moving_averages=features_config.get('use_moving_averages', False),
        use_volume_features=True,
        use_volatility=True,
        use_momentum=True,
        use_returns=features_config.get('use_returns', True)
    )

    # Salvar features
    features_path = Path(data_config['processed_data_path']) / 'features.csv'
    df_features.to_csv(features_path)
    print(f"\n💾 Features salvas em: {features_path}")

    # ============================================
    # 4. PREPROCESSAMENTO (COM ANTI-LEAKAGE!)
    # ============================================
    print("\n🔧 PASSO 4: Preprocessando dados...")

    preprocessor = TimeSeriesPreprocessor(
        sequence_length=model_config['sequence_length'],
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio']
    )

    # Preparar dados (split temporal, normalização, sequências)
    data = preprocessor.prepare_data(df_features, target_column='Close', verbose=True)

    # Salvar scaler
    preprocessor.save_scaler(config.get('model_paths', 'scaler_file'))

    # ============================================
    # 5. CRIAR MODELO
    # ============================================
    print("\n🧠 PASSO 5: Criando modelo LSTM...")

    n_features = data['X_train'].shape[2]
    lstm_model = create_model_from_config(config.config, n_features)

    print(f"\n📊 Resumo do Modelo:")
    print(lstm_model.get_model_summary())

    # ============================================
    # 6. TREINAR MODELO
    # ============================================
    print("\n🏋️  PASSO 6: Treinando modelo...")

    trainer = ModelTrainer(lstm_model.model, config.config)

    history = trainer.train(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        verbose=1
    )

    # Salvar modelo
    lstm_model.save_model(config.get('model_paths', 'model_file'))

    # Salvar informações de treinamento
    trainer.save_training_info(config.get('model_paths', 'metadata_file'))

    # ============================================
    # 7. AVALIAR NO CONJUNTO DE VALIDAÇÃO
    # ============================================
    print("\n📊 PASSO 7: Avaliando no conjunto de VALIDAÇÃO...")

    # Predições no validation set
    y_val_pred_scaled = lstm_model.predict(data['X_val'])
    y_val_pred = preprocessor.inverse_transform_target(y_val_pred_scaled, 'Close')
    y_val_true = preprocessor.inverse_transform_target(data['y_val'], 'Close')

    # Calcular métricas
    val_metrics = calculate_all_metrics(y_val_true, y_val_pred, verbose=True)

    # Comparar com baselines
    val_comparison = compare_with_baselines(
        y_val_true,
        y_val_pred,
        model_name="LSTM",
        verbose=True
    )

    # ============================================
    # 8. AVALIAR NO CONJUNTO DE TESTE
    # ============================================
    print("\n📊 PASSO 8: Avaliando no conjunto de TESTE...")

    # Predições no test set
    y_test_pred_scaled = lstm_model.predict(data['X_test'])
    y_test_pred = preprocessor.inverse_transform_target(y_test_pred_scaled, 'Close')
    y_test_true = preprocessor.inverse_transform_target(data['y_test'], 'Close')

    # Calcular métricas
    test_metrics = calculate_all_metrics(y_test_true, y_test_pred, verbose=True)

    # Comparar com baselines
    test_comparison = compare_with_baselines(
        y_test_true,
        y_test_pred,
        model_name="LSTM",
        verbose=True
    )

    # ============================================
    # 9. TESTES ANTI-LEAKAGE
    # ============================================
    print("\n🔍 PASSO 9: Executando testes anti-leakage...")

    validator = AntiLeakageValidator(strict_mode=False)

    # Obter datas dos splits
    split_info = data['split_info']
    train_dates = pd.date_range(
        split_info['train_period'][0],
        split_info['train_period'][1],
        periods=split_info['train_samples']
    )
    val_dates = pd.date_range(
        split_info['val_period'][0],
        split_info['val_period'][1],
        periods=split_info['val_samples']
    )
    test_dates = pd.date_range(
        split_info['test_period'][0],
        split_info['test_period'][1],
        periods=split_info['test_samples']
    )

    # Executar todos os testes
    all_passed = validator.run_all_tests(
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        train_indices=np.arange(split_info['train_samples']),
        val_indices=np.arange(split_info['train_samples'],
                             split_info['train_samples'] + split_info['val_samples']),
        test_indices=np.arange(split_info['train_samples'] + split_info['val_samples'],
                              split_info['total_samples']),
        model_metrics=test_metrics,
        baseline_metrics=test_comparison['Naive']
    )

    if not all_passed:
        print("\n⚠️  ALERTA: Alguns testes anti-leakage falharam!")
        print("   Revise o pipeline antes de usar o modelo em produção.")

    # ============================================
    # 10. CRIAR VISUALIZAÇÕES
    # ============================================
    print("\n📊 PASSO 10: Criando visualizações...")

    viz = ModelVisualizer(save_dir="outputs/figures")

    # Histórico de treinamento
    viz.plot_training_history(history.history, "training_history.png")

    # Validação
    viz.create_comprehensive_report(
        y_val_true,
        y_val_pred,
        history=history.history,
        metrics_comparison=val_comparison,
        dataset_name="validation"
    )

    # Teste
    viz.create_comprehensive_report(
        y_test_true,
        y_test_pred,
        metrics_comparison=test_comparison,
        dataset_name="test"
    )

    # ============================================
    # 11. RESUMO FINAL
    # ============================================
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
    print(f"   Figuras:    outputs/figures/")

    print(f"\n⏰ Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
