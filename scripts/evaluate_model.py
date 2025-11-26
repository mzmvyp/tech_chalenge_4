"""
Script de Avaliação do Modelo
==============================

Este script carrega um modelo já treinado e avalia sua performance
em novos dados ou re-avalia nos dados de teste.

Uso:
    python scripts/evaluate_model.py

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
from src.models.predictor import StockPredictor
from src.data.preprocessor import TimeSeriesPreprocessor
from src.evaluation.metrics import calculate_all_metrics, compare_with_baselines
from src.evaluation.visualizations import ModelVisualizer


def main():
    """Pipeline de avaliação."""

    print("\n" + "="*60)
    print("📊 AVALIANDO MODELO TREINADO")
    print("="*60)

    # Carregar configurações
    config = get_config()

    # Carregar dados processados
    data_config = config.get_data_config()
    features_path = Path(data_config['processed_data_path']) / 'features.csv'

    if not features_path.exists():
        print(f"❌ Erro: Features não encontradas em {features_path}")
        print("   Execute o treinamento primeiro: python scripts/train_model_stationary.py")
        return

    print(f"\n📂 Carregando features de: {features_path}")
    df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)

    # Preparar dados
    print("\n🔧 Preparando dados...")
    preprocessor = TimeSeriesPreprocessor(
        sequence_length=config.get('model', 'sequence_length'),
        train_ratio=data_config['train_ratio'],
        val_ratio=data_config['val_ratio'],
        test_ratio=data_config['test_ratio']
    )

    data = preprocessor.prepare_data(df_features, target_column='Close', verbose=False)

    # Carregar scaler
    preprocessor.load_scaler(config.get('model_paths', 'scaler_file'))

    # Carregar modelo
    print("\n📂 Carregando modelo...")
    predictor = StockPredictor(
        model_path=config.get('model_paths', 'model_file'),
        scaler_path=config.get('model_paths', 'scaler_file')
    )
    predictor.load_all()

    # Avaliar no conjunto de teste
    print("\n📊 Avaliando no conjunto de TESTE...")

    y_test_pred_scaled = predictor.predict_batch(data['X_test'], return_scaled=True)
    y_test_pred = preprocessor.inverse_transform_target(y_test_pred_scaled, 'Close')
    y_test_true = preprocessor.inverse_transform_target(data['y_test'], 'Close')

    # Métricas
    test_metrics = calculate_all_metrics(y_test_true, y_test_pred, verbose=True)

    # Comparação com baselines
    test_comparison = compare_with_baselines(
        y_test_true,
        y_test_pred,
        model_name="LSTM",
        verbose=True
    )

    # Criar visualizações
    print("\n📊 Criando visualizações...")
    viz = ModelVisualizer(save_dir="outputs/evaluation")

    viz.create_comprehensive_report(
        y_test_true,
        y_test_pred,
        metrics_comparison=test_comparison,
        dataset_name="test_evaluation"
    )

    print("\n✅ Avaliação concluída!")
    print(f"   Figuras salvas em: outputs/evaluation/")


if __name__ == "__main__":
    main()
