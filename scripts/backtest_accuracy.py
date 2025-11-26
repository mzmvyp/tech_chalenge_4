"""
Script de Backtest - Cálculo de Accuracy
========================================

Executa 1000 testes no modelo e calcula:
- Direction Accuracy (acurácia de direção)
- MAE, RMSE, MAPE, R²
- Taxa de acerto geral
- Análise detalhada de erros

Uso:
    python scripts/backtest_accuracy.py
    python scripts/backtest_accuracy.py --n-tests 1000
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
import joblib
import tensorflow as tf

from src.config import get_config
from src.data.data_loader import DataLoader
from src.data.feature_engineering_stationary import create_stationary_features
from src.evaluation.metrics import calculate_all_metrics


def calculate_accuracy_metrics(
    actuals: np.ndarray,
    predictions: np.ndarray,
    previous_values: np.ndarray
) -> dict:
    """
    Calcula métricas de accuracy detalhadas.
    
    Args:
        actuals: Valores reais
        predictions: Valores preditos
        previous_values: Valores anteriores (para direction accuracy)
    
    Returns:
        Dicionário com métricas de accuracy
    """
    # Direction Accuracy
    actual_direction = np.sign(actuals - previous_values)
    pred_direction = np.sign(predictions - previous_values)
    direction_correct = (actual_direction == pred_direction).sum()
    direction_accuracy = (direction_correct / len(actuals)) * 100
    
    # Accuracy por faixa de erro
    errors = np.abs(actuals - predictions)
    error_pct = (errors / actuals) * 100
    
    # Taxa de acerto por faixas
    accuracy_ranges = {
        'within_0.1%': (error_pct <= 0.1).sum() / len(error_pct) * 100,
        'within_0.5%': (error_pct <= 0.5).sum() / len(error_pct) * 100,
        'within_1%': (error_pct <= 1.0).sum() / len(error_pct) * 100,
        'within_2%': (error_pct <= 2.0).sum() / len(error_pct) * 100,
        'within_5%': (error_pct <= 5.0).sum() / len(error_pct) * 100,
    }
    
    # Accuracy de magnitude (se erro absoluto está dentro de X pontos)
    mean_price = np.mean(actuals)
    accuracy_absolute = {
        'within_10_points': (errors <= 10).sum() / len(errors) * 100,
        'within_25_points': (errors <= 25).sum() / len(errors) * 100,
        'within_50_points': (errors <= 50).sum() / len(errors) * 100,
        'within_100_points': (errors <= 100).sum() / len(errors) * 100,
    }
    
    return {
        'direction_accuracy': direction_accuracy,
        'direction_correct': int(direction_correct),
        'direction_total': len(actuals),
        'accuracy_by_percentage': accuracy_ranges,
        'accuracy_by_absolute': accuracy_absolute,
        'mean_error_pct': float(np.mean(error_pct)),
        'median_error_pct': float(np.median(error_pct)),
    }


def run_backtest_accuracy(
    n_tests: int = 1000,
    model_path: str = "models/lstm_model.h5",
    scaler_path: str = "models/scaler.pkl"
) -> dict:
    """
    Executa backtest focado em accuracy.
    
    Args:
        n_tests: Número de testes (padrão: 1000)
        model_path: Caminho do modelo
        scaler_path: Caminho do scaler
    
    Returns:
        Dicionário com resultados
    """
    print("\n" + "="*60)
    print("🎯 BACKTEST - CÁLCULO DE ACCURACY")
    print("="*60)
    print(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Número de testes: {n_tests}")
    print("="*60)
    
    # ============================================
    # 1. CARREGAR DADOS E MODELO
    # ============================================
    print("\n📥 PASSO 1: Carregando dados e modelo...")
    
    config = get_config()
    data_config = config.get_data_config()
    model_config = config.get_model_config()
    
    # Carregar dados
    loader = DataLoader(
        symbol=data_config['symbol'],
        start_date=data_config['start_date'],
        end_date=data_config['end_date'],
        interval=data_config['interval'],
        vix_symbol=data_config.get('vix_symbol')
    )
    
    df_main, df_vix = loader.load_all_data()
    close_series = df_main['Close'].copy()
    
    # Criar features
    print("🔨 Criando features estacionárias...")
    df_features = create_stationary_features(df_main, df_vix)
    close_series = close_series.loc[df_features.index]
    
    # Carregar modelo e scaler
    print("🤖 Carregando modelo...")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    scaler_data = joblib.load(scaler_path)
    if isinstance(scaler_data, dict):
        scaler = scaler_data['scaler']
        feature_names = scaler_data.get('feature_names', df_features.columns.tolist())
        target_idx = scaler_data.get('target_idx', 0)
    else:
        scaler = scaler_data
        feature_names = df_features.columns.tolist()
        target_idx = 0
    
    # Garantir ordem correta das features
    if feature_names:
        available_features = [f for f in feature_names if f in df_features.columns]
        df_features = df_features[available_features]
    
    sequence_length = model_config['sequence_length']
    
    print(f"✓ Dados: {len(df_features)} registros")
    print(f"✓ Features: {len(df_features.columns)}")
    print(f"✓ Sequence length: {sequence_length}")
    
    # ============================================
    # 2. PREPARAR TESTES
    # ============================================
    print(f"\n📊 PASSO 2: Preparando {n_tests} testes...")
    
    # Usar dados de teste (últimos 15% do dataset)
    test_start_idx = int(len(df_features) * 0.85)
    max_available = len(df_features) - test_start_idx - sequence_length
    n_tests = min(n_tests, max_available)
    
    print(f"   Testes disponíveis: {max_available}")
    print(f"   Testes a executar: {n_tests}")
    
    if n_tests <= 0:
        print("❌ Erro: Nenhum teste disponível!")
        return {}
    
    # ============================================
    # 3. EXECUTAR TESTES
    # ============================================
    print(f"\n🧪 PASSO 3: Executando {n_tests} testes...")
    print("="*60)
    
    predictions = []
    actuals = []
    previous_closes = []
    dates = []
    errors = []
    
    for i in range(n_tests):
        if (i + 1) % 100 == 0:
            print(f"   Progresso: {i+1}/{n_tests} ({(i+1)/n_tests*100:.1f}%)")
        
        # Índice do teste
        idx = test_start_idx + sequence_length + i
        
        if idx >= len(df_features):
            break
        
        # Sequência de entrada
        seq_start = idx - sequence_length
        seq_end = idx
        sequence_features = df_features.iloc[seq_start:seq_end]
        
        # Valores reais
        actual_return = df_features.iloc[idx]['Return'] if 'Return' in df_features.columns else None
        actual_close = close_series.iloc[idx] if idx < len(close_series) else None
        last_close = close_series.iloc[idx-1] if idx > 0 else None
        
        if actual_close is None or last_close is None or actual_return is None:
            continue
        
        # Normalizar sequência
        seq_array = scaler.transform(sequence_features.values)
        seq_array = seq_array.reshape(1, sequence_length, len(feature_names))
        
        # Predição
        pred_scaled = model.predict(seq_array, verbose=0)[0, 0]
        
        # Inverse transform
        dummy = np.zeros((1, scaler.n_features_in_))
        dummy[0, target_idx] = pred_scaled
        pred_return = scaler.inverse_transform(dummy)[0, target_idx]
        
        # Converter Return → Close
        pred_close = last_close * (1 + pred_return)
        
        # Armazenar
        predictions.append(pred_close)
        actuals.append(actual_close)
        previous_closes.append(last_close)
        dates.append(df_features.index[idx])
        errors.append(abs(actual_close - pred_close))
    
    print(f"\n✓ {len(predictions)} testes concluídos")
    
    if len(predictions) == 0:
        print("❌ Erro: Nenhuma predição válida!")
        return {}
    
    # Converter para arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    previous_closes = np.array(previous_closes)
    errors = np.array(errors)
    
    # ============================================
    # 4. CALCULAR MÉTRICAS
    # ============================================
    print("\n📊 PASSO 4: Calculando métricas de accuracy...")
    
    # Métricas gerais
    metrics = calculate_all_metrics(actuals, predictions, previous_values=previous_closes, verbose=False)
    
    # Métricas de accuracy detalhadas
    accuracy_metrics = calculate_accuracy_metrics(actuals, predictions, previous_closes)
    
    # ============================================
    # 5. EXIBIR RESULTADOS
    # ============================================
    print("\n" + "="*60)
    print("📊 RESULTADOS DO BACKTEST - ACCURACY")
    print("="*60)
    
    print(f"\n📈 MÉTRICAS GERAIS ({len(predictions)} testes):")
    print(f"   MAE:                 {metrics['MAE']:.4f}")
    print(f"   RMSE:                {metrics['RMSE']:.4f}")
    print(f"   MAPE:                {metrics['MAPE']:.2f}%")
    print(f"   R² Score:            {metrics['R2']:.4f}")
    
    print(f"\n🎯 ACCURACY DE DIREÇÃO:")
    print(f"   Direction Accuracy:  {accuracy_metrics['direction_accuracy']:.2f}%")
    print(f"   Acertos:              {accuracy_metrics['direction_correct']}/{accuracy_metrics['direction_total']}")
    
    print(f"\n📊 ACCURACY POR ERRO PERCENTUAL:")
    for range_name, accuracy in accuracy_metrics['accuracy_by_percentage'].items():
        print(f"   {range_name.replace('_', ' ').title()}: {accuracy:.2f}%")
    
    print(f"\n📊 ACCURACY POR ERRO ABSOLUTO:")
    for range_name, accuracy in accuracy_metrics['accuracy_by_absolute'].items():
        print(f"   {range_name.replace('_', ' ').title()}: {accuracy:.2f}%")
    
    print(f"\n📉 ESTATÍSTICAS DE ERRO:")
    print(f"   Erro médio (%):      {accuracy_metrics['mean_error_pct']:.2f}%")
    print(f"   Erro mediano (%):    {accuracy_metrics['median_error_pct']:.2f}%")
    print(f"   Erro médio (abs):    {np.mean(errors):.2f} pontos")
    print(f"   Erro mediano (abs):  {np.median(errors):.2f} pontos")
    
    # Análise de distribuição
    print(f"\n📊 DISTRIBUIÇÃO DE ERROS:")
    print(f"   Min erro:            {np.min(errors):.2f} pontos")
    print(f"   Max erro:            {np.max(errors):.2f} pontos")
    print(f"   Desvio padrão:       {np.std(errors):.2f} pontos")
    
    # Percentis
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\n📊 PERCENTIS DE ERRO:")
    for p in percentiles:
        print(f"   P{p}: {np.percentile(errors, p):.2f} pontos")
    
    # ============================================
    # 6. SALVAR RESULTADOS
    # ============================================
    print("\n💾 PASSO 5: Salvando resultados...")
    
    results = {
        'backtest_info': {
            'n_tests': len(predictions),
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': model_path,
            'scaler_path': scaler_path
        },
        'metrics': {k: float(v) for k, v in metrics.items()},
        'accuracy_metrics': accuracy_metrics,
        'error_statistics': {
            'mean': float(np.mean(errors)),
            'median': float(np.median(errors)),
            'std': float(np.std(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
            'percentiles': {f'p{p}': float(np.percentile(errors, p)) for p in percentiles}
        }
    }
    
    # Salvar JSON
    results_path = Path("outputs/backtest_accuracy_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Resultados salvos em: {results_path}")
    
    # Salvar CSV com predições
    predictions_df = pd.DataFrame({
        'date': dates,
        'actual_close': actuals,
        'predicted_close': predictions,
        'previous_close': previous_closes,
        'error_absolute': errors,
        'error_percentage': (errors / actuals) * 100,
        'actual_direction': np.sign(actuals - previous_closes),
        'predicted_direction': np.sign(predictions - previous_closes),
        'direction_correct': (np.sign(actuals - previous_closes) == np.sign(predictions - previous_closes))
    })
    
    csv_path = Path("outputs/backtest_accuracy_predictions.csv")
    predictions_df.to_csv(csv_path, index=False)
    print(f"✓ Predições salvas em: {csv_path}")
    
    print("\n" + "="*60)
    print("✅ BACKTEST CONCLUÍDO!")
    print("="*60)
    
    return results


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest de Accuracy do modelo LSTM')
    parser.add_argument('--n-tests', type=int, default=1000, 
                       help='Número de testes (padrão: 1000)')
    parser.add_argument('--model', type=str, default="models/lstm_model.h5",
                       help='Caminho do modelo')
    parser.add_argument('--scaler', type=str, default="models/scaler.pkl",
                       help='Caminho do scaler')
    
    args = parser.parse_args()
    
    results = run_backtest_accuracy(
        n_tests=args.n_tests,
        model_path=args.model,
        scaler_path=args.scaler
    )
    
    return results


if __name__ == "__main__":
    main()

