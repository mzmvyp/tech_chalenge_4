"""
Script de Backtest do Modelo LSTM
==================================

Executa backtest completo com ~1000 testes para:
1. Avaliar performance do modelo
2. Testar sistema de online learning
3. Verificar se modelo melhora ao longo do tempo

Uso:
    python scripts/backtest_model.py
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
from datetime import datetime, timedelta
from typing import List, Dict
import joblib

from src.config import get_config
from src.data.data_loader import DataLoader
from src.data.feature_engineering_stationary import create_stationary_features
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.online_learner import OnlineLearner
from src.evaluation.metrics import calculate_all_metrics


def prepare_backtest_data(
    df_features: pd.DataFrame,
    close_series: pd.Series,
    sequence_length: int,
    start_idx: int = 0,
    n_tests: int = 1000
) -> List[Dict]:
    """
    Prepara dados para backtest.
    
    Args:
        df_features: DataFrame com features estacionárias
        close_series: Série com valores de Close
        sequence_length: Comprimento da sequência
        start_idx: Índice inicial (após sequence_length)
        n_tests: Número de testes a fazer
    
    Returns:
        Lista de dicionários com dados para cada teste
    """
    backtest_data = []
    
    # Limitar número de testes ao disponível
    max_tests = len(df_features) - sequence_length - start_idx
    n_tests = min(n_tests, max_tests)
    
    print(f"\n📊 Preparando {n_tests} testes de backtest...")
    print(f"   Dados disponíveis: {max_tests} testes possíveis")
    
    for i in range(n_tests):
        idx = start_idx + sequence_length + i
        
        if idx >= len(df_features):
            break
        
        # Sequência de entrada (últimos sequence_length dias)
        seq_start = idx - sequence_length
        seq_end = idx
        
        # Features da sequência
        sequence_features = df_features.iloc[seq_start:seq_end]
        
        # Valor real (Return do próximo dia)
        if idx < len(df_features):
            actual_return = df_features.iloc[idx]['Return'] if 'Return' in df_features.columns else None
            actual_close = close_series.iloc[idx] if idx < len(close_series) else None
            last_close = close_series.iloc[idx-1] if idx > 0 else None
        else:
            break
        
        backtest_data.append({
            'index': idx,
            'date': df_features.index[idx] if idx < len(df_features.index) else None,
            'sequence_features': sequence_features,
            'actual_return': actual_return,
            'actual_close': actual_close,
            'last_close': last_close
        })
    
    print(f"✓ {len(backtest_data)} testes preparados")
    return backtest_data


def run_backtest(
    model_path: str = "models/lstm_model.h5",
    scaler_path: str = "models/scaler.pkl",
    n_tests: int = 1000,
    enable_online_learning: bool = True,
    retrain_threshold: int = 50
) -> Dict:
    """
    Executa backtest completo do modelo.
    
    Args:
        model_path: Caminho do modelo
        scaler_path: Caminho do scaler
        n_tests: Número de testes
        enable_online_learning: Se True, ativa online learning
        retrain_threshold: Threshold para retreinar
    
    Returns:
        Dicionário com resultados do backtest
    """
    print("\n" + "="*60)
    print("🧪 BACKTEST DO MODELO LSTM")
    print("="*60)
    print(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Número de testes: {n_tests}")
    print(f"🔄 Online Learning: {'ATIVADO' if enable_online_learning else 'DESATIVADO'}")
    print("="*60)
    
    # ============================================
    # 1. CARREGAR DADOS
    # ============================================
    print("\n📥 PASSO 1: Carregando dados...")
    
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
    
    # Criar features estacionárias
    print("\n🔨 Criando features estacionárias...")
    df_features = create_stationary_features(df_main, df_vix)
    
    # Alinhar close_series
    close_series = close_series.loc[df_features.index]
    
    # Carregar scaler para normalizar sequências
    scaler_data = joblib.load(scaler_path)
    if isinstance(scaler_data, dict):
        scaler = scaler_data['scaler']
        feature_names = scaler_data.get('feature_names', df_features.columns.tolist())
        target_idx = scaler_data.get('target_idx', 0)
    else:
        scaler = scaler_data
        feature_names = df_features.columns.tolist()
        target_idx = 0
    
    # IMPORTANTE: Garantir que df_features tenha apenas as features que o scaler espera
    # E na mesma ordem
    if feature_names:
        # Filtrar apenas features que existem e estão na ordem correta
        available_features = [f for f in feature_names if f in df_features.columns]
        df_features = df_features[available_features]
        
        # Verificar se número de features corresponde
        if len(available_features) != scaler.n_features_in_:
            print(f"⚠️  Aviso: Features disponíveis ({len(available_features)}) != Features do scaler ({scaler.n_features_in_})")
            print(f"   Features disponíveis: {available_features}")
            print(f"   Ajustando para usar apenas features do scaler...")
            
            # Tentar usar apenas as primeiras N features que correspondem
            if len(available_features) > scaler.n_features_in_:
                available_features = available_features[:scaler.n_features_in_]
                df_features = df_features[available_features]
    
    sequence_length = model_config['sequence_length']
    
    print(f"✓ Dados carregados: {len(df_features)} registros")
    print(f"✓ Features: {len(df_features.columns)} (scaler espera {scaler.n_features_in_})")
    print(f"✓ Sequence length: {sequence_length}")
    
    # Validar que número de features corresponde
    if len(df_features.columns) != scaler.n_features_in_:
        raise ValueError(
            f"Incompatibilidade de features: "
            f"DataFrame tem {len(df_features.columns)}, scaler espera {scaler.n_features_in_}"
        )
    
    # ============================================
    # 2. PREPARAR DADOS DE BACKTEST
    # ============================================
    print("\n📊 PASSO 2: Preparando dados de backtest...")
    
    # Usar dados de teste (últimos 15% do dataset)
    test_start_idx = int(len(df_features) * 0.85)  # 85% para treino+val, 15% para teste
    
    backtest_data = prepare_backtest_data(
        df_features=df_features,
        close_series=close_series,
        sequence_length=sequence_length,
        start_idx=test_start_idx - sequence_length,
        n_tests=n_tests
    )
    
    if len(backtest_data) == 0:
        print("❌ Erro: Nenhum dado de backtest preparado!")
        return {}
    
    print(f"✓ {len(backtest_data)} testes preparados")
    
    # ============================================
    # 3. INICIALIZAR MODELO E ONLINE LEARNER
    # ============================================
    print("\n🤖 PASSO 3: Carregando modelo...")
    
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    if enable_online_learning:
        learner = OnlineLearner(
            model_path=model_path,
            scaler_path=scaler_path,
            retrain_threshold=retrain_threshold
        )
        learner.model = model
        learner.scaler = scaler
        learner.feature_names = feature_names
        learner.target_idx = target_idx
        learner.sequence_length = sequence_length
        print("✓ Online Learner inicializado")
    else:
        learner = None
        print("✓ Modelo carregado (sem online learning)")
    
    # ============================================
    # 4. EXECUTAR BACKTEST
    # ============================================
    print("\n🧪 PASSO 4: Executando backtest...")
    print("="*60)
    
    predictions = []
    actuals = []
    dates = []
    errors = []
    retrain_events = []
    
    for i, test_data in enumerate(backtest_data):
        if i % 100 == 0:
            print(f"   Progresso: {i}/{len(backtest_data)} ({i/len(backtest_data)*100:.1f}%)")
        
        # Preparar sequência
        seq_features = test_data['sequence_features']
        
        # Normalizar sequência
        seq_array = scaler.transform(seq_features.values)
        seq_array = seq_array.reshape(1, sequence_length, len(feature_names))
        
        # Fazer predição
        pred_scaled = model.predict(seq_array, verbose=0)[0, 0]
        
        # Inverse transform
        dummy = np.zeros((1, scaler.n_features_in_))
        dummy[0, target_idx] = pred_scaled
        pred_return = scaler.inverse_transform(dummy)[0, target_idx]
        
        # Converter Return → Close
        last_close = test_data['last_close']
        actual_close = test_data['actual_close']
        actual_return = test_data['actual_return']
        
        if last_close is not None and not np.isnan(last_close):
            pred_close = last_close * (1 + pred_return)
        else:
            pred_close = None
        
        # Armazenar resultados
        predictions.append(pred_close if pred_close is not None else np.nan)
        actuals.append(actual_close if actual_close is not None else np.nan)
        dates.append(test_data['date'])
        
        if actual_close is not None and pred_close is not None:
            error = abs(actual_close - pred_close)
            errors.append(error)
        else:
            errors.append(np.nan)
        
        # Online Learning: adicionar feedback
        if enable_online_learning and learner is not None:
            if actual_return is not None and not np.isnan(actual_return):
                # Adicionar feedback
                buffer_before = len(learner.new_data_buffer)
                
                learner.add_prediction_feedback(
                    features=seq_features,
                    actual_return=actual_return,
                    predicted_return=pred_return,
                    actual_close=actual_close if actual_close is not None else 0,
                    predicted_close=pred_close if pred_close is not None else 0,
                    date=test_data['date']
                )
                
                # Verificar se retreinou
                if len(learner.new_data_buffer) < buffer_before:
                    retrain_events.append({
                        'test_number': i,
                        'date': test_data['date'],
                        'total_tests': i
                    })
                    print(f"\n   🔄 Retreinamento #{len(retrain_events)} após {i} testes")
    
    print(f"\n✓ Backtest concluído: {len(predictions)} predições")
    
    # ============================================
    # 5. CALCULAR MÉTRICAS
    # ============================================
    print("\n📊 PASSO 5: Calculando métricas...")
    
    # Remover NaN
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
    predictions_clean = np.array(predictions)[valid_mask]
    actuals_clean = np.array(actuals)[valid_mask]
    errors_clean = np.array(errors)[valid_mask]
    
    if len(predictions_clean) == 0:
        print("❌ Erro: Nenhuma predição válida!")
        return {}
    
    # Métricas gerais
    metrics = calculate_all_metrics(actuals_clean, predictions_clean, verbose=False)
    
    # Métricas por período (antes/depois de retreinamentos)
    period_metrics = {}
    if len(retrain_events) > 0:
        print(f"\n📈 Análise por período (baseado em {len(retrain_events)} retreinamentos):")
        
        for i, event in enumerate(retrain_events):
            start_idx = 0 if i == 0 else retrain_events[i-1]['test_number']
            end_idx = event['test_number']
            
            period_pred = predictions_clean[start_idx:end_idx]
            period_actual = actuals_clean[start_idx:end_idx]
            
            if len(period_pred) > 10:  # Mínimo de dados
                period_metrics[f'Periodo_{i+1}'] = calculate_all_metrics(
                    period_actual, period_pred, verbose=False
                )
        
        # Último período
        if len(retrain_events) > 0:
            last_start = retrain_events[-1]['test_number']
            period_pred = predictions_clean[last_start:]
            period_actual = actuals_clean[last_start:]
            
            if len(period_pred) > 10:
                period_metrics['Periodo_Final'] = calculate_all_metrics(
                    period_actual, period_pred, verbose=False
                )
    else:
        # Dividir em 3 períodos iguais
        n_periods = 3
        period_size = len(predictions_clean) // n_periods
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(predictions_clean)
            
            period_pred = predictions_clean[start_idx:end_idx]
            period_actual = actuals_clean[start_idx:end_idx]
            
            period_metrics[f'Periodo_{i+1}'] = calculate_all_metrics(
                period_actual, period_pred, verbose=False
            )
    
    # ============================================
    # 6. EXIBIR RESULTADOS
    # ============================================
    print("\n" + "="*60)
    print("📊 RESULTADOS DO BACKTEST")
    print("="*60)
    
    print(f"\n📈 Métricas Gerais ({len(predictions_clean)} testes válidos):")
    print(f"   MAE:                 {metrics['MAE']:.4f}")
    print(f"   RMSE:                {metrics['RMSE']:.4f}")
    print(f"   MAPE:                {metrics['MAPE']:.2f}%")
    print(f"   R² Score:            {metrics['R2']:.4f}")
    print(f"   Direction Accuracy:  {metrics['Direction_Accuracy']:.2f}%")
    
    if enable_online_learning:
        print(f"\n🔄 Online Learning:")
        print(f"   Retreinamentos:     {len(retrain_events)}")
        print(f"   Feedback adicionado: {sum(len(learner.new_data_buffer) for _ in [1]) if learner else 0} exemplos")
    
    if len(period_metrics) > 1:
        print(f"\n📊 Métricas por Período:")
        print(f"{'Período':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R²':<10}")
        print("-" * 55)
        
        for period_name, period_met in period_metrics.items():
            print(f"{period_name:<15} "
                  f"{period_met['MAE']:<10.2f} "
                  f"{period_met['RMSE']:<10.2f} "
                  f"{period_met['MAPE']:<10.2f} "
                  f"{period_met['R2']:<10.4f}")
        
        # Verificar melhoria
        if len(period_metrics) >= 2:
            first_r2 = list(period_metrics.values())[0]['R2']
            last_r2 = list(period_metrics.values())[-1]['R2']
            improvement = last_r2 - first_r2
            
            print(f"\n📈 Melhoria (R²): {improvement:+.4f}")
            if improvement > 0:
                print("   ✅ Modelo MELHOROU ao longo do tempo!")
            elif improvement < -0.05:
                print("   ⚠️  Modelo piorou (pode indicar overfitting)")
            else:
                print("   ➡️  Modelo manteve performance estável")
    
    # ============================================
    # 7. SALVAR RESULTADOS
    # ============================================
    print("\n💾 PASSO 6: Salvando resultados...")
    
    results = {
        'backtest_info': {
            'n_tests': len(predictions_clean),
            'n_retrains': len(retrain_events),
            'online_learning_enabled': enable_online_learning,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'metrics': metrics,
        'period_metrics': {k: {m: float(v) for m, v in met.items()} 
                          for k, met in period_metrics.items()},
        'retrain_events': [{'test_number': e['test_number'], 
                          'date': str(e['date'])} for e in retrain_events]
    }
    
    results_path = Path("outputs/backtest_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Resultados salvos em: {results_path}")
    
    # Salvar predições
    predictions_df = pd.DataFrame({
        'date': dates,
        'actual_close': actuals,
        'predicted_close': predictions,
        'error': errors
    })
    predictions_df = predictions_df.dropna()
    
    predictions_path = Path("outputs/backtest_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Predições salvas em: {predictions_path}")
    
    print("\n" + "="*60)
    print("✅ BACKTEST CONCLUÍDO!")
    print("="*60)
    
    return results


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest do modelo LSTM')
    parser.add_argument('--n-tests', type=int, default=1000, help='Número de testes')
    parser.add_argument('--no-online-learning', action='store_true', 
                       help='Desabilitar online learning')
    parser.add_argument('--retrain-threshold', type=int, default=50,
                       help='Threshold para retreinar (online learning)')
    
    args = parser.parse_args()
    
    results = run_backtest(
        n_tests=args.n_tests,
        enable_online_learning=not args.no_online_learning,
        retrain_threshold=args.retrain_threshold
    )
    
    return results


if __name__ == "__main__":
    main()

