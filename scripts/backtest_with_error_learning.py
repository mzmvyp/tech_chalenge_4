"""
Backtest com Aprendizado Focado em Erros
========================================

Executa backtest usando o sistema de aprendizado adaptativo que aprende
especificamente dos erros, similar ao que foi visto na AWS.

Quando o modelo erra uma predição:
1. O erro é identificado
2. O modelo aprende imediatamente desse erro (ou adiciona ao buffer)
3. Foco em casos difíceis (hard examples)
4. Melhora direction accuracy ao longo do tempo

Uso:
    python scripts/backtest_with_error_learning.py
    python scripts/backtest_with_error_learning.py --n-tests 1000 --immediate-learn
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
from src.models.error_focused_learner import ErrorFocusedLearner
from src.models.ensemble_predictor import EnsemblePredictor, AdaptiveThresholdPredictor
from src.evaluation.metrics import calculate_all_metrics


def run_backtest_with_error_learning(
    n_tests: int = 1000,
    immediate_learn: bool = False,
    use_ensemble: bool = False,
    use_adaptive_threshold: bool = True,
    model_path: str = "models/lstm_model.h5",
    scaler_path: str = "models/scaler.pkl"
) -> dict:
    """
    Executa backtest com aprendizado focado em erros.
    
    Args:
        n_tests: Número de testes
        immediate_learn: Se True, aprende imediatamente de cada erro
        use_ensemble: Se True, usa ensemble de modelos
        use_adaptive_threshold: Se True, usa threshold adaptativo
        model_path: Caminho do modelo
        scaler_path: Caminho do scaler
    """
    print("\n" + "="*60)
    print("🎯 BACKTEST COM APRENDIZADO FOCADO EM ERROS")
    print("="*60)
    print(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Número de testes: {n_tests}")
    print(f"🔄 Aprendizado imediato: {'SIM' if immediate_learn else 'NÃO'}")
    print(f"🎯 Ensemble: {'SIM' if use_ensemble else 'NÃO'}")
    print(f"📊 Threshold adaptativo: {'SIM' if use_adaptive_threshold else 'NÃO'}")
    print("="*60)
    
    # ============================================
    # 1. CARREGAR DADOS
    # ============================================
    print("\n📥 PASSO 1: Carregando dados...")
    
    config = get_config()
    data_config = config.get_data_config()
    model_config = config.get_model_config()
    
    loader = DataLoader(
        symbol=data_config['symbol'],
        start_date=data_config['start_date'],
        end_date=data_config['end_date'],
        interval=data_config['interval'],
        vix_symbol=data_config.get('vix_symbol')
    )
    
    df_main, df_vix = loader.load_all_data()
    close_series = df_main['Close'].copy()
    
    # Criar features (com momentum melhorado)
    print("🔨 Criando features estacionárias (com momentum melhorado)...")
    df_features = create_stationary_features(df_main, df_vix)
    close_series = close_series.loc[df_features.index]
    
    sequence_length = model_config['sequence_length']
    
    # ============================================
    # 2. INICIALIZAR SISTEMAS DE APRENDIZADO
    # ============================================
    print("\n🤖 PASSO 2: Inicializando sistemas de aprendizado...")
    
    # Error-Focused Learner
    error_learner = ErrorFocusedLearner(
        model_path=model_path,
        scaler_path=scaler_path,
        error_threshold_percentile=75.0,
        error_weight_multiplier=2.0
    )
    error_learner.load_model_and_scaler()
    
    # ✅ CORREÇÃO: Mostrar sequence_length do modelo
    print(f"   Sequence length do modelo: {error_learner.sequence_length}")
    
    # ✅ CORREÇÃO CRÍTICA: Garantir que df_features tenha apenas as features que o scaler conhece
    if error_learner.feature_names:
        print(f"🔧 Ajustando features para corresponder ao scaler...")
        print(f"   Features disponíveis: {len(df_features.columns)}")
        print(f"   Features do scaler: {len(error_learner.feature_names)}")
        
        # Filtrar apenas features que o scaler conhece, na ordem correta
        available_features = [f for f in error_learner.feature_names if f in df_features.columns]
        
        if len(available_features) != len(error_learner.feature_names):
            print(f"⚠️  Aviso: Algumas features do scaler não estão disponíveis")
            missing = [f for f in error_learner.feature_names if f not in df_features.columns]
            if missing:
                print(f"   Features faltando: {missing}")
            
            # Se faltam muitas features, pode ser problema
            if len(available_features) < len(error_learner.feature_names) * 0.8:
                print(f"❌ Erro: Muitas features faltando! ({len(available_features)}/{len(error_learner.feature_names)})")
                print(f"   Solução: Retreine o modelo com: python scripts/train_model_stationary.py")
                raise ValueError(f"Incompatibilidade de features: {len(available_features)} disponíveis, {len(error_learner.feature_names)} esperadas")
        
        # Reordenar para corresponder à ordem do scaler
        df_features = df_features[available_features]
        
        print(f"✓ Features ajustadas: {len(df_features.columns)} (corresponde ao scaler)")
    else:
        print("⚠️  Aviso: feature_names não disponíveis no scaler")
        print("   Usando todas as features disponíveis (pode causar erro se número não corresponder)")
    
    # Ensemble (se habilitado)
    ensemble = None
    if use_ensemble:
        print("🎯 Inicializando ensemble...")
        ensemble = EnsemblePredictor(
            model_paths=[model_path],  # Por enquanto, mesmo modelo
            scaler_paths=[scaler_path],
            voting_method='weighted'
        )
        ensemble.load_all_models()
    
    # Adaptive Threshold
    adaptive_threshold = None
    if use_adaptive_threshold:
        print("📊 Inicializando threshold adaptativo...")
        adaptive_threshold = AdaptiveThresholdPredictor(
            initial_threshold=0.0,
            learning_rate=0.0001
        )
    
    # ============================================
    # 3. PREPARAR TESTES
    # ============================================
    print(f"\n📊 PASSO 3: Preparando {n_tests} testes...")
    
    # ✅ CORREÇÃO: Usar sequence_length do modelo
    model_sequence_length = error_learner.sequence_length
    test_start_idx = int(len(df_features) * 0.85)
    max_available = len(df_features) - test_start_idx - model_sequence_length
    n_tests = min(n_tests, max_available)
    
    print(f"   Testes disponíveis: {max_available}")
    print(f"   Testes a executar: {n_tests}")
    
    # ============================================
    # 4. EXECUTAR BACKTEST COM APRENDIZADO
    # ============================================
    print(f"\n🧪 PASSO 4: Executando backtest com aprendizado de erros...")
    print("="*60)
    
    predictions = []
    actuals = []
    previous_closes = []
    dates = []
    errors = []
    directions_pred = []
    directions_actual = []
    direction_correct = []
    learning_events = []
    
    for i in range(n_tests):
        if (i + 1) % 50 == 0:
            print(f"   Progresso: {i+1}/{n_tests} ({(i+1)/n_tests*100:.1f}%)")
        
        idx = test_start_idx + model_sequence_length + i
        
        if idx >= len(df_features):
            break
        
        # Sequência de entrada
        # ✅ CORREÇÃO: Usar sequence_length do modelo, não do config
        model_sequence_length = error_learner.sequence_length
        seq_start = idx - model_sequence_length
        seq_end = idx
        sequence_features = df_features.iloc[seq_start:seq_end]
        
        # ✅ CORREÇÃO: Garantir que sequence_features tem apenas as features do scaler
        if error_learner.feature_names:
            available_features = [f for f in error_learner.feature_names if f in sequence_features.columns]
            if len(available_features) != len(error_learner.feature_names):
                # Se faltam features, pular este teste
                continue
            sequence_features = sequence_features[available_features]
        
        # Valores reais
        actual_return = df_features.iloc[idx]['Return'] if 'Return' in df_features.columns else None
        actual_close = close_series.iloc[idx] if idx < len(close_series) else None
        last_close = close_series.iloc[idx-1] if idx > 0 else None
        
        if actual_close is None or last_close is None or actual_return is None:
            continue
        
        # Fazer predição
        if use_ensemble and ensemble is not None:
            # Usar ensemble
            ensemble_result = ensemble.predict_ensemble(sequence_features)
            pred_return = ensemble_result['ensemble_return']
            pred_direction = ensemble_result['ensemble_direction']
        else:
            # Usar modelo único
            seq_array = error_learner.scaler.transform(sequence_features.values)
            seq_array = seq_array.reshape(1, model_sequence_length, len(error_learner.feature_names))
            
            pred_scaled = error_learner.model.predict(seq_array, verbose=0)[0, 0]
            
            dummy = np.zeros((1, error_learner.scaler.n_features_in_))
            dummy[0, error_learner.target_idx] = pred_scaled
            pred_return = error_learner.scaler.inverse_transform(dummy)[0, error_learner.target_idx]
            
            # Aplicar threshold adaptativo se habilitado
            if use_adaptive_threshold and adaptive_threshold is not None:
                pred_direction, confidence = adaptive_threshold.predict_direction(
                    pred_return, actual_return
                )
            else:
                pred_direction = 1 if pred_return > 0 else -1
        
        # Converter Return → Close
        pred_close = last_close * (1 + pred_return)
        
        # Calcular direção real
        actual_direction = 1 if actual_return > 0 else -1
        correct_direction = (pred_direction == actual_direction)
        
        # Armazenar
        predictions.append(pred_close)
        actuals.append(actual_close)
        previous_closes.append(last_close)
        dates.append(df_features.index[idx])
        errors.append(abs(actual_close - pred_close))
        directions_pred.append(pred_direction)
        directions_actual.append(actual_direction)
        direction_correct.append(correct_direction)
        
        # ✅ APRENDIZADO FOCADO EM ERROS
        error_pct = (abs(actual_close - pred_close) / actual_close) * 100
        
        # Se erro é significativo, aprender
        if error_pct > 0.5:  # Erro maior que 0.5%
            if immediate_learn:
                # ✅ CORREÇÃO: Garantir que sequence_features tem o tamanho correto
                # Re-criar sequence_features com o tamanho correto do modelo
                model_seq_len = error_learner.sequence_length
                seq_start_correct = idx - model_seq_len
                seq_end_correct = idx
                sequence_features_correct = df_features.iloc[seq_start_correct:seq_end_correct]
                
                # Filtrar features
                if error_learner.feature_names:
                    available_features = [f for f in error_learner.feature_names if f in sequence_features_correct.columns]
                    sequence_features_correct = sequence_features_correct[available_features]
                
                # Aprender imediatamente (como na AWS)
                error_learner.adaptive_learn_from_error(
                    sequence_features_correct,
                    actual_return,
                    pred_return,
                    actual_close,
                    pred_close,
                    df_features.index[idx],
                    immediate_learn=True
                )
                learning_events.append({
                    'test_number': i,
                    'date': df_features.index[idx],
                    'error_pct': error_pct,
                    'type': 'immediate'
                })
            else:
                # Adicionar ao buffer para aprender depois
                # ✅ CORREÇÃO: Usar sequence_features já filtrado
                error_learner.add_error_feedback(
                    sequence_features,  # Já está filtrado acima
                    actual_return,
                    pred_return,
                    actual_close,
                    pred_close,
                    df_features.index[idx]
                )
        
        # Aprender periodicamente do buffer (a cada 50 erros)
        if len(error_learner.error_buffer) >= 50 and len(error_learner.error_buffer) % 50 == 0:
            print(f"\n   🎯 Aprendendo de {len(error_learner.error_buffer)} erros acumulados...")
            error_learner.learn_from_errors(epochs=5, verbose=False)
            learning_events.append({
                'test_number': i,
                'date': df_features.index[idx],
                'errors_learned': len(error_learner.error_buffer),
                'type': 'batch'
            })
            # Limpar buffer após aprender
            error_learner.error_buffer = []
    
    print(f"\n✓ {len(predictions)} testes concluídos")
    print(f"✓ {len(learning_events)} eventos de aprendizado")
    
    # Converter para arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    previous_closes = np.array(previous_closes)
    errors = np.array(errors)
    direction_correct = np.array(direction_correct)
    
    # ============================================
    # 5. CALCULAR MÉTRICAS
    # ============================================
    print("\n📊 PASSO 5: Calculando métricas...")
    
    metrics = calculate_all_metrics(actuals, predictions, previous_closes, verbose=False)
    
    # Direction Accuracy
    direction_accuracy = (direction_correct.sum() / len(direction_correct)) * 100
    
    # Métricas de erro
    error_stats = error_learner.get_error_statistics()
    
    # Threshold statistics
    threshold_stats = {}
    if use_adaptive_threshold and adaptive_threshold is not None:
        threshold_stats = adaptive_threshold.get_threshold_statistics()
    
    # ============================================
    # 6. EXIBIR RESULTADOS
    # ============================================
    print("\n" + "="*60)
    print("📊 RESULTADOS DO BACKTEST COM APRENDIZADO DE ERROS")
    print("="*60)
    
    print(f"\n📈 MÉTRICAS GERAIS ({len(predictions)} testes):")
    print(f"   MAE:                 {metrics['MAE']:.4f}")
    print(f"   RMSE:                {metrics['RMSE']:.4f}")
    print(f"   MAPE:                {metrics['MAPE']:.2f}%")
    print(f"   R² Score:            {metrics['R2']:.4f}")
    
    print(f"\n🎯 DIRECTION ACCURACY:")
    print(f"   Direction Accuracy:  {direction_accuracy:.2f}%")
    print(f"   Acertos:              {direction_correct.sum()}/{len(direction_correct)}")
    
    if use_adaptive_threshold and threshold_stats:
        print(f"\n📊 THRESHOLD ADAPTATIVO:")
        print(f"   Threshold final:    {threshold_stats.get('current_threshold', 0):.6f}")
        print(f"   Accuracy:           {threshold_stats.get('accuracy', 0):.2f}%")
    
    if error_stats:
        print(f"\n🎯 APRENDIZADO DE ERROS:")
        print(f"   Total de erros:     {error_stats.get('total_errors', 0)}")
        print(f"   Erros no buffer:    {error_stats.get('buffer_size', 0)}")
        print(f"   Erro médio:         {error_stats.get('mean_error_pct', 0):.2f}%")
        print(f"   Eventos de aprendizado: {len(learning_events)}")
    
    # Comparar accuracy antes/depois
    if len(learning_events) > 0:
        # Dividir em períodos
        mid_point = len(predictions) // 2
        first_half_acc = (direction_correct[:mid_point].sum() / mid_point) * 100
        second_half_acc = (direction_correct[mid_point:].sum() / (len(predictions) - mid_point)) * 100
        
        print(f"\n📈 EVOLUÇÃO DA ACCURACY:")
        print(f"   Primeira metade:    {first_half_acc:.2f}%")
        print(f"   Segunda metade:     {second_half_acc:.2f}%")
        improvement = second_half_acc - first_half_acc
        print(f"   Melhoria:           {improvement:+.2f}%")
        
        if improvement > 0:
            print("   ✅ Accuracy MELHOROU ao longo do tempo!")
        elif improvement < -2:
            print("   ⚠️  Accuracy piorou (pode indicar overfitting)")
        else:
            print("   ➡️  Accuracy manteve-se estável")
    
    # ============================================
    # 7. SALVAR RESULTADOS
    # ============================================
    print("\n💾 PASSO 6: Salvando resultados...")
    
    results = {
        'backtest_info': {
            'n_tests': len(predictions),
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'immediate_learn': immediate_learn,
            'use_ensemble': use_ensemble,
            'use_adaptive_threshold': use_adaptive_threshold,
            'learning_events': len(learning_events)
        },
        'metrics': {k: float(v) for k, v in metrics.items()},
        'direction_accuracy': float(direction_accuracy),
        'error_statistics': error_stats,
        'threshold_statistics': threshold_stats,
        'learning_events': learning_events
    }
    
    results_path = Path("outputs/backtest_error_learning_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Resultados salvos em: {results_path}")
    
    # Salvar CSV
    predictions_df = pd.DataFrame({
        'date': dates,
        'actual_close': actuals,
        'predicted_close': predictions,
        'error_absolute': errors,
        'error_percentage': (errors / actuals) * 100,
        'actual_direction': directions_actual,
        'predicted_direction': directions_pred,
        'direction_correct': direction_correct
    })
    
    csv_path = Path("outputs/backtest_error_learning_predictions.csv")
    predictions_df.to_csv(csv_path, index=False)
    print(f"✓ Predições salvas em: {csv_path}")
    
    # Salvar modelo atualizado
    if len(learning_events) > 0:
        updated_model_path = "models/lstm_model_error_learned.h5"
        error_learner.save_model(updated_model_path)
        print(f"✓ Modelo atualizado salvo em: {updated_model_path}")
    
    print("\n" + "="*60)
    print("✅ BACKTEST CONCLUÍDO!")
    print("="*60)
    
    return results


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest com aprendizado focado em erros')
    parser.add_argument('--n-tests', type=int, default=1000,
                       help='Número de testes (padrão: 1000)')
    parser.add_argument('--immediate-learn', action='store_true',
                       help='Aprender imediatamente de cada erro (como AWS)')
    parser.add_argument('--use-ensemble', action='store_true',
                       help='Usar ensemble de modelos')
    parser.add_argument('--use-adaptive-threshold', action='store_true', default=True,
                       help='Usar threshold adaptativo (padrão: True)')
    parser.add_argument('--model', type=str, default="models/lstm_model.h5",
                       help='Caminho do modelo')
    parser.add_argument('--scaler', type=str, default="models/scaler.pkl",
                       help='Caminho do scaler')
    
    args = parser.parse_args()
    
    results = run_backtest_with_error_learning(
        n_tests=args.n_tests,
        immediate_learn=args.immediate_learn,
        use_ensemble=args.use_ensemble,
        use_adaptive_threshold=args.use_adaptive_threshold,
        model_path=args.model,
        scaler_path=args.scaler
    )
    
    return results


if __name__ == "__main__":
    main()

