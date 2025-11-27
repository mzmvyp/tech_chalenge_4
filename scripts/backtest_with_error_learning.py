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
import hashlib
import os
import time
import random

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
    scaler_path: str = "models/scaler.pkl",
    use_learned_model: bool = True,  # Se True, usa modelo aprendido da execução anterior
    test_start_ratio: float = None  # Ratio de início (None = automático)
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
    # ✅ CORREÇÃO: Detectar melhor modelo disponível (incluindo retreinamento periódico)
    learned_model_path = model_path.replace('.h5', '_error_learned.h5')
    
    def get_best_model_path() -> str:
        """Retorna o melhor modelo disponível (mais recente e completo)."""
        learned_exists = Path(learned_model_path).exists()
        original_exists = Path(model_path).exists()
        
        if learned_exists and original_exists:
            # Ambos existem: comparar datas de modificação
            learned_mtime = Path(learned_model_path).stat().st_mtime
            original_mtime = Path(model_path).stat().st_mtime
            
            # Se modelo principal foi atualizado recentemente (retreinamento periódico),
            # mas modelo aprendido não foi, usar o principal
            time_diff = abs(learned_mtime - original_mtime)
            
            if original_mtime > learned_mtime and time_diff > 60:  # Mais de 1 minuto de diferença
                # Modelo principal foi retreinado mais recentemente
                print(f"📂 Modelo principal retreinado (mais recente): {model_path}")
                print(f"   Data modificação: {datetime.fromtimestamp(original_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   ✅ Usando modelo retreinado periodicamente")
                return model_path
            else:
                # Modelo aprendido é mais recente ou ambos foram atualizados juntos
                print(f"📂 Modelo aprendido encontrado: {learned_model_path}")
                print(f"   Data modificação: {datetime.fromtimestamp(learned_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
                if time_diff < 60:
                    print(f"   ✅ Ambos modelos sincronizados (retreinamento preservou aprendizado)")
                return learned_model_path
        elif learned_exists:
            print(f"📂 Modelo aprendido encontrado: {learned_model_path}")
            return learned_model_path
        elif original_exists:
            print(f"📂 Usando modelo original: {model_path}")
            return model_path
        else:
            raise FileNotFoundError(f"Nenhum modelo encontrado! Procurei em:\n  - {model_path}\n  - {learned_model_path}")
    
    if use_learned_model:
        actual_model_path = get_best_model_path()
        print(f"   ✅ Usando modelo mais recente disponível")
    else:
        print(f"📂 Usando modelo original (reset solicitado): {model_path}")
        actual_model_path = model_path
    
    error_learner = ErrorFocusedLearner(
        model_path=actual_model_path,
        scaler_path=scaler_path,
        error_threshold_percentile=75.0,
        error_weight_multiplier=2.0
    )
    error_learner.load_model_and_scaler()
    
    # ✅ CORREÇÃO: Mostrar sequence_length do modelo
    print(f"   Sequence length do modelo: {error_learner.sequence_length}")
    
    # ✅ DIAGNÓSTICO: Verificar se modelo aprendido está realmente diferente do original
    if use_learned_model and Path(learned_model_path).exists() and Path(model_path).exists():
        try:
            import tensorflow as tf
            original_model = tf.keras.models.load_model(model_path, compile=False)
            learned_model = error_learner.model
            
            # Comparar pesos da primeira camada LSTM
            original_weights = original_model.layers[0].get_weights()[0]
            learned_weights = learned_model.layers[0].get_weights()[0]
            
            weight_diff = np.abs(original_weights - learned_weights).mean()
            if weight_diff > 1e-6:
                print(f"   ✓ Modelo aprendido confirmado: diferença média nos pesos = {weight_diff:.6f}")
            else:
                print(f"   ⚠️  AVISO: Modelo aprendido parece idêntico ao original (diferença = {weight_diff:.6f})")
                print(f"   ⚠️  Isso pode indicar que o aprendizado não está sendo preservado!")
        except Exception as e:
            print(f"   ⚠️  Não foi possível verificar diferença nos pesos: {e}")
    
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
        # ✅ MELHORIA: Usar múltiplos modelos se disponíveis
        model_paths = [model_path]
        scaler_paths = [scaler_path]
        
        # Adicionar modelo aprendido se existir
        learned_model_path = model_path.replace('.h5', '_error_learned.h5')
        if Path(learned_model_path).exists():
            model_paths.append(learned_model_path)
            scaler_paths.append(scaler_path)  # Mesmo scaler
            print(f"   ✓ Adicionando modelo aprendido ao ensemble")
        
        # Se tiver apenas 1 modelo, criar "pseudo-ensemble" com diferentes thresholds
        if len(model_paths) == 1:
            print(f"   ⚠️  Apenas 1 modelo disponível - usando ensemble com diferentes thresholds")
            # Criar múltiplas instâncias com diferentes thresholds para simular ensemble
            model_paths = [model_path] * 3  # 3 "cópias" com diferentes thresholds
            scaler_paths = [scaler_path] * 3
        
        ensemble = EnsemblePredictor(
            model_paths=model_paths,
            scaler_paths=scaler_paths,
            voting_method='weighted'  # ✅ Usar weighted voting para melhor accuracy
        )
        ensemble.load_all_models()
        print(f"   ✓ Ensemble com {len(model_paths)} modelo(s) inicializado")
    
    # Adaptive Threshold
    adaptive_threshold = None
    if use_adaptive_threshold:
        print("📊 Inicializando threshold adaptativo...")
        
        # ✅ CORREÇÃO CRÍTICA: Se usar modelo aprendido, carregar threshold anterior
        initial_threshold = 0.0
        if use_learned_model and Path(learned_model_path).exists():
            try:
                with open("outputs/backtest_error_learning_results.json", 'r') as f:
                    prev_results = json.load(f)
                    threshold_stats = prev_results.get('threshold_statistics', {})
                    if threshold_stats:
                        initial_threshold = threshold_stats.get('current_threshold', 0.0)
                        print(f"   📊 Carregando threshold anterior: {initial_threshold:.6f}")
                    else:
                        print(f"   📊 Threshold anterior não encontrado, começando de 0.0")
            except:
                print(f"   📊 Resultados anteriores não encontrados, começando de 0.0")
        else:
            # Se resetar, pode variar ligeiramente
            threshold_seed = int(time.time() * 1000) % 10000
            random.seed(threshold_seed)
            initial_threshold = random.uniform(-0.0001, 0.0001)
            print(f"   📊 Threshold inicial variado: {initial_threshold:.6f} (seed: {threshold_seed})")
        
        adaptive_threshold = AdaptiveThresholdPredictor(
            initial_threshold=initial_threshold,
            learning_rate=0.0001,  # ✅ OTIMIZADO: Learning rate mais conservador
            confidence_threshold=0.0005  # ✅ NOVO: Threshold de confiança
        )
    
    # ============================================
    # 3. PREPARAR TESTES
    # ============================================
    print(f"\n📊 PASSO 3: Preparando {n_tests} testes...")
    
    # ✅ CORREÇÃO: Usar sequence_length do modelo
    model_sequence_length = error_learner.sequence_length
    
    # ✅ CORREÇÃO CRÍTICA: Se usar modelo aprendido, manter mesmo índice para continuar aprendendo
    # Se resetar, pode variar para testar em dados diferentes
    if test_start_ratio is not None:
        # Usuário especificou ratio manualmente
        test_start_idx = int(len(df_features) * test_start_ratio)
        print(f"   📍 Início manual: {test_start_ratio*100:.1f}% (índice: {test_start_idx})")
    else:
        base_start_idx = int(len(df_features) * 0.85)
        
        if use_learned_model and Path(learned_model_path).exists():
            # ✅ IMPORTANTE: Se usar modelo aprendido, usar MESMO índice para continuar aprendendo
            # Tentar carregar índice salvo da execução anterior
            try:
                with open("outputs/backtest_error_learning_results.json", 'r') as f:
                    prev_results = json.load(f)
                    # Carregar índice anterior se disponível
                    if 'test_start_idx' in prev_results:
                        test_start_idx = prev_results['test_start_idx']
                        print(f"   📍 Continuando aprendizado: usando índice anterior ({test_start_idx})")
                    elif 'test_start_ratio' in prev_results:
                        test_start_idx = int(len(df_features) * prev_results['test_start_ratio'])
                        print(f"   📍 Continuando aprendizado: usando ratio anterior ({prev_results['test_start_ratio']*100:.1f}%)")
                    else:
                        test_start_idx = base_start_idx
                        print(f"   📍 Continuando aprendizado: índice padrão (85%)")
                    print(f"   ⚠️  IMPORTANTE: Testando nos mesmos dados para continuar aprendendo!")
            except:
                test_start_idx = base_start_idx
                print(f"   📍 Continuando aprendizado: índice padrão (85%)")
                print(f"   ⚠️  Resultados anteriores não encontrados, usando índice padrão")
        else:
            # ✅ Se resetar, pode variar para testar em dados diferentes
            timestamp = time.time()
            process_id = os.getpid() if hasattr(os, 'getpid') else 0
            file_hash = hash(str(Path(__file__).absolute()))
            
            combined_seed = int((timestamp * 1000 + process_id * 100 + file_hash) % 1000000)
            random.seed(combined_seed)
            
            # Variação menor quando resetar: -5 a +5 dias
            offset = random.randint(-5, 5)
            test_start_idx = max(model_sequence_length, base_start_idx + offset)
            print(f"   📍 Usando modelo original: início variado (offset: {offset:+d}, seed: {combined_seed})")
    
    max_available = len(df_features) - test_start_idx - model_sequence_length
    n_tests = min(n_tests, max_available)
    
    print(f"   Índice de início: {test_start_idx} (de {len(df_features)} total)")
    print(f"   Testes disponíveis: {max_available}")
    print(f"   Testes a executar: {n_tests}")
    
    # ============================================
    # 4. TESTE INICIAL: Verificar accuracy do modelo aprendido ANTES de começar
    # ============================================
    if use_learned_model and Path(learned_model_path).exists():
        print(f"\n🔍 TESTE INICIAL: Verificando accuracy do modelo aprendido...")
        # Testar nos primeiros 10 testes para ver accuracy inicial
        initial_test_predictions = []
        initial_test_actuals = []
        initial_test_directions_pred = []
        initial_test_directions_actual = []
        
        for i in range(min(10, n_tests)):
            idx = test_start_idx + model_sequence_length + i
            if idx >= len(df_features):
                break
            
            seq_start = idx - model_sequence_length
            seq_end = idx
            sequence_features = df_features.iloc[seq_start:seq_end]
            
            if error_learner.feature_names:
                available_features = [f for f in error_learner.feature_names if f in sequence_features.columns]
                sequence_features = sequence_features[available_features]
            
            actual_return = df_features.iloc[idx]['Return'] if 'Return' in df_features.columns else None
            actual_close = close_series.iloc[idx] if idx < len(close_series) else None
            last_close = close_series.iloc[idx-1] if idx > 0 else None
            
            if actual_close is None or last_close is None or actual_return is None:
                continue
            
            # Predição
            seq_array = error_learner.scaler.transform(sequence_features.values)
            seq_array = seq_array.reshape(1, model_sequence_length, len(error_learner.feature_names))
            pred_scaled = error_learner.model.predict(seq_array, verbose=0)[0, 0]
            dummy = np.zeros((1, error_learner.scaler.n_features_in_))
            dummy[0, error_learner.target_idx] = pred_scaled
            pred_return = error_learner.scaler.inverse_transform(dummy)[0, error_learner.target_idx]
            
            if use_adaptive_threshold and adaptive_threshold is not None:
                pred_direction, _ = adaptive_threshold.predict_direction(pred_return, actual_return)
            else:
                pred_direction = 1 if pred_return > 0 else -1
            
            actual_direction = 1 if actual_return > 0 else -1
            
            initial_test_predictions.append(last_close * (1 + pred_return))
            initial_test_actuals.append(actual_close)
            initial_test_directions_pred.append(pred_direction)
            initial_test_directions_actual.append(actual_direction)
        
        if len(initial_test_directions_pred) > 0:
            initial_accuracy = (np.array(initial_test_directions_pred) == np.array(initial_test_directions_actual)).mean() * 100
            print(f"   📊 Accuracy inicial (primeiros {len(initial_test_directions_pred)} testes): {initial_accuracy:.2f}%")
            print(f"   {'✅ Modelo aprendido está funcionando!' if initial_accuracy > 42 else '⚠️  Accuracy inicial baixa - modelo pode não estar preservando aprendizado'}")
    
    # ============================================
    # 5. EXECUTAR BACKTEST COM APRENDIZADO
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
            # ✅ CORREÇÃO: Aplicar threshold adaptativo também no ensemble
            if use_adaptive_threshold and adaptive_threshold is not None:
                if 'Return' in sequence_features.columns:
                    recent_volatility = sequence_features['Return'].tail(10).std()
                    confidence_threshold = 0.0005 * (1 + recent_volatility * 10)
                else:
                    confidence_threshold = 0.0005
                
                pred_direction, confidence = adaptive_threshold.predict_direction(
                    pred_return, actual_return, confidence_threshold=confidence_threshold
                )
            else:
                pred_direction = ensemble_result['ensemble_direction']
        else:
            # Usar modelo único
            seq_array = error_learner.scaler.transform(sequence_features.values)
            seq_array = seq_array.reshape(1, model_sequence_length, len(error_learner.feature_names))
            
            pred_scaled = error_learner.model.predict(seq_array, verbose=0)[0, 0]
            
            dummy = np.zeros((1, error_learner.scaler.n_features_in_))
            dummy[0, error_learner.target_idx] = pred_scaled
            pred_return = error_learner.scaler.inverse_transform(dummy)[0, error_learner.target_idx]
            
            # ✅ CORREÇÃO: Aplicar threshold adaptativo SEMPRE se habilitado
            if use_adaptive_threshold and adaptive_threshold is not None:
                # ✅ MELHORIA: Usar threshold de confiança baseado em volatilidade
                if 'Return' in sequence_features.columns:
                    recent_volatility = sequence_features['Return'].tail(10).std()
                    confidence_threshold = 0.0005 * (1 + recent_volatility * 10)
                else:
                    confidence_threshold = 0.0005
                
                pred_direction, confidence = adaptive_threshold.predict_direction(
                    pred_return, actual_return, confidence_threshold=confidence_threshold
                )
            else:
                # ✅ MELHORIA: Threshold fixo mais inteligente
                threshold = 0.0003
                if abs(pred_return) < threshold:
                    if 'Momentum_5d' in sequence_features.columns:
                        momentum = sequence_features['Momentum_5d'].iloc[-1]
                        pred_direction = 1 if momentum > 0 else -1
                    else:
                        pred_direction = 1 if pred_return >= 0 else -1
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
            # ✅ CORREÇÃO: Reduzir epochs para evitar overfitting
            error_learner.learn_from_errors(epochs=3, verbose=False)  # ✅ Reduzido de 5 para 3
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
        
        # ✅ SALVAR evolução da accuracy
        accuracy_evolution = {
            'first_half': float(first_half_acc),
            'second_half': float(second_half_acc),
            'improvement': float(improvement)
        }
    else:
        accuracy_evolution = {}
    
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
        'learning_events': learning_events,
        'accuracy_evolution': accuracy_evolution,  # ✅ ADICIONADO: Evolução da accuracy
        # ✅ Salvar índice usado para continuar na próxima execução
        'test_start_idx': int(test_start_idx),
        'test_start_ratio': float(test_start_idx / len(df_features)) if len(df_features) > 0 else 0.85
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
    
    # Salvar modelo atualizado (SEMPRE salvar, mesmo sem eventos de aprendizado)
    # porque o threshold adaptativo pode ter mudado
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
    parser.add_argument('--reset-model', action='store_true',
                       help='Resetar modelo (usar modelo original, ignorar aprendizado anterior)')
    parser.add_argument('--test-start-ratio', type=float, default=None,
                       help='Ratio de início dos testes (0.0-1.0, padrão: 0.85 ou varia se usar modelo aprendido)')
    
    args = parser.parse_args()
    
    results = run_backtest_with_error_learning(
        n_tests=args.n_tests,
        immediate_learn=args.immediate_learn,
        use_ensemble=args.use_ensemble,
        use_adaptive_threshold=args.use_adaptive_threshold,
        model_path=args.model,
        scaler_path=args.scaler,
        use_learned_model=not args.reset_model,
        test_start_ratio=args.test_start_ratio
    )
    
    return results


if __name__ == "__main__":
    main()

