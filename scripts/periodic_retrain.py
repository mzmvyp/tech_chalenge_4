"""
Script de Retreinamento Periódico do Modelo LSTM
=================================================

Este script:
1. Baixa novos dados do S&P 500 desde a última data de treinamento
2. Adiciona novos dados ao dataset histórico
3. Retreina o modelo incrementalmente (fine-tuning)
4. Valida performance antes de salvar
5. Pode ser executado via cron/scheduler (diariamente, semanalmente)

Uso:
    # Retreinamento diário (adiciona apenas novos dados)
    python scripts/periodic_retrain.py --mode incremental
    
    # Retreinamento completo (retreina do zero com todos os dados)
    python scripts/periodic_retrain.py --mode full
    
    # Retreinamento com período específico
    python scripts/periodic_retrain.py --days 30

Autor: Tech Challenge - Fase 04
Data: 2025-11-26
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime, timedelta
import warnings

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import get_config
from src.data.data_loader import DataLoader
from src.data.feature_engineering_stationary import create_stationary_features, predict_close_from_return
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.lstm_model import create_model_from_config
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import calculate_all_metrics
from src.validation.anti_leakage_tests import AntiLeakageValidator

warnings.filterwarnings('ignore')


def load_last_training_date() -> datetime:
    """Carrega a data do último treinamento."""
    info_path = Path("models/model_info.json")
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
            training_date_str = info.get('training_date', None)
            if training_date_str:
                return datetime.fromisoformat(training_date_str)
    
    # Se não encontrou, retornar data padrão (1 ano atrás)
    return datetime.now() - timedelta(days=365)


def save_training_date(date: datetime):
    """Salva a data do treinamento."""
    info_path = Path("models/model_info.json")
    if info_path.exists():
        with open(info_path, 'r') as f:
            info = json.load(f)
    else:
        info = {}
    
    info['training_date'] = date.isoformat()
    info['last_retrain'] = date.isoformat()
    
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)


def download_new_data(symbol: str = "^GSPC", start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
    """Baixa novos dados do Yahoo Finance."""
    print(f"\nBaixando novos dados de {symbol}...")
    
    if start_date is None:
        start_date = load_last_training_date()
    
    if end_date is None:
        end_date = datetime.now()
    
    # Adicionar alguns dias de margem para garantir dados completos
    start_date = start_date - timedelta(days=5)
    
    print(f"   Período: {start_date.date()} a {end_date.date()}")
    
    loader = DataLoader(
        symbol=symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        interval="1d",
        vix_symbol="^VIX"  # Adicionar VIX também
    )
    
    df_main, df_vix = loader.load_all_data()
    
    if df_main is None or len(df_main) == 0:
        print("AVISO: Nenhum dado novo encontrado!")
        return None, None
    
    print(f"OK: {len(df_main)} novos registros baixados")
    
    return df_main, df_vix


def load_historical_data() -> tuple:
    """Carrega dados históricos já processados."""
    features_path = Path("data/processed/features_stationary.csv")
    
    if features_path.exists():
        print(f"\nCarregando dados historicos de {features_path}...")
        df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
        print(f"OK: {len(df_features)} registros historicos carregados")
        return df_features
    else:
        print("AVISO: Nenhum dado historico encontrado. Sera feito treinamento completo.")
        return None


def merge_data(df_historical: pd.DataFrame, df_new_main: pd.DataFrame, df_new_vix: pd.DataFrame) -> pd.DataFrame:
    """Mescla dados históricos com novos dados."""
    print("\nMesclando dados historicos com novos dados...")
    
    # Criar features dos novos dados
    df_new_features = create_stationary_features(df_new_main, df_new_vix)
    
    # Remover duplicatas (caso haja sobreposição)
    if df_historical is not None:
        # Combinar e remover duplicatas por índice (data)
        df_combined = pd.concat([df_historical, df_new_features])
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        df_combined = df_combined.sort_index()
        
        print(f"OK: Dados mesclados: {len(df_combined)} registros totais")
        print(f"   Históricos: {len(df_historical)}")
        print(f"   Novos: {len(df_new_features)}")
        print(f"   Após mesclagem: {len(df_combined)}")
        
        return df_combined
    else:
        return df_new_features


def get_best_model_path() -> str:
    """
    Retorna o caminho do melhor modelo disponível.
    Prioriza modelo aprendido de erros, depois modelo original.
    """
    learned_model_path = Path("models/lstm_model_error_learned.h5")
    original_model_path = Path("models/lstm_model.h5")
    
    if learned_model_path.exists():
        print(f"OK: Usando modelo aprendido de erros: {learned_model_path}")
        return str(learned_model_path)
    elif original_model_path.exists():
        print(f"OK: Usando modelo original: {original_model_path}")
        return str(original_model_path)
    else:
        raise FileNotFoundError("Nenhum modelo encontrado! Execute o treinamento primeiro.")


def incremental_retrain(
    df_features: pd.DataFrame,
    model_path: str = None,  # None = detectar automaticamente
    scaler_path: str = "models/scaler.pkl",
    epochs: int = 20,  # Menos épocas para fine-tuning
    learning_rate: float = 0.00001  # Learning rate muito baixo
) -> bool:
    """
    Faz retreinamento incremental (fine-tuning) do modelo existente.
    
    Args:
        df_features: DataFrame com features (histórico + novos)
        model_path: Caminho do modelo existente
        scaler_path: Caminho do scaler existente
        epochs: Número de épocas para fine-tuning
        learning_rate: Learning rate para fine-tuning
    
    Returns:
        True se retreinamento foi bem-sucedido
    """
    print("\n" + "="*60)
    print("RETREINAMENTO INCREMENTAL (Fine-Tuning)")
    print("="*60)
    
    try:
        # ✅ CORREÇÃO: Detectar melhor modelo (preserva aprendizado de erros)
        if model_path is None:
            model_path = get_best_model_path()
        
        # Carregar modelo existente
        print(f"\nCarregando modelo de: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Carregar scaler existente
        import joblib
        scaler_data = joblib.load(scaler_path)
        if isinstance(scaler_data, dict):
            scaler = scaler_data['scaler']
            feature_names = scaler_data.get('feature_names', [])
            target_idx = scaler_data.get('target_idx', 0)
        else:
            scaler = scaler_data
            feature_names = []
            target_idx = 0
        
        print(f"OK: Modelo carregado")
        print(f"OK: Scaler carregado ({len(feature_names)} features)")
        
        # Preparar dados
        print(f"\nPreparando dados para fine-tuning...")
        
        # Filtrar features para corresponder ao scaler
        if feature_names:
            available_features = [f for f in feature_names if f in df_features.columns]
            if len(available_features) != len(feature_names):
                print(f"AVISO: {len(available_features)}/{len(feature_names)} features disponiveis")
            df_features_filtered = df_features[available_features]
        else:
            df_features_filtered = df_features
        
        # Normalizar
        data_scaled = scaler.transform(df_features_filtered.values)
        
        # Criar sequências
        config = get_config()
        model_config = config.get_model_config()
        sequence_length = model_config['sequence_length']
        
        X, y = [], []
        for i in range(sequence_length, len(data_scaled)):
            X.append(data_scaled[i-sequence_length:i])
            y.append(data_scaled[i, target_idx])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"OK: {len(X)} sequencias preparadas")
        
        # Split temporal (últimos 15% para validação)
        split_idx = int(len(X) * 0.85)
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]
        
        print(f"   Treino: {len(X_train)} sequências")
        print(f"   Validação: {len(X_val)} sequências")
        
        # Recompilar modelo com learning rate menor
        from tensorflow.keras.optimizers import Adam
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Fine-tuning com early stopping
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print(f"\nIniciando fine-tuning ({epochs} epocas, lr={learning_rate})...")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Avaliar
        val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
        print(f"\nOK: Fine-tuning concluido!")
        print(f"   Val Loss: {val_loss:.6f}")
        
        # ✅ CORREÇÃO: Salvar em ambos os arquivos para preservar aprendizado
        # Salvar no modelo principal
        main_model_path = "models/lstm_model.h5"
        model.save(main_model_path)
        print(f"OK: Modelo salvo em: {main_model_path}")
        
        # Se estava usando modelo aprendido, salvar também lá
        if "error_learned" in model_path:
            learned_model_path = "models/lstm_model_error_learned.h5"
            model.save(learned_model_path)
            print(f"OK: Modelo aprendido atualizado em: {learned_model_path}")
            print(f"  (Preservando aprendizado de erros)")
        
        return True
        
    except Exception as e:
        print(f"ERRO: Erro no retreinamento incremental: {e}")
        import traceback
        traceback.print_exc()
        return False


def full_retrain(df_features: pd.DataFrame):
    """Faz retreinamento completo do modelo (do zero)."""
    print("\n" + "="*60)
    print("RETREINAMENTO COMPLETO")
    print("="*60)
    
    # Usar o script de treinamento normal
    print("\n📝 Executando pipeline completo de treinamento...")
    print("   (Usando scripts/train_model_stationary.py)")
    
    # Importar e executar
    from scripts.train_model_stationary import main as train_main
    
    # Executar treinamento
    train_main()


def main():
    """Pipeline principal de retreinamento periódico."""
    parser = argparse.ArgumentParser(description='Retreinamento Periódico do Modelo LSTM')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['incremental', 'full'],
        default='incremental',
        help='Modo de retreinamento: incremental (fine-tuning) ou full (completo)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='^GSPC',
        help='Símbolo da ação (padrão: ^GSPC para S&P 500)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=None,
        help='Número de dias para baixar (padrão: desde último treinamento)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Número de épocas para fine-tuning (padrão: 20)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.00001,
        help='Learning rate para fine-tuning (padrão: 0.00001)'
    )
    parser.add_argument(
        '--run-backtest',
        action='store_true',
        default=True,
        help='Executar backtest com error learning após retreinamento (padrão: True)'
    )
    parser.add_argument(
        '--backtest-tests',
        type=int,
        default=100,
        help='Número de testes no backtest (padrão: 100)'
    )
    parser.add_argument(
        '--no-backtest',
        action='store_true',
        help='Não executar backtest após retreinamento'
    )
    
    args = parser.parse_args()
    
    # Se --no-backtest foi passado, desabilitar backtest
    if args.no_backtest:
        args.run_backtest = False
    
    print("\n" + "="*60)
    print("RETREINAMENTO PERIODICO DO MODELO LSTM")
    print("="*60)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Modo: {args.mode}")
    print(f"Simbolo: {args.symbol}")
    print("="*60)
    
    try:
        # 1. Determinar período de dados
        if args.days:
            start_date = datetime.now() - timedelta(days=args.days)
        else:
            start_date = load_last_training_date()
        
        end_date = datetime.now()
        
        # 2. Baixar novos dados
        df_new_main, df_new_vix = download_new_data(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        if df_new_main is None or len(df_new_main) == 0:
            print("\nOK: Nenhum dado novo. Modelo ja esta atualizado!")
            return 0
        
        # 3. Carregar dados históricos (se modo incremental)
        df_historical = None
        if args.mode == 'incremental':
            df_historical = load_historical_data()
        
        # 4. Mesclar dados
        df_features = merge_data(df_historical, df_new_main, df_new_vix)
        
        # 5. Retreinar
        if args.mode == 'incremental':
            success = incremental_retrain(
                df_features,
                epochs=args.epochs,
                learning_rate=args.learning_rate
            )
            
            if success:
                # Salvar dados atualizados
                df_features.to_csv("data/processed/features_stationary.csv")
                print(f"\nOK: Dados atualizados salvos")
                
                # Salvar data de treinamento
                save_training_date(datetime.now())
                print(f"OK: Data de treinamento atualizada")
                
                # ✅ NOVO: Executar backtest com error learning após retreinamento
                if args.run_backtest:
                    print("\n" + "="*60)
                    print("EXECUTANDO BACKTEST COM ERROR LEARNING")
                    print("="*60)
                    print("   (Validando modelo recém-retreinado)")
                    
                    try:
                        from scripts.backtest_with_error_learning import run_backtest_with_error_learning
                        
                        backtest_results = run_backtest_with_error_learning(
                            n_tests=args.backtest_tests,
                            immediate_learn=False,  # Não aprender durante validação
                            use_ensemble=True,
                            use_adaptive_threshold=True,
                            use_learned_model=True  # Usar modelo recém-retreinado
                        )
                        
                        if backtest_results:
                            direction_accuracy = backtest_results.get('direction_accuracy', 0)
                            print(f"\nOK: Backtest concluido!")
                            print(f"   Direction Accuracy: {direction_accuracy:.2f}%")
                            
                            # Salvar resultado do backtest
                            backtest_info = {
                                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'after_retrain': True,
                                'direction_accuracy': direction_accuracy,
                                'n_tests': args.backtest_tests
                            }
                            
                            # Adicionar ao histórico de backtests
                            backtest_history_path = Path("outputs/backtest_history.json")
                            if backtest_history_path.exists():
                                with open(backtest_history_path, 'r') as f:
                                    history = json.load(f)
                            else:
                                history = []
                            
                            history.append(backtest_info)
                            
                            # Manter apenas últimos 30 backtests
                            if len(history) > 30:
                                history = history[-30:]
                            
                            backtest_history_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(backtest_history_path, 'w') as f:
                                json.dump(history, f, indent=2, default=str)
                            
                            print(f"   OK: Resultado salvo em: {backtest_history_path}")
                    except Exception as e:
                        print(f"\nAVISO: Erro ao executar backtest: {e}")
                        import traceback
                        traceback.print_exc()
                        print("   (Retreinamento foi bem-sucedido, mas backtest falhou)")
        else:
            # Modo full: salvar dados e executar treinamento completo
            df_features.to_csv("data/processed/features_stationary.csv")
            full_retrain(df_features)
            save_training_date(datetime.now())
        
        print("\n" + "="*60)
        print("OK: RETREINAMENTO CONCLUIDO COM SUCESSO!")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\nERRO: Erro durante retreinamento: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

