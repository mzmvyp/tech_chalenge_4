"""
Script de Aprendizado Diário com Erros
======================================

Processa predições validadas do dia e aprende dos erros usando ErrorFocusedLearner.

Uso:
    python scripts/learn_from_daily_errors.py
    python scripts/learn_from_daily_errors.py --date 2025-11-27
    python scripts/learn_from_daily_errors.py --min-error 0.5 --epochs 3
"""

import sys
from pathlib import Path
import io

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(str(Path(__file__).parent.parent))

import argparse
from datetime import datetime, date
import pandas as pd
import numpy as np

from src.api.prediction_storage import PredictionStorage
from src.models.error_focused_learner import ErrorFocusedLearner
from src.data.data_loader import DataLoader
from src.data.feature_engineering_stationary import create_stationary_features
from src.config import get_config


def learn_from_daily_errors(
    target_date: date = None,
    min_error_pct: float = 0.5,
    epochs: int = 3,
    model_path: str = "models/lstm_model.h5",
    scaler_path: str = "models/scaler.pkl"
):
    """
    Processa erros do dia e aprende deles.
    
    Args:
        target_date: Data a processar (None = hoje)
        min_error_pct: Erro mínimo percentual para considerar
        epochs: Épocas de retreinamento
        model_path: Caminho do modelo
        scaler_path: Caminho do scaler
    """
    if target_date is None:
        target_date = date.today()
    
    print("\n" + "="*60)
    print("APRENDIZADO DIARIO COM ERROS")
    print("="*60)
    print(f"Data: {target_date}")
    print(f"Erro minimo: {min_error_pct}%")
    print(f"Epochs: {epochs}")
    print("="*60)
    
    # Carregar storage
    storage = PredictionStorage()
    
    # Obter erros validados
    errors = storage.get_errors_for_learning(
        min_error_pct=min_error_pct,
        max_errors=1000
    )
    
    # Filtrar por data se especificado
    if target_date:
        target_date_str = target_date.isoformat()
        errors = [e for e in errors if e.get("actual_date") == target_date_str]
    
    if not errors:
        print(f"\nOK: Nenhum erro encontrado para a data {target_date}")
        return
    
    print(f"\nEncontrados {len(errors)} erros para aprender")
    print(f"Erro medio: {np.mean([e.get('error_percentage', 0) for e in errors]):.2f}%")
    
    # Carregar dados históricos para reconstruir features
    config = get_config()
    data_config = config.get_data_config()
    
    loader = DataLoader(
        symbol=data_config['symbol'],
        start_date=data_config['start_date'],
        end_date=data_config['end_date'],
        interval=data_config['interval'],
        vix_symbol=data_config.get('vix_symbol')
    )
    
    df_main, df_vix = loader.load_all_data()
    df_features = create_stationary_features(df_main, df_vix)
    
    # Inicializar ErrorFocusedLearner
    error_learner = ErrorFocusedLearner(
        model_path=model_path,
        scaler_path=scaler_path,
        error_threshold_percentile=75.0,
        error_weight_multiplier=2.0,
        min_error_to_learn=min_error_pct / 100,  # Converter para decimal
        max_error_buffer=1000
    )
    
    error_learner.load_model_and_scaler()
    
    # Adicionar erros ao buffer
    print("\nAdicionando erros ao buffer de aprendizado...")
    for error_entry in errors:
        # Reconstruir features da predição
        # Nota: Isso requer que tenhamos salvo as features originais
        # Por enquanto, vamos usar features do dia da predição
        
        # Calcular Return real
        actual_price = error_entry['actual_price']
        predicted_price = error_entry['predicted_price']
        
        # Para calcular Return, precisamos do preço anterior
        # Vamos usar o último Close disponível antes da predição
        pred_date = datetime.fromisoformat(error_entry['timestamp']).date()
        
        # Encontrar índice da data
        try:
            pred_idx = df_features.index.get_loc(pred_date)
            if pred_idx > 0:
                last_close = df_main['Close'].iloc[pred_idx - 1]
                actual_return = (actual_price - last_close) / last_close
                
                # Obter sequência de features
                sequence_length = error_learner.sequence_length
                seq_start = max(0, pred_idx - sequence_length)
                seq_features = df_features.iloc[seq_start:pred_idx]
                
                if len(seq_features) >= sequence_length:
                    # Adicionar ao buffer
                    error_learner.adaptive_learn_from_error(
                        sequence_features=seq_features,
                        actual_return=actual_return,
                        predicted_return=error_entry['predicted_return'],
                        actual_close=actual_price,
                        predicted_close=predicted_price,
                        date=datetime.fromisoformat(error_entry['timestamp']),
                        immediate_learn=False  # Aprender em batch
                    )
        except (KeyError, IndexError) as e:
            print(f"AVISO: Nao foi possivel processar erro {error_entry.get('prediction_id')}: {e}")
            continue
    
    # Aprender dos erros em batch
    if len(error_learner.error_buffer) > 0:
        print(f"\nAprendendo de {len(error_learner.error_buffer)} erros...")
        error_learner.learn_from_errors(epochs=epochs, verbose=True)
        
        # Salvar modelo atualizado
        updated_model_path = "models/lstm_model_error_learned.h5"
        error_learner.save_model(updated_model_path)
        print(f"\nOK: Modelo atualizado salvo em: {updated_model_path}")
    else:
        print("\nAVISO: Nenhum erro valido para aprender")
    
    print("\n" + "="*60)
    print("APRENDIZADO CONCLUIDO")
    print("="*60)


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Aprender dos erros do dia")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Data a processar (YYYY-MM-DD, default: hoje)"
    )
    parser.add_argument(
        "--min-error",
        type=float,
        default=0.5,
        help="Erro minimo percentual para considerar (default: 0.5)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Epochs de retreinamento (default: 3)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/lstm_model.h5",
        help="Caminho do modelo"
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="models/scaler.pkl",
        help="Caminho do scaler"
    )
    
    args = parser.parse_args()
    
    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    
    learn_from_daily_errors(
        target_date=target_date,
        min_error_pct=args.min_error,
        epochs=args.epochs,
        model_path=args.model_path,
        scaler_path=args.scaler_path
    )


if __name__ == "__main__":
    main()

