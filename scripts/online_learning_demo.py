"""
Demonstração de Online Learning
=================================

Este script demonstra como usar o sistema de online learning:
1. Carrega modelo existente
2. Faz predições
3. Adiciona feedback quando valores reais são conhecidos
4. Retreina automaticamente quando buffer está cheio

Uso:
    python scripts/online_learning_demo.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.models.online_learner import OnlineLearner, create_online_learner_from_config


def main():
    """Demonstração do sistema de online learning."""
    
    print("\n" + "="*60)
    print("🔄 DEMONSTRAÇÃO DE ONLINE LEARNING")
    print("="*60)
    
    # Criar learner
    try:
        learner = create_online_learner_from_config()
    except:
        print("⚠️  Usando configuração padrão (config.yaml pode não ter online_learning)")
        learner = OnlineLearner(
            retrain_threshold=5,  # Baixo para demonstração
            fine_tune_epochs=3
        )
    
    # Carregar modelo
    learner.load_model_and_scaler()
    
    print("\n" + "="*60)
    print("📊 SIMULANDO PREDIÇÕES COM FEEDBACK")
    print("="*60)
    
    # Simular algumas predições
    print("\n💡 Simulando predições diárias...")
    print("   (Em produção, isso viria de predições reais)")
    
    for i in range(7):
        # Simular sequência (em produção, viria do pipeline)
        sequence = np.random.randn(learner.sequence_length, len(learner.feature_names))
        
        # Fazer predição
        result = learner.predict_with_learning(
            sequence=sequence,
            actual_return=np.random.randn() * 0.02,  # Simular return real
            date=datetime.now() - timedelta(days=7-i)
        )
        
        print(f"\n📅 Dia {i+1}:")
        print(f"   Return predito: {result['predicted_return']:.4f}")
        print(f"   Buffer: {len(learner.new_data_buffer)}/{learner.retrain_threshold}")
    
    print("\n" + "="*60)
    print("✅ DEMONSTRAÇÃO CONCLUÍDA")
    print("="*60)
    print("\n💡 Como usar em produção:")
    print("   1. Modelo faz predição para amanhã")
    print("   2. Quando amanhã chega, você tem o valor real")
    print("   3. Chama add_prediction_feedback() com valor real")
    print("   4. Quando buffer atinge threshold, modelo retreina automaticamente")
    print("   5. Modelo melhora ao longo do tempo!")


if __name__ == "__main__":
    main()

