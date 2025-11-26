"""
Demonstração do Sistema de Aprendizado Focado em Erros
========================================================

Demonstra como o sistema aprende especificamente dos erros,
similar ao que foi visto na AWS.

Uso:
    python scripts/demo_error_learning.py
"""

import sys
from pathlib import Path
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(str(Path(__file__).parent.parent))

from src.models.error_focused_learner import ErrorFocusedLearner
from src.models.ensemble_predictor import EnsemblePredictor, AdaptiveThresholdPredictor
import pandas as pd
import numpy as np

def main():
    """Demonstra o sistema de aprendizado focado em erros."""
    
    print("\n" + "="*60)
    print("🎯 DEMONSTRAÇÃO: APRENDIZADO FOCADO EM ERROS")
    print("="*60)
    
    print("\n📚 Este sistema implementa aprendizado adaptativo que:")
    print("   1. Identifica predições com erro alto")
    print("   2. Armazena esses erros com peso maior")
    print("   3. Retreina o modelo focando nesses casos difíceis")
    print("   4. Melhora performance em áreas problemáticas")
    print("\n   Similar ao sistema visto na AWS onde a rede neural")
    print("   aprende especificamente dos erros!")
    
    print("\n" + "="*60)
    print("✅ Sistema implementado e pronto para uso!")
    print("="*60)
    print("\n📝 Para usar:")
    print("   1. Execute: python scripts/backtest_with_error_learning.py")
    print("   2. Use --immediate-learn para aprender imediatamente de cada erro")
    print("   3. O modelo será atualizado automaticamente")
    
    print("\n💡 Exemplo de uso:")
    print("   python scripts/backtest_with_error_learning.py --n-tests 500 --immediate-learn")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

