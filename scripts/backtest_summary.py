"""
Resumo Rápido do Backtest
==========================

Exibe resumo dos resultados do último backtest.

Uso:
    python scripts/backtest_summary.py
"""

import json
from pathlib import Path
import pandas as pd

def main():
    """Exibe resumo do backtest."""
    
    results_path = Path("outputs/backtest_results.json")
    
    if not results_path.exists():
        print("❌ Nenhum resultado de backtest encontrado!")
        print(f"   Execute: python scripts/backtest_model.py")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*60)
    print("RESUMO DO BACKTEST")
    print("="*60)
    
    info = results.get('backtest_info', {})
    metrics = results.get('metrics', {})
    
    print(f"\nMetricas Gerais:")
    print(f"   Testes:              {info.get('n_tests', 'N/A')}")
    print(f"   Online Learning:     {'SIM' if info.get('online_learning_enabled') else 'NAO'}")
    print(f"   Retreinamentos:     {info.get('n_retrains', 0)}")
    print(f"\n   MAE:                 {metrics.get('MAE', 0):.2f}")
    print(f"   RMSE:                {metrics.get('RMSE', 0):.2f}")
    print(f"   MAPE:                {metrics.get('MAPE', 0):.2f}%")
    print(f"   R2 Score:            {metrics.get('R2', 0):.4f}")
    print(f"   Direction Accuracy:  {metrics.get('Direction_Accuracy', 0):.2f}%")
    
    if 'period_metrics' in results:
        print(f"\nMetricas por Periodo:")
        print(f"{'Período':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R²':<10}")
        print("-" * 55)
        
        for period_name, period_met in results['period_metrics'].items():
            print(f"{period_name:<15} "
                  f"{period_met.get('MAE', 0):<10.2f} "
                  f"{period_met.get('RMSE', 0):<10.2f} "
                  f"{period_met.get('MAPE', 0):<10.2f} "
                  f"{period_met.get('R2', 0):<10.4f}")
    
    if 'retrain_events' in results and results['retrain_events']:
        print(f"\nEventos de Retreinamento:")
        for i, event in enumerate(results['retrain_events'], 1):
            print(f"   #{i}: Teste {event.get('test_number', 'N/A')} - {event.get('date', 'N/A')}")
    
    print("\n" + "="*60)
    print(f"Resultados completos: {results_path}")
    print(f"Predicoes: outputs/backtest_predictions.csv")
    print("="*60)


if __name__ == "__main__":
    main()

