"""
Módulo de Testes de Estacionariedade
=====================================

Este módulo implementa testes de estacionariedade para séries temporais,
usando o teste de Dickey-Fuller Aumentado (ADF).

Por que é importante:
- LSTM, apesar de ser boa com sequências, assume implicitamente que os dados
  têm distribuição estável ao longo do tempo.
- Features não-estacionárias podem fazer o modelo aprender tendências ao invés
  de padrões, causando overfitting e baixa generalização.

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class StationarityValidator:
    """Valida estacionariedade de séries temporais."""

    def __init__(self, significance_level: float = 0.05):
        """
        Args:
            significance_level: Nível de significância (padrão: 0.05 = 95%)
        """
        self.significance_level = significance_level

    def test_stationarity(
        self,
        series: pd.Series,
        name: str = "Series"
    ) -> Tuple[bool, Dict]:
        """
        Testa estacionariedade usando ADF test.

        Args:
            series: Série temporal para testar
            name: Nome da série (para logging)

        Returns:
            Tupla (is_stationary, test_results)
        """
        # Remover NaN
        series_clean = series.dropna()

        if len(series_clean) < 10:
            # Muito poucos dados para testar
            return False, {
                'name': name,
                'adf_statistic': np.nan,
                'p_value': 1.0,
                'n_lags': 0,
                'n_observations': len(series_clean),
                'critical_values': {},
                'is_stationary': False,
                'interpretation': 'Dados insuficientes para teste (< 10 observações)'
            }

        # Executar teste ADF
        try:
            result = adfuller(series_clean, autolag='AIC')
        except Exception as e:
            return False, {
                'name': name,
                'adf_statistic': np.nan,
                'p_value': 1.0,
                'n_lags': 0,
                'n_observations': len(series_clean),
                'critical_values': {},
                'is_stationary': False,
                'interpretation': f'Erro no teste: {str(e)}'
            }

        # Extrair resultados
        adf_statistic = result[0]
        p_value = result[1]
        n_lags = result[2]
        n_obs = result[3]
        critical_values = result[4]

        # Interpretar resultado
        is_stationary = p_value < self.significance_level

        test_results = {
            'name': name,
            'adf_statistic': adf_statistic,
            'p_value': p_value,
            'n_lags': n_lags,
            'n_observations': n_obs,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'interpretation': self._interpret_result(p_value, adf_statistic, critical_values)
        }

        return is_stationary, test_results

    def _interpret_result(self, p_value, adf_stat, critical_values):
        """Interpreta resultado do teste."""
        if p_value < 0.01:
            return "Fortemente estacionária (p < 0.01)"
        elif p_value < 0.05:
            return "Estacionária (p < 0.05)"
        elif p_value < 0.10:
            return "Fracamente estacionária (p < 0.10)"
        else:
            return "NÃO estacionária (p >= 0.10)"

    def test_all_features(
        self,
        df: pd.DataFrame,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Testa estacionariedade de todas as features.

        Args:
            df: DataFrame com features
            verbose: Se True, exibe resultados

        Returns:
            Dicionário com resultados de cada feature
        """
        results = {}

        if verbose:
            print("\n" + "="*60)
            print("📊 TESTE DE ESTACIONARIEDADE (ADF)")
            print("="*60)

        for column in df.columns:
            is_stationary, test_result = self.test_stationarity(
                df[column],
                name=column
            )
            results[column] = test_result

            if verbose:
                symbol = "✅" if is_stationary else "❌"
                print(f"\n{symbol} {column:25s} p={test_result['p_value']:.4f}")
                print(f"   {test_result['interpretation']}")

        # Resumo
        n_stationary = sum(1 for r in results.values() if r['is_stationary'])
        n_total = len(results)

        if verbose:
            print("\n" + "="*60)
            print(f"📊 RESUMO: {n_stationary}/{n_total} features estacionárias")
            print("="*60)

            if n_stationary < n_total:
                print("\n⚠️  RECOMENDAÇÃO:")
                print("   Considere transformar features não-estacionárias:")
                for name, result in results.items():
                    if not result['is_stationary']:
                        suggestion = self.suggest_transformation(name)
                        print(f"   - {name}: {suggestion}")

        return results

    def suggest_transformation(self, feature_name: str) -> str:
        """Sugere transformação para tornar feature estacionária."""
        suggestions = {
            'Close': 'Usar Return ao invés de Close',
            'Open': 'Usar Return ao invés de Open',
            'High': 'Usar Return ao invés de High',
            'Low': 'Usar Return ao invés de Low',
            'Volume': 'Usar Volume_Change ao invés de Volume',
            'VIX': 'Usar VIX_Change ou diferenciação',
        }

        # Verificar se é uma média móvel
        if 'MA_' in feature_name:
            return f'Usar diferença: df["{feature_name}"].diff()'

        # Verificar se é volume relativo
        if 'Volume_Relative' in feature_name:
            return 'Feature já é relativa, mas pode precisar de diferenciação'

        return suggestions.get(feature_name, f'Usar diferenciação: df["{feature_name}"].diff()')


if __name__ == "__main__":
    # Exemplo de uso
    import pandas as pd
    import numpy as np

    # Criar dados de teste
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    df = pd.DataFrame({
        'Close': np.random.randn(1000).cumsum() + 100,  # Não-estacionária
        'Return': np.random.randn(1000) * 0.02,         # Estacionária
        'Volume': np.random.randint(1e6, 1e7, 1000),    # Não-estacionária
    }, index=dates)

    # Testar
    validator = StationarityValidator()
    results = validator.test_all_features(df)

