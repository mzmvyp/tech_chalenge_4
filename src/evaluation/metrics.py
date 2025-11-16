"""
Módulo de Métricas de Avaliação
================================

Este módulo implementa métricas para avaliar a performance do modelo:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score
- Direction Accuracy (% de acertos na direção do movimento)

Também implementa baselines para comparação:
- Naive Forecast (amanhã = hoje)
- Moving Average

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, Optional


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula Mean Absolute Error.

    Args:
        y_true: Valores verdadeiros
        y_pred: Valores preditos

    Returns:
        MAE
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula Root Mean Square Error.

    Args:
        y_true: Valores verdadeiros
        y_pred: Valores preditos

    Returns:
        RMSE
    """
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula Mean Absolute Percentage Error.

    Args:
        y_true: Valores verdadeiros
        y_pred: Valores preditos

    Returns:
        MAPE (em porcentagem)
    """
    # Evitar divisão por zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula R² Score.

    Args:
        y_true: Valores verdadeiros
        y_pred: Valores preditos

    Returns:
        R² Score
    """
    return r2_score(y_true, y_pred)


def calculate_direction_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    previous_values: Optional[np.ndarray] = None
) -> float:
    """
    Calcula a acurácia na direção do movimento (subida/descida).

    Args:
        y_true: Valores verdadeiros
        y_pred: Valores preditos
        previous_values: Valores anteriores (t-1). Se None, usa y_true[:-1]

    Returns:
        Direction Accuracy (0 a 100%)
    """
    if previous_values is None:
        # Assumir que queremos comparar t vs t-1
        if len(y_true) <= 1:
            return 0.0
        true_direction = np.sign(y_true[1:] - y_true[:-1])
        pred_direction = np.sign(y_pred[1:] - y_true[:-1])
    else:
        true_direction = np.sign(y_true - previous_values)
        pred_direction = np.sign(y_pred - previous_values)

    # Calcular acurácia
    correct = (true_direction == pred_direction).sum()
    total = len(true_direction)

    return (correct / total) * 100 if total > 0 else 0.0


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    previous_values: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Calcula todas as métricas de uma vez.

    Args:
        y_true: Valores verdadeiros
        y_pred: Valores preditos
        previous_values: Valores anteriores para direction accuracy
        verbose: Se True, exibe resultados

    Returns:
        Dicionário com todas as métricas
    """
    metrics = {
        'MAE': calculate_mae(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred),
        'Direction_Accuracy': calculate_direction_accuracy(y_true, y_pred, previous_values)
    }

    if verbose:
        print("\n" + "="*60)
        print("📊 MÉTRICAS DE AVALIAÇÃO")
        print("="*60)
        print(f"MAE:                 {metrics['MAE']:.4f}")
        print(f"RMSE:                {metrics['RMSE']:.4f}")
        print(f"MAPE:                {metrics['MAPE']:.2f}%")
        print(f"R² Score:            {metrics['R2']:.4f}")
        print(f"Direction Accuracy:  {metrics['Direction_Accuracy']:.2f}%")
        print("="*60)

    return metrics


class BaselinePredictor:
    """
    Implementa modelos baseline para comparação.
    """

    @staticmethod
    def naive_forecast(y_true: np.ndarray) -> np.ndarray:
        """
        Naive Forecast: predição de amanhã = valor de hoje.

        Args:
            y_true: Valores verdadeiros

        Returns:
            Predições naive (shifted by 1)
        """
        # Predição: amanhã = hoje
        # Para alinhar, usamos y[:-1] como predição de y[1:]
        return y_true[:-1]

    @staticmethod
    def moving_average(
        y_true: np.ndarray,
        window: int = 5
    ) -> np.ndarray:
        """
        Média móvel como baseline.

        Args:
            y_true: Valores verdadeiros
            window: Janela da média móvel

        Returns:
            Predições usando média móvel
        """
        if len(y_true) < window:
            raise ValueError(f"Dados insuficientes para janela de {window}")

        predictions = []
        for i in range(window, len(y_true)):
            ma = np.mean(y_true[i-window:i])
            predictions.append(ma)

        return np.array(predictions)

    @staticmethod
    def evaluate_baselines(
        y_true: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Avalia todos os baselines.

        Args:
            y_true: Valores verdadeiros
            verbose: Se True, exibe resultados

        Returns:
            Dicionário com métricas de cada baseline
        """
        results = {}

        # Naive Forecast
        y_pred_naive = BaselinePredictor.naive_forecast(y_true)
        y_true_aligned = y_true[1:]  # Alinhar com predições naive

        results['Naive'] = calculate_all_metrics(
            y_true_aligned,
            y_pred_naive,
            verbose=False
        )

        # Moving Average (janela 5)
        try:
            y_pred_ma5 = BaselinePredictor.moving_average(y_true, window=5)
            y_true_aligned_ma5 = y_true[5:]

            results['MA_5'] = calculate_all_metrics(
                y_true_aligned_ma5,
                y_pred_ma5,
                verbose=False
            )
        except ValueError:
            results['MA_5'] = None

        # Moving Average (janela 20)
        try:
            y_pred_ma20 = BaselinePredictor.moving_average(y_true, window=20)
            y_true_aligned_ma20 = y_true[20:]

            results['MA_20'] = calculate_all_metrics(
                y_true_aligned_ma20,
                y_pred_ma20,
                verbose=False
            )
        except ValueError:
            results['MA_20'] = None

        if verbose:
            print("\n" + "="*60)
            print("📊 COMPARAÇÃO COM BASELINES")
            print("="*60)

            for baseline_name, metrics in results.items():
                if metrics is None:
                    print(f"\n{baseline_name}: Dados insuficientes")
                    continue

                print(f"\n{baseline_name}:")
                print(f"  MAE:  {metrics['MAE']:.4f}")
                print(f"  RMSE: {metrics['RMSE']:.4f}")
                print(f"  MAPE: {metrics['MAPE']:.2f}%")
                print(f"  R²:   {metrics['R2']:.4f}")
                print(f"  Dir:  {metrics['Direction_Accuracy']:.2f}%")

            print("="*60)

        return results


def compare_with_baselines(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "LSTM",
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compara métricas do modelo com baselines.

    Args:
        y_true: Valores verdadeiros
        y_pred: Predições do modelo
        model_name: Nome do modelo
        verbose: Se True, exibe comparação

    Returns:
        Dicionário com métricas de todos os modelos
    """
    # Métricas do modelo
    model_metrics = calculate_all_metrics(y_true, y_pred, verbose=False)

    # Métricas dos baselines
    baseline_metrics = BaselinePredictor.evaluate_baselines(y_true, verbose=False)

    # Combinar
    all_metrics = {
        model_name: model_metrics,
        **baseline_metrics
    }

    if verbose:
        print("\n" + "="*60)
        print(f"📊 COMPARAÇÃO: {model_name} vs BASELINES")
        print("="*60)
        print(f"\n{'Modelo':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R²':<10} {'Dir_Acc':<10}")
        print("-" * 65)

        for model, metrics in all_metrics.items():
            if metrics is None:
                continue

            print(f"{model:<15} "
                  f"{metrics['MAE']:<10.4f} "
                  f"{metrics['RMSE']:<10.4f} "
                  f"{metrics['MAPE']:<10.2f} "
                  f"{metrics['R2']:<10.4f} "
                  f"{metrics['Direction_Accuracy']:<10.2f}")

        print("="*60)

        # Verificar se o modelo é melhor que naive
        if model_metrics['RMSE'] < all_metrics['Naive']['RMSE']:
            improvement = ((all_metrics['Naive']['RMSE'] - model_metrics['RMSE'])
                          / all_metrics['Naive']['RMSE'] * 100)
            print(f"\n✅ Modelo {model_name} é {improvement:.1f}% melhor que Naive Forecast (RMSE)")
        else:
            print(f"\n⚠️  Modelo {model_name} NÃO superou o Naive Forecast!")
            print("   Isso pode indicar problemas no modelo ou data leakage.")

    return all_metrics


if __name__ == "__main__":
    # Teste do módulo
    print("\n" + "="*60)
    print("TESTE DO MÓDULO METRICS")
    print("="*60)

    # Criar dados de teste
    np.random.seed(42)
    n = 100
    y_true = np.random.randn(n).cumsum() + 100

    # Simular predições com algum erro
    y_pred = y_true + np.random.randn(n) * 2

    # Calcular métricas
    print("\n1️⃣ Testando cálculo de métricas individuais:")
    metrics = calculate_all_metrics(y_true, y_pred)

    # Testar baselines
    print("\n2️⃣ Testando baselines:")
    baseline_results = BaselinePredictor.evaluate_baselines(y_true)

    # Comparar com baselines
    print("\n3️⃣ Testando comparação:")
    comparison = compare_with_baselines(y_true, y_pred, model_name="Test_Model")

    print("\n✅ Teste concluído com sucesso!")
