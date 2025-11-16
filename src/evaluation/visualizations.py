"""
Módulo de Visualizações
========================

Este módulo cria visualizações para análise do modelo:
- Histórico de treinamento
- Predições vs Valores Reais
- Distribuição de erros
- Análise de resíduos
- Comparação com baselines

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from pathlib import Path

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ModelVisualizer:
    """
    Cria visualizações para análise do modelo LSTM.
    """

    def __init__(self, save_dir: str = "outputs/figures"):
        """
        Inicializa o visualizador.

        Args:
            save_dir: Diretório para salvar figuras
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"📊 ModelVisualizer inicializado")
        print(f"   Figuras serão salvas em: {self.save_dir}")

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_name: str = "training_history.png"
    ):
        """
        Plota histórico de treinamento (loss e métricas).

        Args:
            history: Dicionário com histórico do Keras
            save_name: Nome do arquivo para salvar
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Evolução da Loss durante Treinamento')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE
        if 'mae' in history:
            axes[1].plot(history['mae'], label='Train MAE', linewidth=2)
            axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
            axes[1].set_xlabel('Época')
            axes[1].set_ylabel('MAE')
            axes[1].set_title('Evolução do MAE durante Treinamento')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.close()

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predições vs Valores Reais",
        save_name: str = "predictions.png",
        dates: Optional[pd.DatetimeIndex] = None,
        show_first_n: Optional[int] = None
    ):
        """
        Plota predições vs valores reais.

        Args:
            y_true: Valores verdadeiros
            y_pred: Valores preditos
            title: Título do gráfico
            save_name: Nome do arquivo para salvar
            dates: Índice de datas (opcional)
            show_first_n: Mostrar apenas primeiros N pontos
        """
        if show_first_n:
            y_true = y_true[:show_first_n]
            y_pred = y_pred[:show_first_n]
            if dates is not None:
                dates = dates[:show_first_n]

        fig, ax = plt.subplots(figsize=(15, 6))

        x = dates if dates is not None else np.arange(len(y_true))

        ax.plot(x, y_true, label='Valores Reais', linewidth=2, alpha=0.7)
        ax.plot(x, y_pred, label='Predições', linewidth=2, alpha=0.7)

        ax.set_xlabel('Data' if dates is not None else 'Amostra')
        ax.set_ylabel('Preço')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if dates is not None:
            plt.xticks(rotation=45)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.close()

    def plot_scatter_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_name: str = "scatter_predictions.png"
    ):
        """
        Plota scatter plot de predições vs valores reais.

        Args:
            y_true: Valores verdadeiros
            y_pred: Valores preditos
            save_name: Nome do arquivo para salvar
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)

        # Linha de referência (predição perfeita)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='Predição Perfeita')

        ax.set_xlabel('Valores Reais')
        ax.set_ylabel('Predições')
        ax.set_title('Scatter Plot: Predições vs Valores Reais')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.close()

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_name: str = "residuals.png"
    ):
        """
        Plota análise de resíduos.

        Args:
            y_true: Valores verdadeiros
            y_pred: Valores preditos
            save_name: Nome do arquivo para salvar
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Resíduos vs Predições
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predições')
        axes[0, 0].set_ylabel('Resíduos')
        axes[0, 0].set_title('Resíduos vs Predições')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Distribuição dos Resíduos
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Resíduos')
        axes[0, 1].set_ylabel('Frequência')
        axes[0, 1].set_title('Distribuição dos Resíduos')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Resíduos ao longo do tempo
        axes[1, 1].plot(residuals, alpha=0.7)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Amostra')
        axes[1, 1].set_ylabel('Resíduos')
        axes[1, 1].set_title('Resíduos ao Longo do Tempo')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.close()

    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_name: str = "error_distribution.png"
    ):
        """
        Plota distribuição de erros (MAE, MAPE).

        Args:
            y_true: Valores verdadeiros
            y_pred: Valores preditos
            save_name: Nome do arquivo para salvar
        """
        absolute_errors = np.abs(y_true - y_pred)
        percentage_errors = np.abs((y_true - y_pred) / y_true) * 100

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Erros absolutos
        axes[0].hist(absolute_errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=np.mean(absolute_errors), color='r',
                       linestyle='--', linewidth=2,
                       label=f'Média: {np.mean(absolute_errors):.2f}')
        axes[0].set_xlabel('Erro Absoluto')
        axes[0].set_ylabel('Frequência')
        axes[0].set_title('Distribuição de Erros Absolutos')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Erros percentuais
        axes[1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=np.mean(percentage_errors), color='r',
                       linestyle='--', linewidth=2,
                       label=f'Média: {np.mean(percentage_errors):.2f}%')
        axes[1].set_xlabel('Erro Percentual (%)')
        axes[1].set_ylabel('Frequência')
        axes[1].set_title('Distribuição de Erros Percentuais (MAPE)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.close()

    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_name: str = "metrics_comparison.png"
    ):
        """
        Plota comparação de métricas entre modelos.

        Args:
            metrics_dict: Dicionário {modelo: {métrica: valor}}
            save_name: Nome do arquivo para salvar
        """
        # Converter para DataFrame
        df = pd.DataFrame(metrics_dict).T

        # Criar subplots
        metrics_to_plot = ['MAE', 'RMSE', 'MAPE', 'R2', 'Direction_Accuracy']
        n_metrics = len([m for m in metrics_to_plot if m in df.columns])

        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        idx = 0
        for metric in metrics_to_plot:
            if metric not in df.columns:
                continue

            ax = axes[idx]
            df[metric].plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.set_ylabel(metric)
            ax.set_title(f'Comparação: {metric}')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            idx += 1

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
        plt.close()

    def create_comprehensive_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        history: Optional[Dict] = None,
        metrics_comparison: Optional[Dict] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        dataset_name: str = "test"
    ):
        """
        Cria relatório visual completo.

        Args:
            y_true: Valores verdadeiros
            y_pred: Valores preditos
            history: Histórico de treinamento
            metrics_comparison: Comparação com baselines
            dates: Índice de datas
            dataset_name: Nome do dataset (para salvar arquivos)
        """
        print("\n" + "="*60)
        print(f"📊 CRIANDO RELATÓRIO VISUAL - {dataset_name.upper()}")
        print("="*60)

        # 1. Histórico de treinamento
        if history is not None:
            self.plot_training_history(history, f"{dataset_name}_training_history.png")

        # 2. Predições vs Reais
        self.plot_predictions(
            y_true, y_pred,
            title=f"Predições vs Valores Reais - {dataset_name.upper()}",
            save_name=f"{dataset_name}_predictions.png",
            dates=dates
        )

        # 3. Scatter plot
        self.plot_scatter_predictions(
            y_true, y_pred,
            save_name=f"{dataset_name}_scatter.png"
        )

        # 4. Análise de resíduos
        self.plot_residuals(
            y_true, y_pred,
            save_name=f"{dataset_name}_residuals.png"
        )

        # 5. Distribuição de erros
        self.plot_error_distribution(
            y_true, y_pred,
            save_name=f"{dataset_name}_errors.png"
        )

        # 6. Comparação de métricas
        if metrics_comparison is not None:
            self.plot_metrics_comparison(
                metrics_comparison,
                save_name=f"{dataset_name}_metrics_comparison.png"
            )

        print("\n✅ Relatório visual completo!")
        print(f"   Figuras salvas em: {self.save_dir}")
        print("="*60)


if __name__ == "__main__":
    # Teste do módulo
    print("\n" + "="*60)
    print("TESTE DO MÓDULO VISUALIZATIONS")
    print("="*60)

    # Criar dados de teste
    np.random.seed(42)
    n = 200
    y_true = np.random.randn(n).cumsum() + 100
    y_pred = y_true + np.random.randn(n) * 3

    # Criar visualizador
    viz = ModelVisualizer(save_dir="outputs/test_figures")

    # Testar predições
    print("\n1️⃣ Testando plot de predições...")
    viz.plot_predictions(y_true, y_pred, save_name="test_predictions.png")

    # Testar scatter
    print("\n2️⃣ Testando scatter plot...")
    viz.plot_scatter_predictions(y_true, y_pred, save_name="test_scatter.png")

    # Testar resíduos
    print("\n3️⃣ Testando análise de resíduos...")
    viz.plot_residuals(y_true, y_pred, save_name="test_residuals.png")

    # Testar distribuição de erros
    print("\n4️⃣ Testando distribuição de erros...")
    viz.plot_error_distribution(y_true, y_pred, save_name="test_errors.png")

    print("\n✅ Teste concluído com sucesso!")
    print(f"   Verifique as figuras em: outputs/test_figures/")
