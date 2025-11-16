"""
Módulo de Seleção de Features
==============================

Remove features correlacionadas para reduzir overfitting causado por multicolinearidade.

PROBLEMA: Features altamente correlacionadas (ex: Return, Momentum_5d, Momentum_10d com r>0.85)
causam overfitting pois o modelo "aprende" o mesmo padrão várias vezes.

SOLUÇÃO: Remove features redundantes baseado em correlação.

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class FeatureSelector:
    """Remove features redundantes baseado em correlação."""

    def __init__(self, correlation_threshold: float = 0.8):
        """
        Inicializa o seletor de features.

        Args:
            correlation_threshold: Remove features com correlação maior que este valor (padrão: 0.8)
        """
        self.correlation_threshold = correlation_threshold
        self.selected_features = None
        self.dropped_features = None

        # Features essenciais que NUNCA devem ser removidas
        self.features_to_keep = [
            'Open', 'High', 'Low', 'Close', 'Volume',  # OHLCV básico
            'VIX', 'Return'  # Features importantes
        ]

        print(f"🔍 FeatureSelector inicializado")
        print(f"   Threshold de correlação: {correlation_threshold}")
        print(f"   Features protegidas: {self.features_to_keep}")

    def select_features(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Remove features altamente correlacionadas.

        Estratégia:
        1. Calcula matriz de correlação
        2. Identifica pares com correlação > threshold
        3. Remove features correlacionadas (exceto essenciais)
        4. Retorna DataFrame com features selecionadas

        Args:
            df: DataFrame com todas as features
            verbose: Se True, exibe informações detalhadas

        Returns:
            DataFrame com features selecionadas
        """
        if verbose:
            print("\n" + "="*60)
            print("🔍 SELEÇÃO DE FEATURES (Anti-Multicolinearidade)")
            print("="*60)
            print(f"📊 Features originais: {len(df.columns)}")

        # Calcular matriz de correlação
        corr_matrix = df.corr().abs()

        # Identificar features para remover
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = []
        correlations_found = {}

        for column in upper.columns:
            # Não remover features essenciais
            if column in self.features_to_keep:
                continue

            # Verificar correlações altas
            high_corr_features = upper.index[upper[column] > self.correlation_threshold].tolist()

            if high_corr_features:
                # Se correlacionado com feature essencial, remover esta feature
                essential_correlations = [f for f in high_corr_features if f in self.features_to_keep]

                if essential_correlations:
                    # Correlacionado com feature essencial - remover
                    to_drop.append(column)
                    correlations_found[column] = essential_correlations
                else:
                    # Correlacionado com features não-essenciais
                    # Manter a primeira, remover as outras
                    if column not in to_drop and not any(f in to_drop for f in high_corr_features):
                        # Esta é a primeira - manter
                        pass
                    else:
                        # Outras features já foram marcadas - remover esta
                        to_drop.append(column)
                        correlations_found[column] = high_corr_features

        # Remover duplicatas
        to_drop = list(set(to_drop))
        self.dropped_features = to_drop

        if verbose:
            print(f"❌ Features removidas: {len(to_drop)}")
            print(f"✅ Features mantidas: {len(df.columns) - len(to_drop)}")

            if to_drop:
                print(f"\n📋 Removidas (correlação > {self.correlation_threshold}):")
                for feat in sorted(to_drop):
                    if feat in correlations_found:
                        corr_with = correlations_found[feat]
                        print(f"   ❌ {feat}")
                        print(f"      Correlacionado com: {corr_with}")

        # Remover features
        df_selected = df.drop(columns=to_drop, errors='ignore')
        self.selected_features = df_selected.columns.tolist()

        if verbose:
            print(f"\n✅ Features finais: {len(self.selected_features)}")
            print(f"   {self.selected_features}")
            print("="*60)

        return df_selected

    def get_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Retorna matriz de correlação das features.

        Args:
            df: DataFrame com features

        Returns:
            Matriz de correlação
        """
        return df.corr()

    def get_high_correlations(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Retorna lista de pares de features com alta correlação.

        Args:
            df: DataFrame com features
            threshold: Threshold de correlação (opcional, usa o do __init__)

        Returns:
            Lista de tuplas (feature1, feature2, correlação)
        """
        if threshold is None:
            threshold = self.correlation_threshold

        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        high_corrs = []
        for column in upper.columns:
            high_corr_idx = upper.index[upper[column] > threshold]
            for idx in high_corr_idx:
                high_corrs.append((column, idx, upper.loc[idx, column]))

        # Ordenar por correlação (maior primeiro)
        high_corrs.sort(key=lambda x: x[2], reverse=True)

        return high_corrs

    def analyze_correlations(self, df: pd.DataFrame, top_n: int = 10):
        """
        Analisa e exibe as maiores correlações.

        Args:
            df: DataFrame com features
            top_n: Número de maiores correlações para exibir
        """
        print("\n" + "="*60)
        print("📊 ANÁLISE DE CORRELAÇÕES")
        print("="*60)

        high_corrs = self.get_high_correlations(df)

        if not high_corrs:
            print("✅ Nenhuma correlação alta encontrada!")
            return

        print(f"\nTop {top_n} correlações mais altas:")
        for i, (feat1, feat2, corr) in enumerate(high_corrs[:top_n], 1):
            print(f"{i:2d}. {feat1:20s} <-> {feat2:20s} : {corr:.4f}")

        print(f"\nTotal de pares com correlação > {self.correlation_threshold}: {len(high_corrs)}")
        print("="*60)


if __name__ == "__main__":
    # Teste do módulo
    print("\n" + "="*60)
    print("TESTE DO MÓDULO FEATURE_SELECTOR")
    print("="*60)

    # Criar features correlacionadas de exemplo
    np.random.seed(42)
    n = 100
    base = np.random.randn(n)

    df_test = pd.DataFrame({
        'Close': base,
        'Open': base + np.random.randn(n) * 0.5,
        'Feature1': base + np.random.randn(n) * 0.1,  # Alta correlação com Close
        'Feature2': base + np.random.randn(n) * 0.1,  # Alta correlação com Close
        'Feature3': np.random.randn(n),  # Independente
        'Return': np.random.randn(n),
        'Volume': np.random.randn(n)
    })

    print(f"\nDataset de teste criado: {df_test.shape}")

    # Criar selector
    selector = FeatureSelector(correlation_threshold=0.8)

    # Analisar correlações
    selector.analyze_correlations(df_test)

    # Selecionar features
    df_selected = selector.select_features(df_test)

    print(f"\n✅ Teste concluído!")
    print(f"   Features originais: {len(df_test.columns)}")
    print(f"   Features selecionadas: {len(df_selected.columns)}")
    print(f"   Redução: {(1 - len(df_selected.columns)/len(df_test.columns))*100:.1f}%")
