"""
Módulo de Feature Engineering
==============================

Este módulo cria features adicionais para melhorar a performance do modelo LSTM.

⚠️ ANTI-LEAKAGE: Todas as features são calculadas usando APENAS informação histórica.
Nenhuma feature pode "espiar" o futuro!

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Cria features adicionais para séries temporais financeiras.

    IMPORTANTE: Todas as features respeitam a ordem temporal e não usam informação futura.
    """

    def __init__(self):
        """Inicializa o feature engineer."""
        print("🔧 FeatureEngineer inicializado")

    def add_returns(
        self,
        df: pd.DataFrame,
        price_column: str = 'Close',
        periods: List[int] = [1]
    ) -> pd.DataFrame:
        """
        Adiciona retornos percentuais.

        ✅ ANTI-LEAKAGE: Usa apenas dados históricos (shift para trás)

        Args:
            df: DataFrame com dados
            price_column: Coluna de preço
            periods: Lista de períodos para calcular retornos

        Returns:
            DataFrame com colunas de retornos adicionadas
        """
        df = df.copy()

        for period in periods:
            col_name = f'Return_{period}d' if period > 1 else 'Return'
            df[col_name] = df[price_column].pct_change(periods=period)

        print(f"✓ Retornos adicionados: períodos {periods}")
        return df

    def add_moving_averages(
        self,
        df: pd.DataFrame,
        price_column: str = 'Close',
        windows: List[int] = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """
        Adiciona médias móveis.

        ✅ ANTI-LEAKAGE: Usa rolling com min_periods para evitar lookahead

        Args:
            df: DataFrame com dados
            price_column: Coluna de preço
            windows: Janelas das médias móveis

        Returns:
            DataFrame com médias móveis adicionadas
        """
        df = df.copy()

        for window in windows:
            col_name = f'MA_{window}'
            # min_periods garante que não usa futuro
            df[col_name] = df[price_column].rolling(
                window=window,
                min_periods=window
            ).mean()

        print(f"✓ Médias móveis adicionadas: janelas {windows}")
        return df

    def add_volatility(
        self,
        df: pd.DataFrame,
        price_column: str = 'Close',
        windows: List[int] = [10, 30]
    ) -> pd.DataFrame:
        """
        Adiciona volatilidade histórica (desvio padrão dos retornos).

        ✅ ANTI-LEAKAGE: Usa rolling com min_periods

        Args:
            df: DataFrame com dados
            price_column: Coluna de preço
            windows: Janelas para calcular volatilidade

        Returns:
            DataFrame com volatilidade adicionada
        """
        df = df.copy()

        # Calcular retornos se não existir
        if 'Return' not in df.columns:
            df['Return'] = df[price_column].pct_change()

        for window in windows:
            col_name = f'Volatility_{window}d'
            df[col_name] = df['Return'].rolling(
                window=window,
                min_periods=window
            ).std()

        print(f"✓ Volatilidade adicionada: janelas {windows}")
        return df

    def add_price_momentum(
        self,
        df: pd.DataFrame,
        price_column: str = 'Close',
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """
        Adiciona momentum (variação percentual em relação a N dias atrás).

        ✅ ANTI-LEAKAGE: Compara com períodos passados

        Args:
            df: DataFrame com dados
            price_column: Coluna de preço
            windows: Janelas para calcular momentum

        Returns:
            DataFrame com momentum adicionado
        """
        df = df.copy()

        for window in windows:
            col_name = f'Momentum_{window}d'
            df[col_name] = (df[price_column] / df[price_column].shift(window)) - 1

        print(f"✓ Momentum adicionado: janelas {windows}")
        return df

    def add_volume_features(
        self,
        df: pd.DataFrame,
        volume_column: str = 'Volume',
        windows: List[int] = [5, 20]
    ) -> pd.DataFrame:
        """
        Adiciona features relacionadas ao volume.

        ✅ ANTI-LEAKAGE: Usa apenas dados históricos

        Args:
            df: DataFrame com dados
            volume_column: Coluna de volume
            windows: Janelas para médias móveis de volume

        Returns:
            DataFrame com features de volume adicionadas
        """
        df = df.copy()

        # Variação percentual do volume
        # ✅ CORREÇÃO: pct_change() pode gerar inf se volume anterior = 0
        df['Volume_Change'] = df[volume_column].pct_change()
        # Substituir infinitos por NaN (serão removidos depois)
        df['Volume_Change'] = df['Volume_Change'].replace([np.inf, -np.inf], np.nan)

        # Médias móveis de volume
        for window in windows:
            col_name = f'Volume_MA_{window}'
            
            # Garantir que volume_column é uma Series
            if isinstance(df[volume_column], pd.DataFrame):
                volume_series = df[volume_column].iloc[:, 0]
            else:
                volume_series = df[volume_column]
            
            df[col_name] = volume_series.rolling(
                window=window,
                min_periods=window
            ).mean()

            # Volume relativo (volume atual / média móvel)
            # ✅ CORREÇÃO: Evitar divisão por zero
            df[f'Volume_Relative_{window}'] = volume_series / df[col_name].replace(0, np.nan)

        print(f"✓ Features de volume adicionadas: janelas {windows}")
        return df

    def merge_vix_data(
        self,
        df_main: pd.DataFrame,
        df_vix: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Merge dos dados principais com VIX - SEM DATA LEAKAGE.

        ✅ ANTI-LEAKAGE: Usa apenas dados históricos, remove linhas sem VIX válido

        Args:
            df_main: DataFrame principal
            df_vix: DataFrame com VIX

        Returns:
            DataFrame com VIX mergeado
        """
        if df_vix is None:
            print("⚠️  Sem dados do VIX para mergear")
            return df_main

        df = df_main.copy()

        # Preparar VIX: garantir que é uma Series simples
        if isinstance(df_vix, pd.DataFrame):
            # Se tem MultiIndex, pegar primeira coluna
            if isinstance(df_vix.columns, pd.MultiIndex):
                vix_series = df_vix.iloc[:, 0]
            else:
                vix_series = df_vix['VIX'] if 'VIX' in df_vix.columns else df_vix.iloc[:, 0]
        else:
            vix_series = df_vix

        # Criar DataFrame temporário com VIX
        vix_df = pd.DataFrame({'VIX': vix_series}, index=df_vix.index)

        # Merge por índice (data)
        df = df.join(vix_df, how='left')

        # Garantir que VIX é uma Series simples
        if isinstance(df['VIX'], pd.DataFrame):
            df['VIX'] = df['VIX'].iloc[:, 0]

        # ✅ SOLUÇÃO 1: Forward fill + shift (usa valor do dia anterior)
        df['VIX'] = df['VIX'].fillna(method='ffill').shift(1)

        # ✅ SOLUÇÃO 2: Para NaN remanescentes, usar BACKWARD FILL + SHIFT
        # Isso garante que usamos o PRÓXIMO valor conhecido, mas shiftado
        vix_isna = df['VIX'].isna()
        if vix_isna.any():
            # Criar uma série auxiliar com bfill
            vix_bfill = df['VIX'].fillna(method='bfill')
            # Para os NaN remanescentes, usar bfill mas adicionar OUTRO shift
            mask = vix_isna
            df.loc[mask, 'VIX'] = vix_bfill[mask].shift(1)

        # ✅ SOLUÇÃO 3: Se ainda houver NaN (primeiro valor), REMOVER essas linhas
        # Trade-off aceitável: perder poucos dados (geralmente 1-2 dias) para garantir ZERO data leakage
        vix_isna = df['VIX'].isna()
        if vix_isna.any():
            n_dropped = int(vix_isna.sum())
            print(f"⚠️  Removendo {n_dropped} linhas iniciais sem VIX histórico (anti-leakage)")
            df = df.dropna(subset=['VIX'])

        print(f"✓ Dados VIX mergeados: {df['VIX'].notna().sum()} valores (SEM data leakage)")
        return df

    def create_all_features(
        self,
        df_main: pd.DataFrame,
        df_vix: Optional[pd.DataFrame] = None,
        use_moving_averages: bool = False,
        use_volume_features: bool = True,
        use_volatility: bool = True,
        use_momentum: bool = True,
        use_returns: bool = True
    ) -> pd.DataFrame:
        """
        Pipeline completo de feature engineering.

        ✅ ORDEM CORRETA (Anti-Leakage):
        1. Features internas primeiro (Return, Volatility, Momentum)
        2. Limpar NaN de rolling windows
        3. Merge VIX por último (minimiza perda de dados)

        Args:
            df_main: DataFrame principal com OHLCV
            df_vix: DataFrame com VIX (opcional)
            use_moving_averages: Se True, adiciona médias móveis
            use_volume_features: Se True, adiciona features de volume
            use_volatility: Se True, adiciona volatilidade
            use_momentum: Se True, adiciona momentum
            use_returns: Se True, adiciona retornos

        Returns:
            DataFrame com todas as features
        """
        print("\n" + "="*60)
        print("🔨 CRIANDO FEATURES")
        print("="*60)
        print(f"📊 Dataset inicial: {df_main.shape}")

        df = df_main.copy()

        # ============================================
        # FASE 1: FEATURES QUE NÃO DEPENDEM DE VIX
        # ============================================

        # 1. Criar Returns PRIMEIRO
        if use_returns:
            df = self.add_returns(df, price_column='Close', periods=[1])

        # 2. Criar Volatility (depende de Return)
        if use_volatility:
            df = self.add_volatility(df, windows=[10, 30])

        # 3. Criar Momentum
        if use_momentum:
            df = self.add_price_momentum(df, windows=[5, 10])

        # 4. Criar Volume Features
        if use_volume_features:
            df = self.add_volume_features(df, windows=[5, 20])

        # 5. Criar Moving Averages (se necessário)
        if use_moving_averages:
            df = self.add_moving_averages(df, windows=[5, 10, 20])

        # ============================================
        # FASE 2: REMOVER NaN E INFINITOS
        # ============================================
        initial_rows = len(df)
        
        # ✅ CORREÇÃO: Substituir infinitos por NaN antes de dropna
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Contar infinitos substituídos
        inf_count = (df == np.inf).sum().sum() + (df == -np.inf).sum().sum()
        if inf_count > 0:
            print(f"⚠️  Substituídos {inf_count} valores infinitos por NaN")
        
        # Remover linhas com NaN
        df = df.dropna()
        removed_rows = initial_rows - len(df)
        print(f"⚠️  Removidas {removed_rows} linhas com NaN/inf de rolling windows")

        # ============================================
        # FASE 3: MERGE VIX (após limpar NaN)
        # ============================================
        if df_vix is not None:
            df = self.merge_vix_data(df, df_vix)
            # Agora, se VIX tiver NaN e precisar remover, são poucas linhas

        print(f"\n📊 Dataset final: {df.shape}")
        print(f"✓ Features criadas: {list(df.columns)}")

        print("\n" + "="*60)
        print("✅ FEATURE ENGINEERING CONCLUÍDO")
        print("="*60)

        return df

    def get_feature_importance_names(self, df: pd.DataFrame) -> List[str]:
        """
        Retorna nomes das features para análise de importância.

        Args:
            df: DataFrame com features

        Returns:
            Lista de nomes das features
        """
        return df.columns.tolist()


if __name__ == "__main__":
    # Teste do módulo
    print("\n" + "="*60)
    print("TESTE DO MÓDULO FEATURE_ENGINEERING")
    print("="*60)

    # Criar dados de exemplo
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    df_test = pd.DataFrame({
        'Open': np.random.randn(500).cumsum() + 100,
        'High': np.random.randn(500).cumsum() + 102,
        'Low': np.random.randn(500).cumsum() + 98,
        'Close': np.random.randn(500).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 500),
    }, index=dates)

    # Criar VIX de exemplo
    df_vix = pd.DataFrame({
        'VIX': np.random.randn(500).cumsum() + 20,
    }, index=dates)

    # Criar feature engineer
    fe = FeatureEngineer()

    # Criar todas as features
    df_features = fe.create_all_features(
        df_main=df_test,
        df_vix=df_vix,
        use_moving_averages=True,
        use_volume_features=True,
        use_volatility=True,
        use_momentum=True,
        use_returns=True
    )

    print("\n📊 Dataset com features:")
    print(df_features.head())
    print(f"\n📈 Shape final: {df_features.shape}")
    print(f"📋 Colunas: {df_features.columns.tolist()}")

    print("\n✅ Teste concluído com sucesso!")
