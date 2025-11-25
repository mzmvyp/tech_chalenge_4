"""
Feature Engineering - Versão com Apenas Features Estacionárias
================================================================

Esta versão cria features APENAS estacionárias, removendo Close/High/Low/Open
que são não-estacionárias e causam problemas no modelo LSTM.

NOVA: Agora inclui padrões de candles e indicadores técnicos!

Autor: Tech Challenge - Fase 04
Data: 2025-11-25
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import warnings

warnings.filterwarnings('ignore')

# Importar módulos de padrões e indicadores
try:
    from src.data.candlestick_patterns import detect_candlestick_patterns, detect_multi_candle_patterns
    from src.data.technical_indicators import add_all_technical_indicators
    HAS_PATTERNS = True
except ImportError:
    HAS_PATTERNS = False
    print("⚠️  Módulos de padrões de candles e indicadores não encontrados")


def create_stationary_features(
    df_main: pd.DataFrame,
    df_vix: Optional[pd.DataFrame] = None,
    use_candlestick_patterns: bool = True,
    use_technical_indicators: bool = True
) -> pd.DataFrame:
    """
    Cria features APENAS estacionárias para o modelo.
    
    Remove: Close, High, Low, Open (não-estacionárias)
    Mantém: Return, Volatility, Momentum, Volume features, VIX
    NOVO: Padrões de candles e indicadores técnicos!
    
    Args:
        df_main: DataFrame principal com OHLCV
        df_vix: DataFrame com VIX (opcional)
        use_candlestick_patterns: Se True, adiciona padrões de candles
        use_technical_indicators: Se True, adiciona indicadores técnicos
    
    Returns:
        DataFrame com apenas features estacionárias
    """
    df = df_main.copy()
    
    # ============================================
    # 0. PADRÕES DE CANDLES E INDICADORES (ANTES DE REMOVER OHLC)
    # ============================================
    if use_candlestick_patterns and HAS_PATTERNS:
        print("   Detectando padroes de candles...")
        df = detect_candlestick_patterns(df)
        df = detect_multi_candle_patterns(df, lookback=3)
        print(f"   ✓ Padroes de candles adicionados")
    
    if use_technical_indicators and HAS_PATTERNS:
        df = add_all_technical_indicators(df)
    
    # ============================================
    # 1. RETURNS (estacionário)
    # ============================================
    if 'Close' in df.columns:
        df['Return'] = df['Close'].pct_change(periods=1)
    
    # ============================================
    # 2. VOLATILIDADE (estacionário)
    # ============================================
    if 'Return' in df.columns:
        for window in [10, 30]:
            df[f'Volatility_{window}d'] = df['Return'].rolling(
                window=window,
                min_periods=window
            ).std()
    
    # ============================================
    # 3. MOMENTUM (estacionário)
    # ============================================
    if 'Return' in df.columns:
        for window in [5, 10]:
            df[f'Momentum_{window}d'] = df['Return'].rolling(
                window=window,
                min_periods=window
            ).mean()
    
    # ============================================
    # 4. VOLUME FEATURES (estacionário)
    # ============================================
    if 'Volume' in df.columns:
        # Volume change
        df['Volume_Change'] = df['Volume'].pct_change(periods=1)
        
        # Volume moving averages
        for window in [5, 20]:
            df[f'Volume_MA_{window}'] = df['Volume'].rolling(
                window=window,
                min_periods=window
            ).mean()
            
            # Volume relativo
            volume_ma = df[f'Volume_MA_{window}']
            df[f'Volume_Relative_{window}'] = df['Volume'] / volume_ma.replace(0, np.nan)
    
    # ============================================
    # 5. VIX (estacionário após merge correto)
    # ============================================
    if df_vix is not None:
        # Preparar VIX
        if isinstance(df_vix, pd.DataFrame):
            if isinstance(df_vix.columns, pd.MultiIndex):
                vix_series = df_vix.iloc[:, 0]
            else:
                vix_series = df_vix['VIX'] if 'VIX' in df_vix.columns else df_vix.iloc[:, 0]
        else:
            vix_series = df_vix
        
        vix_df = pd.DataFrame({'VIX': vix_series}, index=df_vix.index)
        df = df.join(vix_df, how='left')
        
        # Anti-leakage: forward fill + shift
        df['VIX'] = df['VIX'].fillna(method='ffill').shift(1)
        
        # Backward fill para NaN remanescentes
        vix_isna = df['VIX'].isna()
        if vix_isna.any():
            vix_bfill = df['VIX'].fillna(method='bfill')
            df.loc[vix_isna, 'VIX'] = vix_bfill[vix_isna].shift(1)
    
    # ============================================
    # 6. REMOVER FEATURES NÃO-ESTACIONÁRIAS
    # ============================================
    # IMPORTANTE: Remover OHLC mas MANTER indicadores e padrões que foram calculados
    non_stationary = ['Close', 'High', 'Low', 'Open']
    
    # Manter colunas de indicadores e padrões mesmo que usem OHLC no nome
    # (elas já foram calculadas e são estacionárias)
    pattern_keywords = ['Candle_', 'Pattern_', 'RSI', 'MACD', 'BB_', 'Stochastic', 
                        'ATR', 'ADX', 'DI_', 'Return', 'Volatility', 'Momentum', 
                        'Volume', 'VIX']
    
    # Manter colunas que contêm qualquer uma das palavras-chave
    cols_to_keep = [c for c in df.columns if any(keyword in c for keyword in pattern_keywords)]
    
    # Remover apenas OHLC básicos, não os indicadores
    for col in non_stationary:
        if col in df.columns and col not in cols_to_keep:
            df = df.drop(columns=[col])
    
    # ============================================
    # 7. LIMPEZA FINAL
    # ============================================
    # Remover infinitos
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Remover NaN (mantém índice original)
    initial_len = len(df)
    df.dropna(inplace=True)
    removed = initial_len - len(df)
    
    if removed > 0:
        print(f"⚠️  Removidas {removed} linhas com NaN após feature engineering")
    
    # IMPORTANTE: O índice é preservado automaticamente pelo dropna()
    # Isso permite alinhamento correto com close_series depois
    
    return df


def predict_close_from_return(
    last_close: float,
    predicted_return: float
) -> float:
    """
    Converte predição de Return para Close.
    
    Args:
        last_close: Último valor de Close conhecido
        predicted_return: Return predito pelo modelo
    
    Returns:
        Close predito
    """
    return last_close * (1 + predicted_return)

