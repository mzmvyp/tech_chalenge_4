"""
Indicadores Técnicos Avançados
===============================

Implementa indicadores técnicos para análise:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- ADX (Average Directional Index)
- ATR (Average True Range)

Autor: Tech Challenge - Fase 04
Data: 2025-11-25
"""

import pandas as pd
import numpy as np
from typing import Optional


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Adiciona RSI (Relative Strength Index).
    
    RSI > 70: Sobrecomprado (possível queda)
    RSI < 30: Sobrevendido (possível alta)
    
    Args:
        df: DataFrame com coluna 'Close'
        period: Período para cálculo (padrão: 14)
    
    Returns:
        DataFrame com coluna 'RSI'
    """
    df = df.copy()
    
    if 'Close' not in df.columns:
        return df
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Sinais de RSI
    df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
    df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
    df['RSI_Neutral'] = ((df['RSI'] >= 30) & (df['RSI'] <= 70)).astype(int)
    
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Adiciona MACD (Moving Average Convergence Divergence).
    
    Sinal de compra: MACD cruza acima da linha de sinal
    Sinal de venda: MACD cruza abaixo da linha de sinal
    
    Args:
        df: DataFrame com coluna 'Close'
        fast: Período da média rápida
        slow: Período da média lenta
        signal: Período da linha de sinal
    
    Returns:
        DataFrame com colunas MACD, MACD_Signal, MACD_Histogram
    """
    df = df.copy()
    
    if 'Close' not in df.columns:
        return df
    
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Sinais de MACD
    df['MACD_Bullish'] = (
        (df['MACD'] > df['MACD_Signal']) & 
        (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
    ).astype(int)
    
    df['MACD_Bearish'] = (
        (df['MACD'] < df['MACD_Signal']) & 
        (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
    ).astype(int)
    
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    Adiciona Bollinger Bands.
    
    Preço toca banda superior: Possível sobrecompra
    Preço toca banda inferior: Possível sobrevenda
    
    Args:
        df: DataFrame com coluna 'Close'
        period: Período da média móvel
        std_dev: Desvios padrão para as bandas
    
    Returns:
        DataFrame com colunas BB_Upper, BB_Middle, BB_Lower, BB_Width, BB_Position
    """
    df = df.copy()
    
    if 'Close' not in df.columns:
        return df
    
    df['BB_Middle'] = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    
    df['BB_Upper'] = df['BB_Middle'] + (std * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (std * std_dev)
    
    # Largura das bandas (volatilidade)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Posição do preço nas bandas (0 = banda inferior, 1 = banda superior)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
    
    # Sinais
    df['BB_TouchUpper'] = (df['Close'] >= df['BB_Upper']).astype(int)
    df['BB_TouchLower'] = (df['Close'] <= df['BB_Lower']).astype(int)
    
    return df


def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Adiciona Stochastic Oscillator.
    
    %K > 80: Sobrecomprado
    %K < 20: Sobrevendido
    
    Args:
        df: DataFrame com colunas High, Low, Close
        k_period: Período para %K
        d_period: Período para %D (média móvel de %K)
    
    Returns:
        DataFrame com colunas Stochastic_K, Stochastic_D
    """
    df = df.copy()
    
    required_cols = ['High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        return df
    
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    
    df['Stochastic_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min + 1e-10))
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=d_period).mean()
    
    # Sinais
    df['Stochastic_Overbought'] = (df['Stochastic_K'] > 80).astype(int)
    df['Stochastic_Oversold'] = (df['Stochastic_K'] < 20).astype(int)
    
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Adiciona ATR (Average True Range) - medida de volatilidade.
    
    Args:
        df: DataFrame com colunas High, Low, Close
        period: Período para cálculo
    
    Returns:
        DataFrame com coluna 'ATR'
    """
    df = df.copy()
    
    required_cols = ['High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        return df
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=period).mean()
    
    # ATR normalizado (percentual do preço)
    df['ATR_Percent'] = (df['ATR'] / df['Close']) * 100
    
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Adiciona ADX (Average Directional Index) - força da tendência.
    
    ADX > 25: Tendência forte
    ADX < 20: Mercado lateral
    
    Args:
        df: DataFrame com colunas High, Low, Close
        period: Período para cálculo
    
    Returns:
        DataFrame com colunas ADX, +DI, -DI
    """
    df = df.copy()
    
    required_cols = ['High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        return df
    
    # True Range (já calculado se ATR existe)
    if 'ATR' not in df.columns:
        df = add_atr(df, period)
    
    # Directional Movement
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Suavizar
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / df['ATR'])
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / df['ATR'])
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    
    df['ADX'] = adx
    df['DI_Plus'] = plus_di
    df['DI_Minus'] = minus_di
    
    # Sinais
    df['ADX_StrongTrend'] = (df['ADX'] > 25).astype(int)
    df['ADX_WeakTrend'] = (df['ADX'] < 20).astype(int)
    
    return df


def add_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona todos os indicadores técnicos.
    
    Args:
        df: DataFrame com OHLC
    
    Returns:
        DataFrame com todos os indicadores
    """
    df = df.copy()
    
    print("   Adicionando indicadores tecnicos...")
    
    # RSI
    df = add_rsi(df, period=14)
    
    # MACD
    df = add_macd(df, fast=12, slow=26, signal=9)
    
    # Bollinger Bands
    df = add_bollinger_bands(df, period=20, std_dev=2.0)
    
    # Stochastic
    df = add_stochastic(df, k_period=14, d_period=3)
    
    # ATR
    df = add_atr(df, period=14)
    
    # ADX
    df = add_adx(df, period=14)
    
    indicator_keywords = ['RSI', 'MACD', 'BB', 'Stochastic', 'ATR', 'ADX', 'DI']
    n_indicators = len([c for c in df.columns if any(keyword in c for keyword in indicator_keywords)])
    print(f"   OK: {n_indicators} indicadores adicionados")
    
    return df

