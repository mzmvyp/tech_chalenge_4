"""
Padrões de Candles (Candlestick Patterns)
==========================================

Implementa detecção de padrões de candles para análise técnica:
- Padrões de reversão (Hammer, Doji, Engulfing, etc.)
- Padrões de continuação
- Padrões de alta/baixa

Autor: Tech Challenge - Fase 04
Data: 2025-11-25
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta padrões de candles no DataFrame.
    
    Args:
        df: DataFrame com colunas OHLC (Open, High, Low, Close)
    
    Returns:
        DataFrame com colunas adicionais para cada padrão detectado
    """
    df = df.copy()
    
    # Garantir que temos OHLC
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        print("⚠️  Aviso: DataFrame não tem OHLC completo, pulando padrões de candles")
        return df
    
    # Calcular componentes do candle
    body = abs(df['Close'] - df['Open'])
    upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
    lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
    total_range = df['High'] - df['Low']
    
    # Evitar divisão por zero
    body_ratio = body / (total_range + 1e-10)
    upper_shadow_ratio = upper_shadow / (total_range + 1e-10)
    lower_shadow_ratio = lower_shadow / (total_range + 1e-10)
    
    # ============================================
    # PADRÕES DE REVERSÃO (Reversal Patterns)
    # ============================================
    
    # 1. HAMMER (Martelo) - Sinal de alta
    # Corpo pequeno, sombra inferior longa, sombra superior pequena
    df['Candle_Hammer'] = (
        (body_ratio < 0.3) & 
        (lower_shadow_ratio > 0.6) & 
        (upper_shadow_ratio < 0.1) &
        (df['Close'] > df['Open'])  # Candle verde
    ).astype(int)
    
    # 2. HANGING MAN (Enforcado) - Sinal de baixa
    # Similar ao Hammer mas em tendência de alta
    df['Candle_HangingMan'] = (
        (body_ratio < 0.3) & 
        (lower_shadow_ratio > 0.6) & 
        (upper_shadow_ratio < 0.1) &
        (df['Close'] < df['Open'])  # Candle vermelho
    ).astype(int)
    
    # 3. DOJI - Indecisão, possível reversão
    # Corpo muito pequeno
    df['Candle_Doji'] = (body_ratio < 0.1).astype(int)
    
    # 4. ENGULFING BULLISH (Envolvente de Alta) - Reversão de baixa para alta
    df['Candle_EngulfingBullish'] = (
        (df['Close'].shift(1) < df['Open'].shift(1)) &  # Candle anterior vermelho
        (df['Close'] > df['Open']) &  # Candle atual verde
        (df['Open'] < df['Close'].shift(1)) &  # Abre abaixo do fechamento anterior
        (df['Close'] > df['Open'].shift(1))  # Fecha acima da abertura anterior
    ).astype(int)
    
    # 5. ENGULFING BEARISH (Envolvente de Baixa) - Reversão de alta para baixa
    df['Candle_EngulfingBearish'] = (
        (df['Close'].shift(1) > df['Open'].shift(1)) &  # Candle anterior verde
        (df['Close'] < df['Open']) &  # Candle atual vermelho
        (df['Open'] > df['Close'].shift(1)) &  # Abre acima do fechamento anterior
        (df['Close'] < df['Open'].shift(1))  # Fecha abaixo da abertura anterior
    ).astype(int)
    
    # 6. SHOOTING STAR (Estrela Cadente) - Sinal de baixa
    # Corpo pequeno, sombra superior longa
    df['Candle_ShootingStar'] = (
        (body_ratio < 0.3) & 
        (upper_shadow_ratio > 0.6) & 
        (lower_shadow_ratio < 0.1) &
        (df['Close'] < df['Open'])  # Candle vermelho
    ).astype(int)
    
    # 7. INVERTED HAMMER (Martelo Invertido) - Possível reversão de alta
    # Similar ao Shooting Star mas pode ser sinal de alta
    df['Candle_InvertedHammer'] = (
        (body_ratio < 0.3) & 
        (upper_shadow_ratio > 0.6) & 
        (lower_shadow_ratio < 0.1) &
        (df['Close'] > df['Open'])  # Candle verde
    ).astype(int)
    
    # ============================================
    # PADRÕES DE CONTINUAÇÃO
    # ============================================
    
    # 8. MARUBOZU (Candle sem sombras) - Forte tendência
    # Marubozu de Alta
    df['Candle_MarubozuBullish'] = (
        (upper_shadow_ratio < 0.05) & 
        (lower_shadow_ratio < 0.05) &
        (df['Close'] > df['Open']) &
        (body_ratio > 0.8)
    ).astype(int)
    
    # Marubozu de Baixa
    df['Candle_MarubozuBearish'] = (
        (upper_shadow_ratio < 0.05) & 
        (lower_shadow_ratio < 0.05) &
        (df['Close'] < df['Open']) &
        (body_ratio > 0.8)
    ).astype(int)
    
    # ============================================
    # MÉTRICAS DE FORÇA DO CANDLE
    # ============================================
    
    # Força do candle (quanto maior o corpo em relação ao range)
    df['Candle_BodyStrength'] = body_ratio
    
    # Razão de sombras (indica pressão compradora vs vendedora)
    df['Candle_ShadowRatio'] = np.where(
        (upper_shadow + lower_shadow) > 0,
        lower_shadow / (upper_shadow + lower_shadow + 1e-10),
        0.5  # Neutro se não há sombras
    )
    
    # Direção do candle (1 = verde, -1 = vermelho, 0 = doji)
    df['Candle_Direction'] = np.where(
        df['Close'] > df['Open'], 1,
        np.where(df['Close'] < df['Open'], -1, 0)
    )
    
    return df


def detect_multi_candle_patterns(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    Detecta padrões que envolvem múltiplos candles.
    
    Args:
        df: DataFrame com padrões de candles já detectados
        lookback: Quantos candles anteriores analisar
    
    Returns:
        DataFrame com padrões multi-candle
    """
    df = df.copy()
    
    # 1. THREE WHITE SOLDIERS (Três Soldados Brancos) - Forte alta
    # Três candles verdes consecutivos, cada um maior que o anterior
    df['Pattern_ThreeWhiteSoldiers'] = (
        (df['Candle_Direction'].shift(2) == 1) &
        (df['Candle_Direction'].shift(1) == 1) &
        (df['Candle_Direction'] == 1) &
        (df['Close'] > df['Close'].shift(1)) &
        (df['Close'].shift(1) > df['Close'].shift(2))
    ).astype(int)
    
    # 2. THREE BLACK CROWS (Três Corvos) - Forte baixa
    # Três candles vermelhos consecutivos, cada um menor que o anterior
    df['Pattern_ThreeBlackCrows'] = (
        (df['Candle_Direction'].shift(2) == -1) &
        (df['Candle_Direction'].shift(1) == -1) &
        (df['Candle_Direction'] == -1) &
        (df['Close'] < df['Close'].shift(1)) &
        (df['Close'].shift(1) < df['Close'].shift(2))
    ).astype(int)
    
    # 3. MORNING STAR (Estrela da Manhã) - Reversão de baixa para alta
    # Candle vermelho grande, depois pequeno (doji), depois verde grande
    df['Pattern_MorningStar'] = (
        (df['Candle_Direction'].shift(2) == -1) &
        (df['Candle_BodyStrength'].shift(2) > 0.6) &  # Primeiro candle grande
        (df['Candle_Doji'].shift(1) == 1) &  # Segundo é doji
        (df['Candle_Direction'] == 1) &
        (df['Candle_BodyStrength'] > 0.6) &  # Terceiro candle grande
        (df['Close'] > df['Open'].shift(2))  # Fecha acima da abertura do primeiro
    ).astype(int)
    
    # 4. EVENING STAR (Estrela da Tarde) - Reversão de alta para baixa
    # Candle verde grande, depois pequeno (doji), depois vermelho grande
    df['Pattern_EveningStar'] = (
        (df['Candle_Direction'].shift(2) == 1) &
        (df['Candle_BodyStrength'].shift(2) > 0.6) &  # Primeiro candle grande
        (df['Candle_Doji'].shift(1) == 1) &  # Segundo é doji
        (df['Candle_Direction'] == -1) &
        (df['Candle_BodyStrength'] > 0.6) &  # Terceiro candle grande
        (df['Close'] < df['Open'].shift(2))  # Fecha abaixo da abertura do primeiro
    ).astype(int)
    
    return df

