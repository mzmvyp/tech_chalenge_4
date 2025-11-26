"""
Script de Análise e Otimização do Sistema
==========================================

Este script analisa detalhadamente o sistema e testa diferentes estratégias:
1. Predizer Return vs Close direto
2. Diferentes arquiteturas
3. Otimização de hiperparâmetros
4. Análise de features

Autor: Tech Challenge - Fase 04
Data: 2025-11-25
"""

import sys
from pathlib import Path
import io

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import get_config
from src.data.data_loader import DataLoader
from src.data.feature_engineering_stationary import create_stationary_features
from src.data.feature_selector import FeatureSelector
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.lstm_model import create_model_from_config
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import calculate_all_metrics, compare_with_baselines

# Configurar matplotlib para não usar display
import matplotlib
matplotlib.use('Agg')


def analyze_features(df_features: pd.DataFrame) -> dict:
    """Analisa as features disponíveis."""
    print("\n" + "="*60)
    print("📊 ANÁLISE DE FEATURES")
    print("="*60)
    
    analysis = {
        'n_features': len(df_features.columns),
        'feature_names': df_features.columns.tolist(),
        'missing_values': df_features.isnull().sum().to_dict(),
        'statistics': df_features.describe().to_dict(),
        'correlations': {}
    }
    
    print(f"\n✅ Total de features: {analysis['n_features']}")
    print(f"\n📋 Features disponíveis:")
    for i, feat in enumerate(analysis['feature_names'], 1):
        print(f"   {i:2d}. {feat}")
    
    # Verificar valores faltantes
    missing = {k: v for k, v in analysis['missing_values'].items() if v > 0}
    if missing:
        print(f"\n⚠️  Features com valores faltantes:")
        for feat, count in missing.items():
            print(f"   {feat}: {count} ({count/len(df_features)*100:.1f}%)")
    else:
        print("\n✅ Nenhum valor faltante!")
    
    # Análise de correlação com Return (se existir)
    if 'Return' in df_features.columns:
        corr_with_return = df_features.corr()['Return'].abs().sort_values(ascending=False)
        print(f"\n📈 Correlação com Return:")
        for feat, corr in corr_with_return.head(10).items():
            if feat != 'Return':
                print(f"   {feat:25s}: {corr:.4f}")
        analysis['correlations']['with_return'] = corr_with_return.to_dict()
    
    return analysis


def test_strategy_return_vs_close(
    df_features: pd.DataFrame,
    close_series: pd.Series,
    config: dict
) -> dict:
    """Testa estratégia de predizer Return vs Close direto."""
    print("\n" + "="*60)
    print("🧪 TESTE: Return vs Close Direto")
    print("="*60)
    
    results = {}
    
    # Estratégia 1: Predizer Return (atual)
    print("\n📊 ESTRATÉGIA 1: Predizer Return")
    print("   Vantagens: Estacionário, mais fácil de normalizar")
    print("   Desvantagens: Precisa converter para Close, pode acumular erros")
    
    if 'Return' in df_features.columns:
        return_stats = df_features['Return'].describe()
        print(f"\n   Estatísticas do Return:")
        print(f"      Média: {return_stats['mean']:.6f}")
        print(f"      Std:   {return_stats['std']:.6f}")
        print(f"      Min:   {return_stats['min']:.6f}")
        print(f"      Max:   {return_stats['max']:.6f}")
        
        # Verificar se Return é estacionário (teste simples)
        return_mean = return_stats['mean']
        return_std = return_stats['std']
        if abs(return_mean) < 0.001 and return_std > 0:
            print("   ✅ Return parece estacionário (média próxima de 0)")
        else:
            print("   ⚠️  Return pode não ser estacionário")
    
    # Estratégia 2: Predizer Close direto
    print("\n📊 ESTRATÉGIA 2: Predizer Close Direto")
    print("   Vantagens: Mais direto, sem conversão")
    print("   Desvantagens: Não-estacionário, mais difícil de normalizar")
    
    close_stats = close_series.describe()
    print(f"\n   Estatísticas do Close:")
    print(f"      Média: {close_stats['mean']:.2f}")
    print(f"      Std:   {close_stats['std']:.2f}")
    print(f"      Min:   {close_stats['min']:.2f}")
    print(f"      Max:   {close_stats['max']:.2f}")
    
    # Verificar tendência
    if close_series.is_monotonic_increasing:
        print("   ⚠️  Close tem tendência crescente (não-estacionário)")
    else:
        # Calcular variação percentual média
        pct_change = close_series.pct_change().mean()
        if abs(pct_change) > 0.0001:
            print(f"   ⚠️  Close tem tendência ({pct_change*100:.4f}% por dia)")
    
    results['return_stats'] = return_stats.to_dict() if 'Return' in df_features.columns else None
    results['close_stats'] = close_stats.to_dict()
    
    return results


def analyze_model_architecture(config: dict, n_features: int) -> dict:
    """Analisa a arquitetura do modelo."""
    print("\n" + "="*60)
    print("🏗️  ANÁLISE DA ARQUITETURA")
    print("="*60)
    
    model_config = config['model']
    analysis = {
        'sequence_length': model_config['sequence_length'],
        'n_features': n_features,
        'lstm_layers': len(model_config['lstm_layers']),
        'dense_layers': len(model_config['dense_layers']),
        'total_params_estimate': 0,
        'complexity': 'medium'
    }
    
    print(f"\n📐 Configuração Atual:")
    print(f"   Sequence Length: {analysis['sequence_length']}")
    print(f"   Features: {analysis['n_features']}")
    print(f"   Input Shape: ({analysis['sequence_length']}, {analysis['n_features']})")
    
    print(f"\n🧠 Camadas LSTM:")
    total_lstm_units = 0
    for i, layer in enumerate(model_config['lstm_layers'], 1):
        units = layer['units']
        dropout = layer.get('dropout', 0)
        return_seq = layer.get('return_sequences', False)
        total_lstm_units += units
        print(f"   LSTM {i}: {units} units, dropout={dropout}, return_seq={return_seq}")
    
    print(f"\n🔗 Camadas Dense:")
    for i, layer in enumerate(model_config['dense_layers'], 1):
        units = layer['units']
        activation = layer.get('activation', 'linear')
        dropout = layer.get('dropout', 0)
        print(f"   Dense {i}: {units} units, activation={activation}, dropout={dropout}")
    
    # Estimar número de parâmetros
    # Fórmula aproximada: LSTM: 4 * (units^2 + units * features + units)
    lstm_params = 0
    prev_units = analysis['n_features']
    for layer in model_config['lstm_layers']:
        units = layer['units']
        lstm_params += 4 * (units * units + units * prev_units + units)
        prev_units = units
    
    # Dense: units * prev_units + units
    dense_params = 0
    prev_units = model_config['lstm_layers'][-1]['units']
    for layer in model_config['dense_layers']:
        units = layer['units']
        dense_params += units * prev_units + units
        prev_units = units
    
    total_params = lstm_params + dense_params
    analysis['total_params_estimate'] = total_params
    
    print(f"\n📊 Estimativa de Parâmetros:")
    print(f"   LSTM: ~{lstm_params:,}")
    print(f"   Dense: ~{dense_params:,}")
    print(f"   Total: ~{total_params:,}")
    
    # Avaliar complexidade
    n_train_samples = 1000  # Aproximado
    params_per_sample = total_params / n_train_samples
    
    if params_per_sample > 100:
        analysis['complexity'] = 'high'
        print(f"\n⚠️  Complexidade ALTA: {params_per_sample:.1f} parâmetros por amostra")
        print("   Risco de overfitting! Considere reduzir arquitetura ou aumentar regularização.")
    elif params_per_sample < 10:
        analysis['complexity'] = 'low'
        print(f"\n✅ Complexidade BAIXA: {params_per_sample:.1f} parâmetros por amostra")
        print("   Pode ser subajustado. Considere aumentar capacidade.")
    else:
        analysis['complexity'] = 'medium'
        print(f"\n✅ Complexidade MÉDIA: {params_per_sample:.1f} parâmetros por amostra")
        print("   Balanceado.")
    
    return analysis


def recommend_improvements(analysis: dict) -> dict:
    """Recomenda melhorias baseado na análise."""
    print("\n" + "="*60)
    print("💡 RECOMENDAÇÕES DE MELHORIAS")
    print("="*60)
    
    recommendations = {
        'architecture': [],
        'features': [],
        'training': [],
        'strategy': []
    }
    
    # Recomendações de arquitetura
    if analysis['architecture']['complexity'] == 'high':
        recommendations['architecture'].append({
            'issue': 'Arquitetura muito complexa',
            'suggestion': 'Reduzir unidades LSTM ou adicionar mais regularização',
            'action': 'Reduzir LSTM units para [64, 32, 16] ou aumentar dropout para 0.4-0.5'
        })
    elif analysis['architecture']['complexity'] == 'low':
        recommendations['architecture'].append({
            'issue': 'Arquitetura pode ser subajustada',
            'suggestion': 'Aumentar capacidade do modelo',
            'action': 'Aumentar LSTM units ou adicionar mais camadas'
        })
    
    # Verificar sequence_length
    if analysis['architecture']['sequence_length'] > 90:
        recommendations['architecture'].append({
            'issue': 'Sequence length muito longo',
            'suggestion': 'Reduzir para melhorar eficiência',
            'action': 'Testar sequence_length de 60-75'
        })
    elif analysis['architecture']['sequence_length'] < 30:
        recommendations['architecture'].append({
            'issue': 'Sequence length muito curto',
            'suggestion': 'Aumentar para capturar mais contexto',
            'action': 'Testar sequence_length de 60-90'
        })
    
    # Recomendações de estratégia
    if analysis['strategy']['return_stats']:
        return_std = analysis['strategy']['return_stats'].get('std', 0)
        if return_std < 0.01:
            recommendations['strategy'].append({
                'issue': 'Return muito estável',
                'suggestion': 'Pode ser difícil de prever',
                'action': 'Considerar predizer Close direto ou usar features adicionais'
            })
    
    # Exibir recomendações
    for category, recs in recommendations.items():
        if recs:
            print(f"\n📌 {category.upper()}:")
            for i, rec in enumerate(recs, 1):
                print(f"\n   {i}. {rec['issue']}")
                print(f"      💡 {rec['suggestion']}")
                print(f"      ✅ Ação: {rec['action']}")
    
    return recommendations


def main():
    """Pipeline principal de análise."""
    print("\n" + "="*60)
    print("🔍 ANÁLISE DETALHADA DO SISTEMA")
    print("="*60)
    print(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Carregar configurações
    print("\n📋 Carregando configurações...")
    config = get_config()
    data_config = config.get_data_config()
    model_config = config.get_model_config()
    
    # Carregar dados
    print("\n📥 Carregando dados...")
    loader = DataLoader(
        symbol=data_config['symbol'],
        start_date=data_config['start_date'],
        end_date=data_config['end_date'],
        interval=data_config['interval'],
        vix_symbol=data_config.get('vix_symbol')
    )
    
    df_main, df_vix = loader.load_all_data()
    close_series = df_main['Close'].copy()
    
    # Criar features
    print("\n🔨 Criando features...")
    feature_config = config.config.get('features', {})
    df_features = create_stationary_features(
        df_main, 
        df_vix,
        use_candlestick_patterns=feature_config.get('use_candlestick_patterns', True),
        use_technical_indicators=feature_config.get('use_technical_indicators', True)
    )
    close_series = close_series.loc[df_features.index]
    
    # Feature selection
    print("\n🔍 Selecionando features...")
    selector = FeatureSelector(correlation_threshold=0.8)
    df_features = selector.select_features(df_features, verbose=True)
    
    # Análises
    feature_analysis = analyze_features(df_features)
    strategy_analysis = test_strategy_return_vs_close(df_features, close_series, config.config)
    architecture_analysis = analyze_model_architecture(config.config, len(df_features.columns))
    
    # Combinar análises
    full_analysis = {
        'features': feature_analysis,
        'strategy': strategy_analysis,
        'architecture': architecture_analysis,
        'timestamp': datetime.now().isoformat()
    }
    
    # Recomendações
    recommendations = recommend_improvements(full_analysis)
    full_analysis['recommendations'] = recommendations
    
    # Salvar análise
    analysis_path = Path('outputs') / 'system_analysis.json'
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(analysis_path, 'w') as f:
        json.dump(full_analysis, f, indent=2, default=str)
    
    print(f"\n💾 Análise salva em: {analysis_path}")
    
    print("\n" + "="*60)
    print("✅ ANÁLISE CONCLUÍDA")
    print("="*60)
    
    return full_analysis


if __name__ == "__main__":
    analysis = main()

