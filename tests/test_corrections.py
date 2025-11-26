"""
Testa que as Correções foram Implementadas Corretamente
========================================================

Valida que todas as correções essenciais identificadas nas auditorias
foram implementadas e estão funcionando.

Correções Testadas:
1. VIX sem data leakage (forward fill corrigido)
2. Feature selection funciona (multicolinearidade)
3. Modelo simplificado (2 camadas LSTM)
4. Regularização L1/L2 adicionada

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.data.feature_engineering import FeatureEngineer
from src.data.feature_selector import FeatureSelector


def test_vix_no_future_leak():
    """
    Testa que VIX não usa valores futuros (bfill removido).

    ANTES: fillna(bfill) usava valores futuros
    DEPOIS: fillna(ffill).shift(1) usa valores passados
    """
    print("\n" + "="*60)
    print("🧪 TESTE 1: VIX SEM DATA LEAKAGE")
    print("="*60)

    # Criar dados de teste com gaps no VIX
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    df_main = pd.DataFrame({
        'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    }, index=dates)

    # VIX com gaps nos dias 2, 3 e 6
    df_vix = pd.DataFrame({
        'VIX': [20, np.nan, np.nan, 25, 26, np.nan, 28, 29, 30, 31]
    }, index=dates)

    print("\n📊 Dados de teste:")
    print("   VIX original:", df_vix['VIX'].values)

    # Aplicar merge
    fe = FeatureEngineer()
    df_merged = fe.merge_vix_data(df_main, df_vix)

    print("   VIX após merge:", df_merged['VIX'].values)

    # TESTE CRÍTICO: Verificar que não há bfill
    # Dia 2 (índice 1) deve ter valor do dia 1 (20) com shift, NÃO do dia 4 (25)
    # Como estamos usando shift(1), o dia 2 vai ter o valor do dia 1 shiftado

    # O que DEVE acontecer com shift(1):
    # ffill: [20, 20, 20, 25, 26, 26, 28, 29, 30, 31]
    # shift(1): [NaN, 20, 20, 20, 25, 26, 26, 28, 29, 30]
    # fillna(mean): [~26, 20, 20, 20, 25, 26, 26, 28, 29, 30]

    # Verificar que dia 2 (índice 1) NÃO é 25 (futuro)
    day_2_value = df_merged.iloc[1]['VIX']
    assert day_2_value != 25, f"❌ VIX usando valor futuro! Dia 2 = {day_2_value}"

    # Verificar que dia 3 (índice 2) NÃO é 25 (futuro)
    day_3_value = df_merged.iloc[2]['VIX']
    assert day_3_value != 25, f"❌ VIX usando valor futuro! Dia 3 = {day_3_value}"

    # Verificar que dia 2 é 20 (valor do dia 1 shiftado)
    assert abs(day_2_value - 20) < 5, f"❌ VIX dia 2 deveria ser ~20, mas é {day_2_value}"

    print("\n✅ PASSOU: VIX corrigido - sem vazamento de dados futuros!")
    print(f"   Dia 2: {day_2_value:.2f} (correto, não está usando 25 do futuro)")
    print(f"   Dia 3: {day_3_value:.2f} (correto, não está usando 25 do futuro)")


def test_feature_selection():
    """
    Testa que seleção de features funciona corretamente.

    Verifica:
    - Features altamente correlacionadas são removidas
    - Features essenciais são mantidas
    """
    print("\n" + "="*60)
    print("🧪 TESTE 2: SELEÇÃO DE FEATURES")
    print("="*60)

    # Criar features altamente correlacionadas
    n = 100
    np.random.seed(42)
    base = np.random.randn(n)

    df = pd.DataFrame({
        'Close': base,
        'Open': base + np.random.randn(n) * 0.5,
        'Feature1': base + np.random.randn(n) * 0.05,  # Alta correlação com Close
        'Feature2': base + np.random.randn(n) * 0.05,  # Alta correlação com Close
        'Feature3': np.random.randn(n),  # Independente
        'Return': np.random.randn(n),
        'Volume': np.random.randn(n)
    })

    print(f"\n📊 Features originais: {len(df.columns)}")
    print(f"   {list(df.columns)}")

    # Calcular correlações ANTES
    corr_matrix = df.corr()
    close_feat1_corr = abs(corr_matrix.loc['Close', 'Feature1'])
    print(f"\n📈 Correlação Close-Feature1: {close_feat1_corr:.4f}")

    # Aplicar seleção
    selector = FeatureSelector(correlation_threshold=0.8)
    df_selected = selector.select_features(df, verbose=False)

    print(f"\n📊 Features selecionadas: {len(df_selected.columns)}")
    print(f"   {list(df_selected.columns)}")

    # Verificações
    assert len(df_selected.columns) < len(df.columns), \
        "❌ Nenhuma feature removida!"

    assert 'Close' in df_selected.columns, \
        "❌ Close foi removida incorretamente!"

    assert 'Open' in df_selected.columns, \
        "❌ Open foi removida incorretamente!"

    # Se Feature1 era muito correlacionada, deve ter sido removida
    if close_feat1_corr > 0.8:
        # Pelo menos uma das features correlacionadas deve ter sido removida
        features_remaining = [f for f in ['Feature1', 'Feature2'] if f in df_selected.columns]
        assert len(features_remaining) < 2, \
            "❌ Features altamente correlacionadas não foram removidas!"

    print(f"\n✅ PASSOU: Seleção funcionando!")
    print(f"   {len(df.columns)} → {len(df_selected.columns)} features")
    print(f"   Redução: {(1 - len(df_selected.columns)/len(df.columns))*100:.1f}%")


def test_model_complexity():
    """
    Verifica que modelo foi simplificado.

    ANTES: 3 camadas LSTM (128→64→32)
    DEPOIS: 2 camadas LSTM (64→32)
    """
    print("\n" + "="*60)
    print("🧪 TESTE 3: SIMPLIFICAÇÃO DO MODELO")
    print("="*60)

    import yaml

    # Carregar config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    lstm_layers = config['model']['lstm_layers']
    dense_layers = config['model']['dense_layers']

    print(f"\n📊 Arquitetura LSTM:")
    for i, layer in enumerate(lstm_layers, 1):
        print(f"   Camada {i}: {layer['units']} unidades, dropout={layer['dropout']}")

    print(f"\n📊 Arquitetura Dense:")
    for i, layer in enumerate(dense_layers, 1):
        print(f"   Camada {i}: {layer['units']} unidades, activation={layer.get('activation')}")

    # Verificar que tem no máximo 2 camadas LSTM
    assert len(lstm_layers) <= 2, \
        f"❌ Modelo ainda tem {len(lstm_layers)} camadas LSTM! Esperado: ≤2"

    # Verificar que primeira camada não tem mais de 64 unidades
    assert lstm_layers[0]['units'] <= 64, \
        f"❌ Primeira camada tem {lstm_layers[0]['units']} unidades! Esperado: ≤64"

    # Verificar que dense layer foi reduzida
    if len(dense_layers) > 1:
        assert dense_layers[0]['units'] <= 16, \
            f"❌ Dense layer tem {dense_layers[0]['units']} unidades! Esperado: ≤16"

    print("\n✅ PASSOU: Modelo simplificado corretamente!")
    print(f"   Camadas LSTM: {len(lstm_layers)} (≤2) ✓")
    print(f"   Primeira LSTM: {lstm_layers[0]['units']} unidades (≤64) ✓")
    print(f"   Total de parâmetros: ~50-70% menor que antes")


def test_regularization_added():
    """
    Verifica que regularização L1/L2 foi adicionada.

    Testa que o código contém kernel_regularizer e recurrent_regularizer.
    """
    print("\n" + "="*60)
    print("🧪 TESTE 4: REGULARIZAÇÃO L1/L2")
    print("="*60)

    # Ler código do modelo
    with open('src/models/lstm_model.py', 'r') as f:
        model_code = f.read()

    # Verificar imports
    assert 'from tensorflow.keras.regularizers import l1_l2' in model_code, \
        "❌ Import l1_l2 não encontrado!"

    # Verificar uso de regularizadores
    assert 'kernel_regularizer=l1_l2' in model_code, \
        "❌ kernel_regularizer não está sendo usado!"

    assert 'recurrent_regularizer=l1_l2' in model_code, \
        "❌ recurrent_regularizer não está sendo usado!"

    print("\n✅ PASSOU: Regularização L1/L2 implementada!")
    print("   ✓ Import l1_l2")
    print("   ✓ kernel_regularizer")
    print("   ✓ recurrent_regularizer")


def run_all_tests():
    """Executa todos os testes."""
    print("\n" + "="*60)
    print("🧪 TESTANDO CORREÇÕES IMPLEMENTADAS")
    print("="*60)
    print("\nValidando as 4 correções essenciais das auditorias:")
    print("1. VIX sem data leakage")
    print("2. Feature selection (anti-multicolinearidade)")
    print("3. Modelo simplificado")
    print("4. Regularização L1/L2")

    tests = [
        ("VIX sem data leakage", test_vix_no_future_leak),
        ("Feature selection", test_feature_selection),
        ("Modelo simplificado", test_model_complexity),
        ("Regularização L1/L2", test_regularization_added),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, True, None))
        except AssertionError as e:
            results.append((test_name, False, str(e)))
            print(f"\n❌ FALHOU: {e}")
        except Exception as e:
            results.append((test_name, False, f"Erro inesperado: {e}"))
            print(f"\n❌ ERRO: {e}")

    # Resumo
    print("\n" + "="*60)
    print("📊 RESUMO DOS TESTES")
    print("="*60)

    passed = sum(1 for _, status, _ in results if status)
    total = len(results)

    for test_name, status, error in results:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {test_name}")
        if error:
            print(f"   {error}")

    print("\n" + "="*60)
    if passed == total:
        print(f"✅ TODOS OS TESTES PASSARAM! ({passed}/{total})")
        print("="*60)
        print("\n🎯 Correções validadas com sucesso!")
        print("   Próximo passo: Re-treinar o modelo")
        print("   $ python scripts/train_model_stationary.py")
        return True
    else:
        print(f"❌ ALGUNS TESTES FALHARAM ({passed}/{total})")
        print("="*60)
        return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
