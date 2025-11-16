"""
Módulo de Testes Anti-Data-Leakage
===================================

Este módulo implementa testes automáticos para detectar data leakage.

⚠️ CRÍTICO: Data leakage é quando informação do futuro "vaza" para o passado,
resultando em métricas artificialmente boas que não se reproduzem em produção.

Testes implementados:
1. Verificação de ordem temporal nos splits
2. Verificação de que não há sobreposição entre conjuntos
3. Verificação de performance suspeita (muito boa demais)
4. Comparação com baselines
5. Verificação de que scaler foi fitado apenas no treino

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import warnings


class AntiLeakageValidator:
    """
    Valida que não há data leakage no pipeline de ML.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Inicializa o validador.

        Args:
            strict_mode: Se True, falha em qualquer suspeita de leakage
        """
        self.strict_mode = strict_mode
        self.test_results = {}

        print("🔍 AntiLeakageValidator inicializado")
        print(f"   Modo Strict: {'SIM' if strict_mode else 'NÃO'}")

    def test_temporal_order(
        self,
        train_dates: pd.DatetimeIndex,
        val_dates: pd.DatetimeIndex,
        test_dates: pd.DatetimeIndex
    ) -> bool:
        """
        Testa se os splits respeitam ordem temporal.

        ✅ TESTE 1: Ordem Temporal

        Args:
            train_dates: Datas do conjunto de treino
            val_dates: Datas do conjunto de validação
            test_dates: Datas do conjunto de teste

        Returns:
            True se passou no teste
        """
        print("\n🧪 TESTE 1: Verificando Ordem Temporal...")

        # Verificar que treino < validação < teste
        train_max = train_dates.max()
        val_min = val_dates.min()
        val_max = val_dates.max()
        test_min = test_dates.min()

        passed = True
        issues = []

        # Treino deve terminar antes da validação começar
        if train_max >= val_min:
            passed = False
            issues.append(f"Treino termina ({train_max}) depois/durante validação ({val_min})")

        # Validação deve terminar antes do teste começar
        if val_max >= test_min:
            passed = False
            issues.append(f"Validação termina ({val_max}) depois/durante teste ({test_min})")

        # Verificar ordem interna
        if not train_dates.is_monotonic_increasing:
            passed = False
            issues.append("Datas de treino não estão em ordem crescente")

        if not val_dates.is_monotonic_increasing:
            passed = False
            issues.append("Datas de validação não estão em ordem crescente")

        if not test_dates.is_monotonic_increasing:
            passed = False
            issues.append("Datas de teste não estão em ordem crescente")

        # Resultado
        if passed:
            print("   ✅ PASSOU: Ordem temporal está correta")
            print(f"      Treino:    {train_dates.min()} a {train_dates.max()}")
            print(f"      Validação: {val_dates.min()} a {val_dates.max()}")
            print(f"      Teste:     {test_dates.min()} a {test_dates.max()}")
        else:
            print("   ❌ FALHOU: Problemas de ordem temporal detectados!")
            for issue in issues:
                print(f"      - {issue}")

        self.test_results['temporal_order'] = {'passed': passed, 'issues': issues}

        return passed

    def test_no_overlap(
        self,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray
    ) -> bool:
        """
        Testa se não há sobreposição entre os conjuntos.

        ✅ TESTE 2: Sem Sobreposição

        Args:
            train_indices: Índices do treino
            val_indices: Índices da validação
            test_indices: Índices do teste

        Returns:
            True se passou no teste
        """
        print("\n🧪 TESTE 2: Verificando Sobreposição de Dados...")

        passed = True
        issues = []

        # Converter para sets
        train_set = set(train_indices)
        val_set = set(val_indices)
        test_set = set(test_indices)

        # Verificar interseções
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set

        if train_val_overlap:
            passed = False
            issues.append(f"Sobreposição Treino-Validação: {len(train_val_overlap)} amostras")

        if train_test_overlap:
            passed = False
            issues.append(f"Sobreposição Treino-Teste: {len(train_test_overlap)} amostras")

        if val_test_overlap:
            passed = False
            issues.append(f"Sobreposição Validação-Teste: {len(val_test_overlap)} amostras")

        # Resultado
        if passed:
            print("   ✅ PASSOU: Nenhuma sobreposição detectada")
        else:
            print("   ❌ FALHOU: Sobreposição detectada!")
            for issue in issues:
                print(f"      - {issue}")

        self.test_results['no_overlap'] = {'passed': passed, 'issues': issues}

        return passed

    def test_suspicious_performance(
        self,
        r2_score: float,
        mape: float,
        max_acceptable_r2: float = 0.95,
        min_acceptable_mape: float = 0.1
    ) -> bool:
        """
        Testa se a performance é suspeita (muito boa = possível leakage).

        ✅ TESTE 3: Performance Suspeita

        Args:
            r2_score: R² do modelo
            mape: MAPE do modelo
            max_acceptable_r2: R² máximo aceitável (padrão: 0.95)
            min_acceptable_mape: MAPE mínimo aceitável (padrão: 0.1%)

        Returns:
            True se passou no teste
        """
        print("\n🧪 TESTE 3: Verificando Performance Suspeita...")

        passed = True
        issues = []

        # R² muito alto é suspeito
        if r2_score > max_acceptable_r2:
            passed = False
            issues.append(
                f"R² muito alto ({r2_score:.4f} > {max_acceptable_r2}). "
                f"Para séries temporais financeiras, R² > 0.95 é extremamente suspeito."
            )

        # MAPE muito baixo é suspeito
        if mape < min_acceptable_mape:
            passed = False
            issues.append(
                f"MAPE muito baixo ({mape:.4f}% < {min_acceptable_mape}%). "
                f"MAPE < 0.1% em dados financeiros é suspeito."
            )

        # Resultado
        if passed:
            print("   ✅ PASSOU: Performance está em níveis realistas")
            print(f"      R²:   {r2_score:.4f} (limite: {max_acceptable_r2})")
            print(f"      MAPE: {mape:.4f}% (limite mín: {min_acceptable_mape}%)")
        else:
            print("   ⚠️  ALERTA: Performance suspeitamente boa!")
            print("      Isso pode indicar data leakage!")
            for issue in issues:
                print(f"      - {issue}")

        self.test_results['suspicious_performance'] = {'passed': passed, 'issues': issues}

        # No strict mode, este teste é apenas warning
        return passed if self.strict_mode else True

    def test_better_than_baseline(
        self,
        model_rmse: float,
        baseline_rmse: float,
        min_improvement: float = 0.05
    ) -> bool:
        """
        Testa se o modelo é melhor que o baseline.

        ✅ TESTE 4: Comparação com Baseline

        Args:
            model_rmse: RMSE do modelo
            baseline_rmse: RMSE do baseline (naive)
            min_improvement: Melhoria mínima exigida (5% padrão)

        Returns:
            True se passou no teste
        """
        print("\n🧪 TESTE 4: Comparando com Baseline...")

        improvement = (baseline_rmse - model_rmse) / baseline_rmse

        passed = improvement >= min_improvement

        # Resultado
        if passed:
            print(f"   ✅ PASSOU: Modelo é {improvement*100:.2f}% melhor que baseline")
            print(f"      Modelo RMSE:   {model_rmse:.4f}")
            print(f"      Baseline RMSE: {baseline_rmse:.4f}")
        else:
            print(f"   ❌ FALHOU: Modelo NÃO superou baseline!")
            print(f"      Modelo RMSE:   {model_rmse:.4f}")
            print(f"      Baseline RMSE: {baseline_rmse:.4f}")
            print(f"      Melhoria:      {improvement*100:.2f}% (mínimo: {min_improvement*100:.2f}%)")

        self.test_results['better_than_baseline'] = {
            'passed': passed,
            'improvement': improvement
        }

        return passed

    def test_scaler_fit_on_train_only(
        self,
        train_data_sample: np.ndarray,
        val_data_sample: np.ndarray,
        scaler
    ) -> bool:
        """
        Testa se o scaler foi fitado apenas no treino.

        ✅ TESTE 5: Scaler Fitado Apenas no Treino

        Args:
            train_data_sample: Amostra de dados de treino (antes de normalizar)
            val_data_sample: Amostra de dados de validação (antes de normalizar)
            scaler: Scaler utilizado

        Returns:
            True se passou no teste
        """
        print("\n🧪 TESTE 5: Verificando Scaler...")

        # Verificar que min/max do scaler correspondem ao treino
        train_min = train_data_sample.min(axis=0)
        train_max = train_data_sample.max(axis=0)

        val_min = val_data_sample.min(axis=0)
        val_max = val_data_sample.max(axis=0)

        scaler_min = scaler.data_min_
        scaler_max = scaler.data_max_

        passed = True
        issues = []

        # O scaler deve ter sido fitado no treino, então:
        # - scaler_min deve ser próximo de train_min
        # - scaler_max deve ser próximo de train_max
        # - Se val tem valores fora do range do treino, é normal

        # Verificar se scaler foi fitado nos dados de validação (ERRADO!)
        # Se scaler_min ou scaler_max estão fora do range do treino,
        # mas dentro do range de val, pode indicar leakage

        for i in range(len(scaler_min)):
            # Se scaler min é menor que train min, mas igual a val min, é suspeito
            if scaler_min[i] < train_min[i] - 0.01:
                if abs(scaler_min[i] - val_min[i]) < 0.01:
                    passed = False
                    issues.append(
                        f"Feature {i}: scaler_min ({scaler_min[i]:.4f}) corresponde a "
                        f"val_min ({val_min[i]:.4f}), não train_min ({train_min[i]:.4f})"
                    )

            # Se scaler max é maior que train max, mas igual a val max, é suspeito
            if scaler_max[i] > train_max[i] + 0.01:
                if abs(scaler_max[i] - val_max[i]) < 0.01:
                    passed = False
                    issues.append(
                        f"Feature {i}: scaler_max ({scaler_max[i]:.4f}) corresponde a "
                        f"val_max ({val_max[i]:.4f}), não train_max ({train_max[i]:.4f})"
                    )

        # Resultado
        if passed:
            print("   ✅ PASSOU: Scaler foi fitado corretamente no treino")
        else:
            print("   ❌ FALHOU: Scaler pode ter usado dados de validação!")
            for issue in issues:
                print(f"      - {issue}")

        self.test_results['scaler_fit_train_only'] = {'passed': passed, 'issues': issues}

        # Este teste pode ter falsos positivos, então no strict mode apenas warning
        return passed if self.strict_mode else True

    def run_all_tests(
        self,
        train_dates: Optional[pd.DatetimeIndex] = None,
        val_dates: Optional[pd.DatetimeIndex] = None,
        test_dates: Optional[pd.DatetimeIndex] = None,
        train_indices: Optional[np.ndarray] = None,
        val_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        model_metrics: Optional[Dict[str, float]] = None,
        baseline_metrics: Optional[Dict[str, float]] = None,
        scaler: Optional[Any] = None,
        train_data_sample: Optional[np.ndarray] = None,
        val_data_sample: Optional[np.ndarray] = None
    ) -> bool:
        """
        Executa todos os testes disponíveis.

        Args:
            (vários parâmetros opcionais para cada teste)

        Returns:
            True se passou em todos os testes
        """
        print("\n" + "="*60)
        print("🔍 EXECUTANDO TESTES ANTI-DATA-LEAKAGE")
        print("="*60)

        all_passed = True

        # Teste 1: Ordem temporal
        if train_dates is not None and val_dates is not None and test_dates is not None:
            passed = self.test_temporal_order(train_dates, val_dates, test_dates)
            all_passed = all_passed and passed

        # Teste 2: Sobreposição
        if train_indices is not None and val_indices is not None and test_indices is not None:
            passed = self.test_no_overlap(train_indices, val_indices, test_indices)
            all_passed = all_passed and passed

        # Teste 3: Performance suspeita
        if model_metrics is not None:
            passed = self.test_suspicious_performance(
                r2_score=model_metrics.get('R2', 0),
                mape=model_metrics.get('MAPE', 100)
            )
            all_passed = all_passed and passed

        # Teste 4: Melhor que baseline
        if model_metrics is not None and baseline_metrics is not None:
            passed = self.test_better_than_baseline(
                model_rmse=model_metrics.get('RMSE', float('inf')),
                baseline_rmse=baseline_metrics.get('RMSE', 0)
            )
            all_passed = all_passed and passed

        # Teste 5: Scaler
        if scaler is not None and train_data_sample is not None and val_data_sample is not None:
            passed = self.test_scaler_fit_on_train_only(
                train_data_sample, val_data_sample, scaler
            )
            all_passed = all_passed and passed

        # Resultado final
        print("\n" + "="*60)
        if all_passed:
            print("✅ TODOS OS TESTES PASSARAM!")
            print("   Não foram detectados sinais de data leakage.")
        else:
            print("❌ ALGUNS TESTES FALHARAM!")
            print("   Revise o pipeline para garantir ausência de data leakage.")
        print("="*60)

        return all_passed

    def get_report(self) -> Dict[str, Any]:
        """
        Retorna relatório dos testes.

        Returns:
            Dicionário com resultados
        """
        return self.test_results


if __name__ == "__main__":
    # Teste do módulo
    print("\n" + "="*60)
    print("TESTE DO MÓDULO ANTI_LEAKAGE_TESTS")
    print("="*60)

    # Criar datas de teste
    train_dates = pd.date_range('2020-01-01', periods=700, freq='D')
    val_dates = pd.date_range('2021-11-20', periods=150, freq='D')
    test_dates = pd.date_range('2022-04-19', periods=150, freq='D')

    # Criar índices
    train_indices = np.arange(700)
    val_indices = np.arange(700, 850)
    test_indices = np.arange(850, 1000)

    # Métricas simuladas
    model_metrics = {
        'R2': 0.75,
        'RMSE': 10.5,
        'MAPE': 2.3
    }

    baseline_metrics = {
        'RMSE': 15.2
    }

    # Criar validador
    validator = AntiLeakageValidator(strict_mode=True)

    # Executar testes
    passed = validator.run_all_tests(
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics
    )

    # Obter relatório
    report = validator.get_report()
    print("\n📋 Relatório dos testes:")
    for test_name, result in report.items():
        print(f"\n{test_name}: {'✅' if result['passed'] else '❌'}")

    print("\n✅ Teste do módulo concluído!")
