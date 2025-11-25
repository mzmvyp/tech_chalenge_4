"""
Módulo de Preprocessamento com Proteção Anti-Leakage
=====================================================

Este módulo implementa o preprocessamento dos dados com proteções rigorosas contra data leakage.

⚠️ CRÍTICO - PROTEÇÕES ANTI-DATA-LEAKAGE:
==========================================

1. SPLIT TEMPORAL PRIMEIRO
   - O split SEMPRE acontece ANTES de qualquer processamento
   - Treino, validação e teste são separados temporalmente
   - NUNCA usar shuffle ou KFold

2. NORMALIZAÇÃO CORRETA
   - Scaler.fit() APENAS no conjunto de treino
   - Scaler.transform() em validação e teste
   - NUNCA usar fit_transform() em val/test

3. FEATURES SEM INFORMAÇÃO FUTURA
   - Nenhuma feature pode usar informação do futuro
   - Médias móveis e indicadores calculados corretamente
   - Validações automáticas implementadas

4. SEQUÊNCIAS TEMPORAIS CORRETAS
   - Sequências respeitam ordem temporal
   - Não misturar dados de períodos diferentes

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, Dict
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class TimeSeriesPreprocessor:
    """
    Preprocessador para séries temporais com proteção anti-leakage.

    IMPORTANTE: Este preprocessador garante que não haja vazamento de informação
    do futuro para o passado (data leakage).
    """

    def __init__(
        self,
        sequence_length: int = 60,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        """
        Inicializa o preprocessador.

        Args:
            sequence_length: Número de dias para criar sequências
            train_ratio: Proporção de dados para treino
            val_ratio: Proporção de dados para validação
            test_ratio: Proporção de dados para teste
        """
        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Validar que os ratios somam 1.0
        total = train_ratio + val_ratio + test_ratio
        if not abs(total - 1.0) < 0.001:
            raise ValueError(f"Ratios devem somar 1.0. Atual: {total}")

        # Scaler será fitado apenas no treino
        self.scaler = None
        self.feature_names = None
        self.target_column = None  # Nome da coluna target
        self.target_idx = None  # Índice da coluna target (CRÍTICO!)
        self.split_info = {}  # Guardar informações sobre o split

        print(f"⚙️  Preprocessador inicializado:")
        print(f"   Sequence Length: {sequence_length}")
        print(f"   Split: {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}")

    def temporal_split(
        self,
        df: pd.DataFrame,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Realiza split temporal dos dados.

        ⚠️ ANTI-LEAKAGE: O split é TEMPORAL, sem shuffle!

        Args:
            df: DataFrame com índice temporal
            verbose: Se True, exibe informações do split

        Returns:
            Tupla (df_train, df_val, df_test)
        """
        if verbose:
            print("\n" + "="*60)
            print("📊 REALIZANDO SPLIT TEMPORAL")
            print("="*60)

        # Garantir que está ordenado temporalmente
        if not df.index.is_monotonic_increasing:
            print("⚠️  Dados fora de ordem temporal. Ordenando...")
            df = df.sort_index()

        n = len(df)

        # Calcular índices de corte
        train_end_idx = int(n * self.train_ratio)
        val_end_idx = int(n * (self.train_ratio + self.val_ratio))

        # Split temporal
        df_train = df.iloc[:train_end_idx].copy()
        df_val = df.iloc[train_end_idx:val_end_idx].copy()
        df_test = df.iloc[val_end_idx:].copy()

        # Guardar informações do split
        self.split_info = {
            'total_samples': n,
            'train_samples': len(df_train),
            'val_samples': len(df_val),
            'test_samples': len(df_test),
            'train_period': (df_train.index[0], df_train.index[-1]),
            'val_period': (df_val.index[0], df_val.index[-1]),
            'test_period': (df_test.index[0], df_test.index[-1]),
        }

        if verbose:
            print(f"\n✓ Split realizado com sucesso!")
            print(f"\n📈 TREINO: {len(df_train)} amostras ({len(df_train)/n:.1%})")
            print(f"   Período: {df_train.index[0].date()} a {df_train.index[-1].date()}")
            print(f"\n📊 VALIDAÇÃO: {len(df_val)} amostras ({len(df_val)/n:.1%})")
            print(f"   Período: {df_val.index[0].date()} a {df_val.index[-1].date()}")
            print(f"\n📉 TESTE: {len(df_test)} amostras ({len(df_test)/n:.1%})")
            print(f"   Período: {df_test.index[0].date()} a {df_test.index[-1].date()}")

            # ✅ VALIDAÇÃO ANTI-LEAKAGE: Verificar que não há sobreposição temporal
            assert df_train.index[-1] < df_val.index[0], "ERRO: Sobreposição entre treino e validação!"
            assert df_val.index[-1] < df_test.index[0], "ERRO: Sobreposição entre validação e teste!"
            print("\n✅ VALIDAÇÃO ANTI-LEAKAGE: Sem sobreposição temporal detectada!")

        return df_train, df_val, df_test

    def fit_scaler(self, df_train: pd.DataFrame) -> np.ndarray:
        """
        Fita o scaler APENAS nos dados de treino.

        ⚠️ ANTI-LEAKAGE: Scaler é fitado APENAS no treino!

        Args:
            df_train: DataFrame de treino

        Returns:
            Array com dados de treino normalizados
        """
        print("\n🔧 Fitando scaler nos dados de TREINO...")

        # Armazenar nomes das features
        self.feature_names = df_train.columns.tolist()

        # Criar e fitar scaler APENAS no treino
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = self.scaler.fit_transform(df_train.values)

        print(f"✓ Scaler fitado com {len(self.feature_names)} features:")
        print(f"  {self.feature_names}")
        print(f"✓ Dados de treino normalizados: {train_scaled.shape}")

        return train_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforma dados usando o scaler já fitado.

        ⚠️ ANTI-LEAKAGE: Usa apenas transform(), NUNCA fit_transform()!

        Args:
            df: DataFrame para transformar

        Returns:
            Array com dados normalizados
        """
        if self.scaler is None:
            raise ValueError("Scaler não foi fitado! Execute fit_scaler() primeiro.")

        # Verificar que as colunas são as mesmas
        if df.columns.tolist() != self.feature_names:
            raise ValueError(
                f"Colunas não correspondem!\n"
                f"Esperado: {self.feature_names}\n"
                f"Recebido: {df.columns.tolist()}"
            )

        # Transform (sem fit!)
        scaled = self.scaler.transform(df.values)

        return scaled

    def create_sequences(
        self,
        data: np.ndarray,
        target_column_idx: Optional[int] = None  # Opcional - usa self.target_idx se disponível
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria sequências temporais para LSTM.

        ⚠️ ANTI-LEAKAGE: Sequências respeitam ordem temporal!

        Args:
            data: Array com dados normalizados
            target_column_idx: Índice da coluna alvo (OBRIGATÓRIO se não foi salvo)

        Returns:
            Tupla (X, y) onde:
            - X: Sequências de entrada (samples, sequence_length, features)
            - y: Valores alvo (samples,)
        """
        # ✅ VALIDAÇÃO: Usar target_idx salvo OU parâmetro
        if target_column_idx is None:
            if self.target_idx is None:
                raise ValueError(
                    "target_column_idx não foi especificado e não há target_idx salvo. "
                    "Chame prepare_data() primeiro ou passe target_column_idx explicitamente."
                )
            target_column_idx = self.target_idx

        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            # Sequência de entrada: últimos 'sequence_length' dias
            X.append(data[i - self.sequence_length:i])

            # Alvo: valor de fechamento do próximo dia
            y.append(data[i, target_column_idx])

        X = np.array(X)
        y = np.array(y)

        print(f"\n📦 Sequências criadas:")
        print(f"   X shape: {X.shape} (samples, sequence_length, features)")
        print(f"   y shape: {y.shape} (samples,)")

        return X, y

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'Close',
        verbose: bool = True
    ) -> Dict:
        """
        Pipeline completo de preparação de dados COM PROTEÇÃO ANTI-LEAKAGE.

        ⚠️ ORDEM CRÍTICA (NUNCA MUDAR):
        1. Split temporal PRIMEIRO
        2. Fit scaler APENAS no treino
        3. Transform em todos os conjuntos
        4. Criar sequências

        Args:
            df: DataFrame com todas as features
            target_column: Nome da coluna alvo
            verbose: Se True, exibe informações detalhadas

        Returns:
            Dicionário com todos os conjuntos preparados
        """
        if verbose:
            print("\n" + "="*60)
            print("🔧 INICIANDO PIPELINE DE PREPROCESSAMENTO")
            print("="*60)
            print(f"📊 Dataset original: {df.shape}")
            print(f"🎯 Coluna alvo: {target_column}")

        # ============================================
        # PASSO 0: SALVAR INFORMAÇÕES DO TARGET
        # ============================================
        self.target_column = target_column
        self.target_idx = df.columns.tolist().index(target_column)

        if verbose:
            print(f"\n🎯 Target configurado:")
            print(f"   Coluna: {target_column}")
            print(f"   Índice: {self.target_idx}")

        # ============================================
        # PASSO 1: SPLIT TEMPORAL PRIMEIRO
        # ============================================
        df_train, df_val, df_test = self.temporal_split(df, verbose=verbose)

        # ============================================
        # PASSO 2: FIT SCALER APENAS NO TREINO
        # ============================================
        train_scaled = self.fit_scaler(df_train)

        # ============================================
        # PASSO 3: TRANSFORM EM VAL E TEST
        # ============================================
        print("\n🔄 Transformando dados de VALIDAÇÃO...")
        val_scaled = self.transform(df_val)
        print(f"✓ Dados de validação normalizados: {val_scaled.shape}")

        print("\n🔄 Transformando dados de TESTE...")
        test_scaled = self.transform(df_test)
        print(f"✓ Dados de teste normalizados: {test_scaled.shape}")

        # ============================================
        # PASSO 4: CRIAR SEQUÊNCIAS
        # ============================================
        target_idx = df.columns.tolist().index(target_column)

        print("\n📦 Criando sequências de TREINO...")
        X_train, y_train = self.create_sequences(train_scaled, target_idx)

        print("\n📦 Criando sequências de VALIDAÇÃO...")
        X_val, y_val = self.create_sequences(val_scaled, target_idx)

        print("\n📦 Criando sequências de TESTE...")
        X_test, y_test = self.create_sequences(test_scaled, target_idx)

        # ============================================
        # RESULTADO FINAL
        # ============================================
        result = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'target_column': self.target_column,  # Atualizado
            'target_idx': self.target_idx,  # NOVO - Índice do target
            'split_info': self.split_info,
        }

        if verbose:
            print("\n" + "="*60)
            print("✅ PREPROCESSAMENTO CONCLUÍDO")
            print("="*60)
            print(f"\n📊 Conjuntos finais:")
            print(f"   TREINO:     X={X_train.shape}, y={y_train.shape}")
            print(f"   VALIDAÇÃO:  X={X_val.shape}, y={y_val.shape}")
            print(f"   TESTE:      X={X_test.shape}, y={y_test.shape}")

        return result

    def save_scaler(self, filepath: str = "models/scaler.pkl"):
        """
        Salva o scaler treinado com metadados (CORREÇÃO OPUS).

        Args:
            filepath: Caminho para salvar o scaler
        """
        if self.scaler is None:
            raise ValueError("Scaler não foi treinado ainda!")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Salvar scaler com metadados (CORREÇÃO OPUS)
        scaler_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'target_idx': self.target_idx
        }

        joblib.dump(scaler_data, filepath)
        print(f"\n💾 Scaler e metadados salvos em: {filepath}")
        print(f"   Features: {len(self.feature_names) if self.feature_names else 0}")
        print(f"   Target: {self.target_column} (índice {self.target_idx})")

    def load_scaler(self, filepath: str = "models/scaler.pkl"):
        """
        Carrega um scaler salvo com metadados (CORREÇÃO OPUS).

        Args:
            filepath: Caminho do scaler salvo
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Scaler não encontrado: {filepath}")

        scaler_data = joblib.load(filepath)

        # Compatibilidade com versão antiga (CORREÇÃO OPUS)
        if isinstance(scaler_data, dict):
            self.scaler = scaler_data['scaler']
            self.feature_names = scaler_data.get('feature_names')
            self.target_column = scaler_data.get('target_column', 'Close')
            self.target_idx = scaler_data.get('target_idx')
            print(f"\n📂 Scaler e metadados carregados de: {filepath}")
            print(f"   Features: {len(self.feature_names) if self.feature_names else 0}")
            print(f"   Target: {self.target_column} (índice {self.target_idx})")
        else:
            # Versão antiga, só o scaler
            self.scaler = scaler_data
            print(f"\n📂 Scaler carregado de: {filepath} (versão antiga, sem metadados)")
            print(f"   ⚠️ WARNING: Metadados não disponíveis. Re-treine o modelo.")

    def inverse_transform_target(self, y_scaled: np.ndarray, target_column: str = 'Close') -> np.ndarray:
        """
        Reverte a normalização apenas da coluna alvo.

        Args:
            y_scaled: Array com valores normalizados
            target_column: Nome da coluna alvo (opcional se target_idx já está salvo)

        Returns:
            Array com valores na escala original
        """
        if self.scaler is None:
            raise ValueError("Scaler não foi treinado!")

        # Usar target_idx salvo ou buscar pelo nome (CORREÇÃO OPUS)
        if self.target_idx is not None:
            target_idx = self.target_idx
        elif self.feature_names and target_column in self.feature_names:
            target_idx = self.feature_names.index(target_column)
        else:
            # Fallback para posição padrão do Close em OHLCV
            # NOTA: Isso é um fallback de segurança, mas não deveria ser necessário
            target_idx = 3
            print(f"⚠️ WARNING: Usando índice padrão {target_idx} para {target_column}")

        n_features = len(self.feature_names) if self.feature_names else self.scaler.n_features_in_

        # Reshape se necessário
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)

        # Criar array completo
        dummy = np.zeros((len(y_scaled), n_features))
        dummy[:, target_idx] = y_scaled.flatten()

        # Inverse transform
        inversed = self.scaler.inverse_transform(dummy)

        # Retornar apenas a coluna do target
        return inversed[:, target_idx]


if __name__ == "__main__":
    # Teste do módulo
    print("\n" + "="*60)
    print("TESTE DO MÓDULO PREPROCESSOR")
    print("="*60)

    # Criar dados de exemplo
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    df_test = pd.DataFrame({
        'Open': np.random.randn(1000).cumsum() + 100,
        'High': np.random.randn(1000).cumsum() + 102,
        'Low': np.random.randn(1000).cumsum() + 98,
        'Close': np.random.randn(1000).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 1000),
    }, index=dates)

    # Criar preprocessador
    preprocessor = TimeSeriesPreprocessor(
        sequence_length=60,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # Preparar dados
    data = preprocessor.prepare_data(df_test, target_column='Close')

    # Testar inverse transform
    print("\n🔄 Testando inverse transform...")
    y_pred_scaled = data['y_test'][:10]
    y_pred_original = preprocessor.inverse_transform_target(y_pred_scaled)
    print(f"✓ Valores normalizados: {y_pred_scaled[:5]}")
    print(f"✓ Valores originais: {y_pred_original[:5]}")

    print("\n✅ Teste concluído com sucesso!")
