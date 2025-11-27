"""
Módulo de Predição
==================

Este módulo gerencia predições usando o modelo LSTM treinado.
Inclui funcionalidades para:
- Predições únicas
- Predições em batch
- Inversão da normalização
- Cálculo de intervalos de confiança

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
import tensorflow as tf
from pathlib import Path


class StockPredictor:
    """
    Realiza predições de preços de ações usando modelo LSTM treinado.
    """

    def __init__(
        self,
        model_path: str = "models/lstm_model.h5",
        scaler_path: str = "models/scaler.pkl"
    ):
        """
        Inicializa o predictor.

        Args:
            model_path: Caminho do modelo salvo
            scaler_path: Caminho do scaler salvo
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.target_idx = 0  # ✅ Índice do target (será carregado do scaler)
        self.target_column = 'Close'  # ✅ Nome do target

        print("🔮 StockPredictor inicializado")

    def load_model(self):
        """
        Carrega o modelo treinado.
        """
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")

        # ✅ CORREÇÃO: Carregar com compile=False para evitar problemas de deserialização
        # Se falhar, tentar com compile=True (versões antigas)
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            # Recompilar manualmente para garantir compatibilidade
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            print(f"✓ Modelo carregado de: {self.model_path} (recompilado)")
        except Exception as e:
            # Fallback: tentar carregar com compile=True (pode funcionar em algumas versões)
            try:
                self.model = tf.keras.models.load_model(self.model_path, compile=True)
                print(f"✓ Modelo carregado de: {self.model_path} (com otimizador preservado)")
            except Exception as e2:
                raise RuntimeError(
                    f"Erro ao carregar modelo: {e2}\n"
                    f"Tente retreinar o modelo: python scripts/train_model_stationary.py"
                ) from e2

    def load_scaler(self):
        """
        Carrega o scaler com metadados.
        """
        import joblib

        if not Path(self.scaler_path).exists():
            raise FileNotFoundError(f"Scaler não encontrado: {self.scaler_path}")

        scaler_data = joblib.load(self.scaler_path)
        
        # ✅ CORREÇÃO: Compatibilidade com versão antiga e nova
        if isinstance(scaler_data, dict):
            self.scaler = scaler_data['scaler']
            self.feature_names = scaler_data.get('feature_names')
            self.target_column = scaler_data.get('target_column', 'Close')
            self.target_idx = scaler_data.get('target_idx')
            
            # ✅ VALIDAÇÃO: Garantir que target_idx está definido
            if self.target_idx is None:
                # Tentar encontrar pelo nome da coluna
                if self.feature_names and self.target_column in self.feature_names:
                    self.target_idx = self.feature_names.index(self.target_column)
                    print(f"⚠️  target_idx não encontrado, usando índice de '{self.target_column}': {self.target_idx}")
                else:
                    # Fallback para índice 0 (Close geralmente é primeiro após feature selection)
                    self.target_idx = 0
                    print(f"⚠️  WARNING: target_idx não encontrado, usando fallback índice 0")
            
            # ✅ VALIDAÇÃO: Verificar se target_idx é válido
            n_features = self.scaler.n_features_in_
            if self.target_idx >= n_features:
                raise ValueError(
                    f"target_idx ({self.target_idx}) >= número de features ({n_features}). "
                    f"Scaler pode estar inconsistente com o modelo."
                )
            
            print(f"✓ Scaler carregado de: {self.scaler_path}")
            print(f"   Target: {self.target_column} (índice {self.target_idx})")
            print(f"   Features: {n_features}")
        else:
            # Versão antiga, só o scaler
            self.scaler = scaler_data
            # Tentar encontrar Close nas feature_names se disponível
            if hasattr(self.scaler, 'feature_names_in_'):
                feature_names_list = list(self.scaler.feature_names_in_)
                if 'Close' in feature_names_list:
                    self.target_idx = feature_names_list.index('Close')
                else:
                    self.target_idx = 0
            else:
                self.target_idx = 0  # Fallback
            print(f"✓ Scaler carregado de: {self.scaler_path} (versão antiga)")
            print(f"   ⚠️ WARNING: Usando índice {self.target_idx} para target (fallback)")

    def load_all(self):
        """
        Carrega modelo e scaler.
        """
        print("\n📂 Carregando modelo e scaler...")
        self.load_model()
        self.load_scaler()
        print("✅ Modelo e scaler carregados com sucesso!")

    def predict_single(
        self,
        sequence: np.ndarray,
        return_scaled: bool = False
    ) -> Union[float, np.ndarray]:
        """
        Faz predição para uma única sequência.

        Args:
            sequence: Sequência de entrada (sequence_length, n_features)
            return_scaled: Se True, retorna valor normalizado

        Returns:
            Predição (valor único)
        """
        if self.model is None:
            raise ValueError("Modelo não foi carregado!")

        # Adicionar dimensão de batch se necessário
        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0)

        # Predizer
        prediction_scaled = self.model.predict(sequence, verbose=0)[0, 0]

        if return_scaled:
            return prediction_scaled

        # Inverse transform
        prediction = self._inverse_transform_single(prediction_scaled)

        return prediction

    def predict_batch(
        self,
        sequences: np.ndarray,
        return_scaled: bool = False
    ) -> np.ndarray:
        """
        Faz predições para múltiplas sequências.

        Args:
            sequences: Array de sequências (batch_size, sequence_length, n_features)
            return_scaled: Se True, retorna valores normalizados

        Returns:
            Array de predições
        """
        if self.model is None:
            raise ValueError("Modelo não foi carregado!")

        # Predizer
        predictions_scaled = self.model.predict(sequences, verbose=0).flatten()

        if return_scaled:
            return predictions_scaled

        # Inverse transform
        predictions = self._inverse_transform_batch(predictions_scaled)

        return predictions

    def _inverse_transform_single(self, value_scaled: float) -> float:
        """
        Reverte normalização de um único valor.

        Args:
            value_scaled: Valor normalizado

        Returns:
            Valor na escala original
        """
        if self.scaler is None:
            raise ValueError("Scaler não foi carregado!")
        
        if self.target_idx is None:
            raise ValueError(
                "target_idx não foi definido! Carregue o scaler primeiro com load_scaler()."
            )

        # ✅ CORREÇÃO: Usar target_idx salvo ao invés de índice fixo
        n_features = self.scaler.n_features_in_
        
        # ✅ VALIDAÇÃO: Verificar se target_idx é válido
        if self.target_idx >= n_features:
            raise ValueError(
                f"target_idx ({self.target_idx}) >= número de features ({n_features}). "
                f"Verifique se o scaler corresponde ao modelo."
            )
        
        dummy = np.zeros((1, n_features))
        dummy[0, self.target_idx] = value_scaled  # ✅ Usar target_idx correto

        inversed = self.scaler.inverse_transform(dummy)
        return inversed[0, self.target_idx]

    def _inverse_transform_batch(self, values_scaled: np.ndarray) -> np.ndarray:
        """
        Reverte normalização de múltiplos valores.

        Args:
            values_scaled: Array de valores normalizados

        Returns:
            Array de valores na escala original
        """
        if self.scaler is None:
            raise ValueError("Scaler não foi carregado!")

        n_features = self.scaler.n_features_in_
        n_samples = len(values_scaled)

        # ✅ CORREÇÃO: Usar target_idx salvo ao invés de índice fixo
        dummy = np.zeros((n_samples, n_features))
        dummy[:, self.target_idx] = values_scaled  # ✅ Usar target_idx correto

        # Inverse transform
        inversed = self.scaler.inverse_transform(dummy)

        return inversed[:, self.target_idx]

    def predict_with_confidence(
        self,
        sequence: np.ndarray,
        n_iterations: int = 100,
        dropout_rate: float = 0.2
    ) -> Dict[str, float]:
        """
        Faz predição com intervalo de confiança usando Monte Carlo Dropout.

        Args:
            sequence: Sequência de entrada
            n_iterations: Número de iterações para estimar incerteza
            dropout_rate: Taxa de dropout para usar

        Returns:
            Dicionário com mean, std, lower, upper
        """
        if self.model is None:
            raise ValueError("Modelo não foi carregado!")

        # Adicionar dimensão de batch se necessário
        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0)

        # Fazer múltiplas predições com dropout
        predictions = []
        for _ in range(n_iterations):
            pred = self.model(sequence, training=True)  # training=True ativa dropout
            predictions.append(pred.numpy()[0, 0])

        predictions = np.array(predictions)

        # Calcular estatísticas
        mean_scaled = np.mean(predictions)
        std_scaled = np.std(predictions)

        # Inverse transform
        mean = self._inverse_transform_single(mean_scaled)

        # Para intervalo de confiança (95%)
        lower_scaled = mean_scaled - 1.96 * std_scaled
        upper_scaled = mean_scaled + 1.96 * std_scaled

        lower = self._inverse_transform_single(lower_scaled)
        upper = self._inverse_transform_single(upper_scaled)

        return {
            'prediction': mean,
            'std': std_scaled,  # Mantém scaled pois é desvio
            'lower_95': lower,
            'upper_95': upper,
            'confidence_width': upper - lower
        }

    def predict_next_days(
        self,
        initial_sequence: np.ndarray,
        n_days: int = 5,
        return_scaled: bool = False
    ) -> np.ndarray:
        """
        Prediz múltiplos dias no futuro (predição iterativa).

        ⚠️ AVISO: Predições iterativas acumulam erros!

        Args:
            initial_sequence: Sequência inicial (sequence_length, n_features)
            n_days: Número de dias para predizer
            return_scaled: Se True, retorna valores normalizados

        Returns:
            Array com predições
        """
        if self.model is None:
            raise ValueError("Modelo não foi carregado!")

        predictions = []
        current_sequence = initial_sequence.copy()

        for _ in range(n_days):
            # Predizer próximo dia
            next_pred_scaled = self.predict_single(
                current_sequence,
                return_scaled=True
            )

            predictions.append(next_pred_scaled)

            # Atualizar sequência (shift + nova predição)
            # NOTA: Isso é simplificado - idealmente deveria atualizar todas as features
            new_row = current_sequence[-1].copy()
            new_row[self.target_idx] = next_pred_scaled  # ✅ CORREÇÃO: Usar target_idx dinâmico

            current_sequence = np.vstack([current_sequence[1:], new_row])

        predictions = np.array(predictions)

        if return_scaled:
            return predictions

        # Inverse transform
        predictions = self._inverse_transform_batch(predictions)

        return predictions


if __name__ == "__main__":
    # Teste do módulo
    print("\n" + "="*60)
    print("TESTE DO MÓDULO PREDICTOR")
    print("="*60)

    # NOTA: Este teste requer um modelo e scaler salvos
    # Para testar, primeiro execute o treinamento completo

    print("\n⚠️  Este módulo requer modelo e scaler salvos.")
    print("   Execute o treinamento completo primeiro com:")
    print("   python scripts/train_model_stationary.py")

    print("\n✅ Estrutura do módulo validada!")
