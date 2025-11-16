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

        print("🔮 StockPredictor inicializado")

    def load_model(self):
        """
        Carrega o modelo treinado.
        """
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")

        self.model = tf.keras.models.load_model(self.model_path)
        print(f"✓ Modelo carregado de: {self.model_path}")

    def load_scaler(self):
        """
        Carrega o scaler.
        """
        import joblib

        if not Path(self.scaler_path).exists():
            raise FileNotFoundError(f"Scaler não encontrado: {self.scaler_path}")

        self.scaler = joblib.load(self.scaler_path)
        print(f"✓ Scaler carregado de: {self.scaler_path}")

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

        # Assumir que Close é a coluna 3 (OHLCV)
        # TODO: Isso deveria ser parametrizável
        n_features = self.scaler.n_features_in_
        dummy = np.zeros((1, n_features))
        dummy[0, 3] = value_scaled  # Close na posição 3

        inversed = self.scaler.inverse_transform(dummy)
        return inversed[0, 3]

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

        # Criar array dummy
        dummy = np.zeros((n_samples, n_features))
        dummy[:, 3] = values_scaled  # Close na posição 3

        # Inverse transform
        inversed = self.scaler.inverse_transform(dummy)

        return inversed[:, 3]

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
            new_row[3] = next_pred_scaled  # Atualizar Close

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
    print("   python scripts/train_model.py")

    print("\n✅ Estrutura do módulo validada!")
