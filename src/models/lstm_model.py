"""
Módulo de Arquitetura LSTM
===========================

Este módulo define a arquitetura da rede neural LSTM para predição de preços de ações.

Arquitetura:
- Múltiplas camadas LSTM com Dropout
- Camadas Dense para output
- Otimizado para séries temporais financeiras

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2  # CORREÇÃO: Adicionar regularização
from typing import Dict, List, Any, Optional
import numpy as np


class LSTMStockPredictor:
    """
    Modelo LSTM para predição de preços de ações.
    """

    def __init__(
        self,
        sequence_length: int,
        n_features: int,
        lstm_layers: List[Dict[str, Any]],
        dense_layers: List[Dict[str, Any]],
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'mse',
        metrics: List[str] = ['mae']
    ):
        """
        Inicializa o modelo LSTM.

        Args:
            sequence_length: Comprimento da sequência de entrada
            n_features: Número de features
            lstm_layers: Lista de configurações das camadas LSTM
            dense_layers: Lista de configurações das camadas Dense
            optimizer: Otimizador a usar
            learning_rate: Taxa de aprendizado
            loss: Função de perda
            metrics: Métricas para monitorar
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_layers = lstm_layers
        self.dense_layers = dense_layers
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics

        self.model = None

        print(f"🧠 LSTMStockPredictor inicializado:")
        print(f"   Input Shape: ({sequence_length}, {n_features})")
        print(f"   LSTM Layers: {len(lstm_layers)}")
        print(f"   Dense Layers: {len(dense_layers)}")
        print(f"   Optimizer: {optimizer} (lr={learning_rate})")
        print(f"   Loss: {loss}")

    def build_model(self) -> Sequential:
        """
        Constrói a arquitetura do modelo LSTM.

        Returns:
            Modelo Keras compilado
        """
        print("\n🏗️  Construindo arquitetura do modelo...")

        model = Sequential()

        # ============================================
        # CAMADAS LSTM
        # ============================================
        for i, layer_config in enumerate(self.lstm_layers):
            units = layer_config['units']
            return_sequences = layer_config.get('return_sequences', False)
            dropout = layer_config.get('dropout', 0.0)

            # ✅ CORREÇÃO: Usar dropout INTERNO da LSTM (mais eficaz)
            # Dropout interno regulariza DENTRO da LSTM, não só entre camadas
            # Primeira camada precisa especificar input_shape
            if i == 0:
                model.add(LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    input_shape=(self.sequence_length, self.n_features),
                    dropout=dropout,                    # ✅ Dropout nas entradas
                    recurrent_dropout=dropout,          # ✅ Dropout recorrente
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                    recurrent_regularizer=l1_l2(l1=0.001, l2=0.001),
                    name=f'lstm_{i+1}'
                ))
            else:
                model.add(LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=dropout,                    # ✅ Dropout nas entradas
                    recurrent_dropout=dropout,          # ✅ Dropout recorrente
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                    recurrent_regularizer=l1_l2(l1=0.001, l2=0.001),
                    name=f'lstm_{i+1}'
                ))

            # ❌ REMOVIDO: Dropout como camada separada não é mais necessário
            # O dropout interno da LSTM é mais eficaz

            print(f"✓ LSTM Layer {i+1}: units={units}, return_seq={return_sequences}, dropout={dropout} (interno)")

        # ============================================
        # CAMADAS DENSE
        # ============================================
        for i, layer_config in enumerate(self.dense_layers):
            units = layer_config['units']
            activation = layer_config.get('activation', None)

            model.add(Dense(
                units=units,
                activation=activation,
                name=f'dense_{i+1}'
            ))

            print(f"✓ Dense Layer {i+1}: units={units}, activation={activation}")

        # ============================================
        # COMPILAR MODELO
        # ============================================
        optimizer = self._get_optimizer()

        model.compile(
            optimizer=optimizer,
            loss=self.loss,
            metrics=self.metrics
        )

        print(f"\n✅ Modelo compilado com sucesso!")
        print(f"   Total de parâmetros: {model.count_params():,}")

        self.model = model
        return model

    def _get_optimizer(self):
        """
        Retorna o otimizador configurado.

        Returns:
            Otimizador Keras
        """
        if self.optimizer_name.lower() == 'adam':
            return Adam(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'sgd':
            from tensorflow.keras.optimizers import SGD
            return SGD(learning_rate=self.learning_rate)
        elif self.optimizer_name.lower() == 'rmsprop':
            from tensorflow.keras.optimizers import RMSprop
            return RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f"Otimizador desconhecido: {self.optimizer_name}")

    def get_model_summary(self) -> str:
        """
        Retorna um resumo do modelo.

        Returns:
            String com o resumo do modelo
        """
        if self.model is None:
            raise ValueError("Modelo ainda não foi construído!")

        # Capturar summary em string
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)

    def save_model(self, filepath: str = "models/lstm_model.h5"):
        """
        Salva o modelo treinado.

        Args:
            filepath: Caminho para salvar o modelo
        """
        if self.model is None:
            raise ValueError("Modelo ainda não foi construído!")

        self.model.save(filepath)
        print(f"\n💾 Modelo salvo em: {filepath}")

    def load_model(self, filepath: str = "models/lstm_model.h5"):
        """
        Carrega um modelo salvo.

        Args:
            filepath: Caminho do modelo salvo
        """
        self.model = keras.models.load_model(filepath)
        print(f"\n📂 Modelo carregado de: {filepath}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz predições usando o modelo.

        Args:
            X: Array de entrada (samples, sequence_length, features)

        Returns:
            Array de predições
        """
        if self.model is None:
            raise ValueError("Modelo ainda não foi construído ou carregado!")

        return self.model.predict(X, verbose=0)


def create_model_from_config(config: Dict[str, Any], n_features: int) -> LSTMStockPredictor:
    """
    Cria um modelo a partir de um dicionário de configuração.

    Args:
        config: Dicionário com configurações (do config.yaml)
        n_features: Número de features de entrada

    Returns:
        Instância de LSTMStockPredictor
    """
    model_config = config['model']
    training_config = config.get('training', {})

    lstm_model = LSTMStockPredictor(
        sequence_length=model_config['sequence_length'],
        n_features=n_features,
        lstm_layers=model_config['lstm_layers'],
        dense_layers=model_config['dense_layers'],
        optimizer=model_config.get('optimizer', 'adam'),
        learning_rate=model_config.get('learning_rate', 0.001),
        loss=model_config.get('loss', 'mse'),
        metrics=model_config.get('metrics', ['mae'])
    )

    lstm_model.build_model()

    return lstm_model


if __name__ == "__main__":
    # Teste do módulo
    print("\n" + "="*60)
    print("TESTE DO MÓDULO LSTM_MODEL")
    print("="*60)

    # Configuração de exemplo (mesma do config.yaml)
    lstm_config = [
        {'units': 128, 'return_sequences': True, 'dropout': 0.2},
        {'units': 64, 'return_sequences': True, 'dropout': 0.2},
        {'units': 32, 'return_sequences': False, 'dropout': 0.2},
    ]

    dense_config = [
        {'units': 16, 'activation': 'relu'},
        {'units': 1, 'activation': None},
    ]

    # Criar modelo
    model_builder = LSTMStockPredictor(
        sequence_length=60,
        n_features=10,  # Exemplo: OHLCV + VIX + 4 features adicionais
        lstm_layers=lstm_config,
        dense_layers=dense_config,
        optimizer='adam',
        learning_rate=0.001,
        loss='mse',
        metrics=['mae']
    )

    # Construir modelo
    model = model_builder.build_model()

    # Exibir resumo
    print("\n" + "="*60)
    print("RESUMO DO MODELO")
    print("="*60)
    print(model_builder.get_model_summary())

    # Testar predição com dados dummy
    print("\n🧪 Testando predição com dados dummy...")
    X_dummy = np.random.randn(10, 60, 10)  # 10 amostras, 60 timesteps, 10 features
    predictions = model_builder.predict(X_dummy)
    print(f"✓ Predições: shape={predictions.shape}, valores={predictions[:5].flatten()}")

    print("\n✅ Teste concluído com sucesso!")
