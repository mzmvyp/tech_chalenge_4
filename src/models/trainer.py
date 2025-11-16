"""
Módulo de Treinamento
======================

Este módulo gerencia o treinamento do modelo LSTM com callbacks avançados:
- Early Stopping
- Model Checkpoint
- Reduce Learning Rate on Plateau
- Logging customizado

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    Callback
)
from typing import Dict, Any, Optional, Tuple
import json
from pathlib import Path
from datetime import datetime
import time


class TrainingLogger(Callback):
    """Callback customizado para logging durante o treinamento."""

    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.start_time = None

    def on_train_begin(self, logs=None):
        """Chamado no início do treinamento."""
        self.start_time = time.time()
        print("\n" + "="*60)
        print("🚀 INICIANDO TREINAMENTO")
        print("="*60)

    def on_epoch_begin(self, epoch, logs=None):
        """Chamado no início de cada época."""
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """Chamado ao final de cada época."""
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)

        # Obter métricas
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        mae = logs.get('mae', 0)
        val_mae = logs.get('val_mae', 0)
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

        # Exibir informações
        print(f"\nÉpoca {epoch+1:3d}/{self.params['epochs']:3d} | "
              f"Tempo: {epoch_time:.1f}s | LR: {lr:.6f}")
        print(f"  📉 Loss: {loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"  📊 MAE:  {mae:.6f} | Val MAE:  {val_mae:.6f}")

    def on_train_end(self, logs=None):
        """Chamado ao final do treinamento."""
        total_time = time.time() - self.start_time
        avg_epoch_time = np.mean(self.epoch_times)

        print("\n" + "="*60)
        print("✅ TREINAMENTO CONCLUÍDO")
        print("="*60)
        print(f"⏱️  Tempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"⏱️  Tempo médio por época: {avg_epoch_time:.1f}s")


class ModelTrainer:
    """
    Gerenciador de treinamento do modelo LSTM.
    """

    def __init__(
        self,
        model,
        config: Dict[str, Any]
    ):
        """
        Inicializa o trainer.

        Args:
            model: Modelo LSTM a ser treinado
            config: Dicionário de configurações
        """
        self.model = model
        self.config = config
        self.history = None
        self.training_info = {}

        print("🎯 ModelTrainer inicializado")

    def create_callbacks(self) -> list:
        """
        Cria os callbacks para o treinamento.

        Returns:
            Lista de callbacks
        """
        callbacks = []

        training_config = self.config['training']
        model_paths = self.config['model_paths']

        # ============================================
        # EARLY STOPPING
        # ============================================
        if 'early_stopping' in training_config:
            es_config = training_config['early_stopping']
            early_stopping = EarlyStopping(
                monitor=es_config.get('monitor', 'val_loss'),
                patience=es_config.get('patience', 10),
                restore_best_weights=es_config.get('restore_best_weights', True),
                min_delta=es_config.get('min_delta', 0.0001),
                mode='min',
                verbose=1
            )
            callbacks.append(early_stopping)
            print("✓ Early Stopping configurado:")
            print(f"  Monitor: {es_config.get('monitor')}, Patience: {es_config.get('patience')}")

        # ============================================
        # MODEL CHECKPOINT
        # ============================================
        if 'model_checkpoint' in training_config:
            mc_config = training_config['model_checkpoint']

            # Criar diretório se não existir
            Path(model_paths['model_file']).parent.mkdir(parents=True, exist_ok=True)

            checkpoint = ModelCheckpoint(
                filepath=model_paths['model_file'],
                monitor=mc_config.get('monitor', 'val_loss'),
                save_best_only=mc_config.get('save_best_only', True),
                mode=mc_config.get('mode', 'min'),
                verbose=1
            )
            callbacks.append(checkpoint)
            print("✓ Model Checkpoint configurado:")
            print(f"  Filepath: {model_paths['model_file']}")

        # ============================================
        # REDUCE LR ON PLATEAU
        # ============================================
        if 'reduce_lr' in training_config:
            rlr_config = training_config['reduce_lr']
            reduce_lr = ReduceLROnPlateau(
                monitor=rlr_config.get('monitor', 'val_loss'),
                factor=rlr_config.get('factor', 0.5),
                patience=rlr_config.get('patience', 5),
                min_lr=rlr_config.get('min_lr', 0.00001),
                mode='min',
                verbose=1
            )
            callbacks.append(reduce_lr)
            print("✓ Reduce LR on Plateau configurado:")
            print(f"  Factor: {rlr_config.get('factor')}, Patience: {rlr_config.get('patience')}")

        # ============================================
        # TRAINING LOGGER
        # ============================================
        training_logger = TrainingLogger()
        callbacks.append(training_logger)

        return callbacks

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Treina o modelo.

        Args:
            X_train: Dados de treino
            y_train: Alvos de treino
            X_val: Dados de validação
            y_val: Alvos de validação
            verbose: Nível de verbosidade

        Returns:
            Histórico de treinamento
        """
        print("\n" + "="*60)
        print("🏋️  CONFIGURANDO TREINAMENTO")
        print("="*60)

        training_config = self.config['training']

        # Exibir informações
        print(f"\n📊 Dados de Treinamento:")
        print(f"   X_train: {X_train.shape}")
        print(f"   y_train: {y_train.shape}")
        print(f"   X_val:   {X_val.shape}")
        print(f"   y_val:   {y_val.shape}")

        print(f"\n⚙️  Configurações:")
        print(f"   Batch Size: {training_config['batch_size']}")
        print(f"   Epochs: {training_config['epochs']}")

        # Criar callbacks
        callbacks = self.create_callbacks()

        # Registrar tempo de início
        start_time = time.time()
        start_datetime = datetime.now()

        # Treinar modelo
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=training_config['epochs'],
            batch_size=training_config['batch_size'],
            callbacks=callbacks,
            verbose=verbose
        )

        # Registrar tempo de fim
        training_time = time.time() - start_time

        # Guardar informações do treinamento
        self.training_info = {
            'start_datetime': start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'total_epochs': len(self.history.history['loss']),
            'final_train_loss': float(self.history.history['loss'][-1]),
            'final_val_loss': float(self.history.history['val_loss'][-1]),
            'best_val_loss': float(min(self.history.history['val_loss'])),
            'best_epoch': int(np.argmin(self.history.history['val_loss']) + 1),
        }

        print("\n" + "="*60)
        print("📊 RESUMO DO TREINAMENTO")
        print("="*60)
        print(f"⏱️  Tempo total: {training_time:.1f}s ({training_time/60:.1f} min)")
        print(f"📈 Total de épocas: {self.training_info['total_epochs']}")
        print(f"🏆 Melhor época: {self.training_info['best_epoch']}")
        print(f"📉 Melhor val_loss: {self.training_info['best_val_loss']:.6f}")
        print(f"📉 Final train_loss: {self.training_info['final_train_loss']:.6f}")
        print(f"📉 Final val_loss: {self.training_info['final_val_loss']:.6f}")

        return self.history

    def save_training_info(self, filepath: str = "models/model_info.json"):
        """
        Salva informações sobre o treinamento.

        Args:
            filepath: Caminho para salvar as informações
        """
        if not self.training_info:
            raise ValueError("Nenhuma informação de treinamento disponível!")

        # Criar diretório se não existir
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Adicionar configurações
        info_to_save = {
            'training_info': self.training_info,
            'model_config': self.config['model'],
            'training_config': self.config['training'],
            'data_config': self.config['data'],
        }

        # Salvar em JSON
        with open(filepath, 'w') as f:
            json.dump(info_to_save, f, indent=2)

        print(f"\n💾 Informações de treinamento salvas em: {filepath}")

    def get_history_dict(self) -> Dict[str, list]:
        """
        Retorna o histórico de treinamento como dicionário.

        Returns:
            Dicionário com histórico
        """
        if self.history is None:
            raise ValueError("Modelo ainda não foi treinado!")

        return self.history.history


if __name__ == "__main__":
    # Teste do módulo
    print("\n" + "="*60)
    print("TESTE DO MÓDULO TRAINER")
    print("="*60)

    # Criar dados dummy
    np.random.seed(42)
    X_train = np.random.randn(1000, 60, 10)
    y_train = np.random.randn(1000)
    X_val = np.random.randn(200, 60, 10)
    y_val = np.random.randn(200)

    # Criar modelo simples
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    model = Sequential([
        LSTM(32, input_shape=(60, 10)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Configuração de teste
    test_config = {
        'model': {'sequence_length': 60},
        'training': {
            'batch_size': 32,
            'epochs': 5,
            'early_stopping': {
                'monitor': 'val_loss',
                'patience': 3,
                'restore_best_weights': True
            }
        },
        'model_paths': {
            'model_file': 'models/test_model.h5',
            'metadata_file': 'models/test_info.json'
        }
    }

    # Criar trainer
    trainer = ModelTrainer(model, test_config)

    # Treinar
    history = trainer.train(X_train, y_train, X_val, y_val, verbose=0)

    # Salvar informações
    trainer.save_training_info('models/test_info.json')

    print("\n✅ Teste concluído com sucesso!")
