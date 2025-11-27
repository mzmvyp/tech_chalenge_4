"""
Módulo de Configuração Centralizada
====================================

Este módulo carrega e valida as configurações do projeto a partir do arquivo config.yaml.
Garante que todas as configurações estejam corretas antes de iniciar o treinamento.

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
import numpy as np
import tensorflow as tf


class Config:
    """Classe para carregar e gerenciar configurações do projeto."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa as configurações a partir do arquivo YAML.

        Args:
            config_path: Caminho para o arquivo de configuração YAML
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._set_seeds()

    def _load_config(self) -> Dict[str, Any]:
        """
        Carrega o arquivo de configuração YAML.

        Returns:
            Dicionário com as configurações

        Raises:
            FileNotFoundError: Se o arquivo config.yaml não for encontrado
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Arquivo de configuração não encontrado: {self.config_path}"
            )

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def _validate_config(self):
        """
        Valida as configurações carregadas.

        Raises:
            ValueError: Se alguma configuração estiver inválida
        """
        # Validar splits somam 1.0
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        test_ratio = self.config['data']['test_ratio']

        total_ratio = train_ratio + val_ratio + test_ratio
        if not abs(total_ratio - 1.0) < 0.001:
            raise ValueError(
                f"A soma dos ratios de split deve ser 1.0. Atual: {total_ratio}"
            )

        # Validar sequence_length > 0
        if self.config['model']['sequence_length'] <= 0:
            raise ValueError("sequence_length deve ser maior que 0")

        # Validar batch_size > 0
        if self.config['training']['batch_size'] <= 0:
            raise ValueError("batch_size deve ser maior que 0")

        print("OK: Configuracoes validadas com sucesso!")

    def _set_seeds(self):
        """
        Define seeds para reprodutibilidade dos resultados.
        IMPORTANTE: Garante que os resultados sejam reproduzíveis.
        """
        seeds = self.config['random_state']

        # Python
        import random
        random.seed(seeds['python_seed'])

        # NumPy
        np.random.seed(seeds['numpy_seed'])

        # TensorFlow
        tf.random.set_seed(seeds['tensorflow_seed'])

        # Configurar TensorFlow para determinismo (pode reduzir performance)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['PYTHONHASHSEED'] = str(seeds['python_seed'])

        print(f"OK: Seeds configurados: {seeds}")

    def get(self, *keys):
        """
        Obtém um valor de configuração usando chaves aninhadas.

        Args:
            *keys: Chaves aninhadas (ex: 'data', 'symbol')

        Returns:
            Valor da configuração

        Example:
            >>> config = Config()
            >>> symbol = config.get('data', 'symbol')
            >>> print(symbol)  # '^GSPC'
        """
        value = self.config
        for key in keys:
            value = value[key]
        return value

    def get_data_config(self) -> Dict[str, Any]:
        """Retorna configurações de dados."""
        return self.config['data']

    def get_model_config(self) -> Dict[str, Any]:
        """Retorna configurações do modelo."""
        return self.config['model']

    def get_training_config(self) -> Dict[str, Any]:
        """Retorna configurações de treinamento."""
        return self.config['training']

    def get_api_config(self) -> Dict[str, Any]:
        """Retorna configurações da API."""
        return self.config['api']

    def create_directories(self):
        """
        Cria os diretórios necessários para o projeto se não existirem.
        """
        directories = [
            self.config['data']['raw_data_path'],
            self.config['data']['processed_data_path'],
            'models',
            'logs',
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

        print(f"OK: Diretorios criados/verificados: {directories}")


# Instância global de configuração (singleton)
_config_instance = None


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Obtém a instância global de configuração (singleton).

    Args:
        config_path: Caminho para o arquivo de configuração

    Returns:
        Instância de Config
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


if __name__ == "__main__":
    # Teste do módulo
    print("=" * 60)
    print("TESTE DO MÓDULO DE CONFIGURAÇÃO")
    print("=" * 60)

    config = get_config()

    print("\n📊 Configurações de Dados:")
    print(f"  - Symbol: {config.get('data', 'symbol')}")
    print(f"  - VIX Symbol: {config.get('data', 'vix_symbol')}")
    print(f"  - Período: {config.get('data', 'start_date')} a {config.get('data', 'end_date')}")
    print(f"  - Split: {config.get('data', 'train_ratio')}/{config.get('data', 'val_ratio')}/{config.get('data', 'test_ratio')}")

    print("\nConfiguracoes do Modelo:")
    print(f"  - Sequence Length: {config.get('model', 'sequence_length')}")
    print(f"  - LSTM Layers: {len(config.get('model', 'lstm_layers'))}")
    print(f"  - Optimizer: {config.get('model', 'optimizer')}")

    print("\nConfiguracoes de Treinamento:")
    print(f"  - Batch Size: {config.get('training', 'batch_size')}")
    print(f"  - Epochs: {config.get('training', 'epochs')}")
    print(f"  - Early Stopping Patience: {config.get('training', 'early_stopping', 'patience')}")

    config.create_directories()

    print("\nOK: Modulo de configuracao funcionando corretamente!")
    print("=" * 60)
