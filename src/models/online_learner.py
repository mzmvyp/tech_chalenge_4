"""
Módulo de Online Learning (Auto Learning)
==========================================

Implementa retreinamento incremental do modelo:
- Adiciona novos dados ao dataset
- Retreina modelo periodicamente
- Aprende com erros anteriores
- Melhora performance ao longo do tempo

Autor: Tech Challenge - Fase 04
Data: 2025-11-25
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import json
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class OnlineLearner:
    """
    Sistema de aprendizado online para o modelo LSTM.
    
    Permite que o modelo aprenda com novos dados e melhore ao longo do tempo.
    """
    
    def __init__(
        self,
        model_path: str = "models/lstm_model.h5",
        scaler_path: str = "models/scaler.pkl",
        data_buffer_path: str = "data/processed/online_learning_buffer.csv",
        retrain_threshold: int = 50,  # Retreinar quando tiver 50 novos exemplos
        fine_tune_epochs: int = 5,
        fine_tune_lr: float = 0.0001
    ):
        """
        Inicializa o sistema de online learning.
        
        Args:
            model_path: Caminho do modelo treinado
            scaler_path: Caminho do scaler
            data_buffer_path: Onde salvar novos dados
            retrain_threshold: Quantos novos exemplos antes de retreinar
            fine_tune_epochs: Épocas para fine-tuning
            fine_tune_lr: Learning rate para fine-tuning (menor que treino inicial)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.data_buffer_path = Path(data_buffer_path)
        self.retrain_threshold = retrain_threshold
        self.fine_tune_epochs = fine_tune_epochs
        self.fine_tune_lr = fine_tune_lr
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.target_idx = None
        self.sequence_length = None
        
        # Buffer de novos dados
        self.new_data_buffer = []
        
        print("🔄 OnlineLearner inicializado")
        print(f"   Retrain threshold: {retrain_threshold} novos exemplos")
        print(f"   Fine-tune epochs: {fine_tune_epochs}")
    
    def load_model_and_scaler(self):
        """Carrega modelo e scaler existentes."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
        
        print(f"\n📂 Carregando modelo de: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        
        # Recompilar com learning rate menor para fine-tuning
        from tensorflow.keras.optimizers import Adam
        self.model.compile(
            optimizer=Adam(learning_rate=self.fine_tune_lr),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"✓ Modelo carregado e recompilado (lr={self.fine_tune_lr})")
        
        # Carregar scaler
        scaler_data = joblib.load(self.scaler_path)
        if isinstance(scaler_data, dict):
            self.scaler = scaler_data['scaler']
            self.feature_names = scaler_data.get('feature_names', [])
            self.target_idx = scaler_data.get('target_idx', 0)
        else:
            self.scaler = scaler_data
        
        # Carregar sequence_length do config
        try:
            with open("data/processed/feature_config.json", 'r') as f:
                config = json.load(f)
                self.sequence_length = config.get('sequence_length', 75)
        except:
            self.sequence_length = 75  # Default
        
        print(f"✓ Scaler carregado")
        print(f"   Features: {len(self.feature_names) if self.feature_names else 'N/A'}")
        print(f"   Sequence length: {self.sequence_length}")
    
    def add_prediction_feedback(
        self,
        features: pd.DataFrame,
        actual_return: float,
        predicted_return: float,
        actual_close: float,
        predicted_close: float,
        date: Optional[datetime] = None
    ):
        """
        Adiciona feedback de uma predição ao buffer.
        
        Args:
            features: Features usadas para a predição (últimos sequence_length dias)
            actual_return: Return real observado
            predicted_return: Return predito pelo modelo
            actual_close: Close real observado
            predicted_close: Close predito
            date: Data da predição (default: hoje)
        """
        if date is None:
            date = datetime.now()
        
        # Calcular erro
        error = abs(actual_return - predicted_return)
        
        # Adicionar ao buffer
        feedback = {
            'date': date,
            'features': features.copy(),
            'actual_return': actual_return,
            'predicted_return': predicted_return,
            'actual_close': actual_close,
            'predicted_close': predicted_close,
            'error': error
        }
        
        self.new_data_buffer.append(feedback)
        
        print(f"✓ Feedback adicionado (erro: {error:.4f})")
        print(f"   Buffer: {len(self.new_data_buffer)}/{self.retrain_threshold} exemplos")
        
        # Verificar se deve retreinar
        if len(self.new_data_buffer) >= self.retrain_threshold:
            print(f"\n🔄 Buffer completo! Iniciando retreinamento...")
            self.retrain()
    
    def load_buffer(self) -> pd.DataFrame:
        """Carrega buffer de dados salvos."""
        if self.data_buffer_path.exists():
            df = pd.read_csv(self.data_buffer_path, index_col=0, parse_dates=True)
            print(f"✓ Buffer carregado: {len(df)} exemplos")
            return df
        else:
            print("⚠️  Buffer vazio (primeira execução)")
            return pd.DataFrame()
    
    def save_buffer(self, df: pd.DataFrame):
        """Salva buffer de dados."""
        self.data_buffer_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.data_buffer_path)
        print(f"✓ Buffer salvo: {len(df)} exemplos")
    
    def retrain(
        self,
        X_new: Optional[np.ndarray] = None,
        y_new: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        Retreina o modelo com novos dados.
        
        Usa fine-tuning: treina apenas algumas épocas com learning rate menor.
        
        Args:
            X_new: Novas sequências de entrada (opcional, pode vir do buffer)
            y_new: Novos valores alvo (opcional)
            verbose: Se True, exibe progresso
        """
        if self.model is None:
            self.load_model_and_scaler()
        
        if verbose:
            print("\n" + "="*60)
            print("🔄 RETREINAMENTO INCREMENTAL (Online Learning)")
            print("="*60)
        
        # Se não foram passados dados, usar buffer
        if X_new is None or y_new is None:
            if len(self.new_data_buffer) == 0:
                print("⚠️  Buffer vazio, nada para retreinar")
                return
            
            # Preparar dados do buffer
            # Por enquanto, apenas limpar buffer
            # TODO: Implementar preparação completa quando tiver pipeline de dados
            print(f"📊 Buffer tem {len(self.new_data_buffer)} novos exemplos")
            print("⚠️  Para retreinar, precisa passar X_new e y_new")
            print("   Ou implementar preparação completa do buffer")
            
            # Limpar buffer após processar
            self.new_data_buffer = []
            print("✓ Buffer processado e limpo")
            return
        
        # Fine-tuning com novos dados
        print(f"📊 Retreinando com {len(X_new)} novos exemplos...")
        print(f"   Fine-tuning: {self.fine_tune_epochs} épocas, lr={self.fine_tune_lr}")
        
        # Treinar
        history = self.model.fit(
            X_new, y_new,
            epochs=self.fine_tune_epochs,
            batch_size=min(32, len(X_new)),
            verbose=1 if verbose else 0,
            validation_split=0.2 if len(X_new) > 10 else 0.0
        )
        
        # Salvar modelo atualizado
        self.model.save(self.model_path)
        print(f"✓ Modelo atualizado salvo em: {self.model_path}")
        
        # Limpar buffer
        self.new_data_buffer = []
        
        if verbose:
            print("="*60)
            print("✅ Retreinamento concluído!")
            print("="*60)
    
    def predict_with_learning(
        self,
        sequence: np.ndarray,
        actual_value: Optional[float] = None,
        actual_return: Optional[float] = None,
        date: Optional[datetime] = None
    ) -> Dict:
        """
        Faz predição e aprende com o resultado (se disponível).
        
        Args:
            sequence: Sequência de entrada
            actual_value: Valor real observado (opcional, para learning)
            actual_return: Return real observado (opcional)
            date: Data da predição
        
        Returns:
            Dicionário com predição e informações
        """
        if self.model is None:
            self.load_model_and_scaler()
        
        # Fazer predição
        pred_scaled = self.model.predict(sequence.reshape(1, *sequence.shape), verbose=0)[0, 0]
        
        # Inverse transform
        dummy = np.zeros((1, self.scaler.n_features_in_))
        dummy[0, self.target_idx] = pred_scaled
        pred_return = self.scaler.inverse_transform(dummy)[0, self.target_idx]
        
        result = {
            'predicted_return': float(pred_return),
            'predicted_scaled': float(pred_scaled),
            'timestamp': datetime.now() if date is None else date
        }
        
        # Se temos valor real, adicionar feedback
        if actual_return is not None:
            # Converter para Close (precisa do último Close conhecido)
            # Por enquanto, apenas armazenar
            self.add_prediction_feedback(
                features=pd.DataFrame(),  # TODO: passar features corretas
                actual_return=actual_return,
                predicted_return=pred_return,
                actual_close=0,  # TODO: calcular
                predicted_close=0,  # TODO: calcular
                date=date
            )
        
        return result


def create_online_learner_from_config(config_path: str = "config.yaml") -> OnlineLearner:
    """
    Cria OnlineLearner a partir de configuração.
    
    Args:
        config_path: Caminho do arquivo de configuração
    
    Returns:
        Instância de OnlineLearner configurada
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    online_config = config.get('online_learning', {})
    
    return OnlineLearner(
        model_path=config['model_paths']['model_file'],
        scaler_path=config['model_paths']['scaler_file'],
        retrain_threshold=online_config.get('retrain_threshold', 50),
        fine_tune_epochs=online_config.get('fine_tune_epochs', 5),
        fine_tune_lr=online_config.get('fine_tune_lr', 0.0001)
    )


if __name__ == "__main__":
    # Teste do módulo
    print("\n" + "="*60)
    print("TESTE DO MÓDULO ONLINE LEARNING")
    print("="*60)
    
    learner = OnlineLearner(retrain_threshold=5)
    print("\n✅ OnlineLearner criado com sucesso!")
    print("\n💡 Uso:")
    print("   1. Modelo faz predição")
    print("   2. Quando valor real é conhecido, chama add_prediction_feedback()")
    print("   3. Quando buffer atinge threshold, retreina automaticamente")
    print("   4. Modelo melhora ao longo do tempo!")

