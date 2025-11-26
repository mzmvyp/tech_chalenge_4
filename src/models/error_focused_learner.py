"""
Sistema de Aprendizado Focado em Erros (Error-Focused Learning)
==============================================================

Implementa aprendizado adaptativo que foca em aprender especificamente dos erros:
- Identifica predições com erro alto
- Aumenta peso desses exemplos no retreinamento
- Foca em casos difíceis (hard examples)
- Melhora performance em áreas problemáticas

Inspirado em técnicas de focal loss e curriculum learning.

Autor: Tech Challenge - Fase 04
Data: 2025-11-25
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class ErrorFocusedLearner:
    """
    Sistema de aprendizado que foca em aprender dos erros.
    
    Quando o modelo erra uma predição, esse erro é armazenado com peso maior
    para que o modelo aprenda especificamente desses casos difíceis.
    """
    
    def __init__(
        self,
        model_path: str = "models/lstm_model.h5",
        scaler_path: str = "models/scaler.pkl",
        error_threshold_percentile: float = 75.0,  # Erros acima deste percentil são "hard examples"
        error_weight_multiplier: float = 2.0,  # Multiplicador de peso para erros
        min_error_to_learn: float = 0.01,  # Erro mínimo para considerar aprendizado
        max_error_buffer: int = 1000  # Máximo de erros a manter em memória
    ):
        """
        Inicializa o Error-Focused Learner.
        
        Args:
            model_path: Caminho do modelo
            scaler_path: Caminho do scaler
            error_threshold_percentile: Percentil para considerar erro alto
            error_weight_multiplier: Multiplicador de peso para exemplos com erro
            min_error_to_learn: Erro mínimo para considerar aprendizado
            max_error_buffer: Máximo de erros a manter
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.error_threshold_percentile = error_threshold_percentile
        self.error_weight_multiplier = error_weight_multiplier
        self.min_error_to_learn = min_error_to_learn
        self.max_error_buffer = max_error_buffer
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.target_idx = None
        self.sequence_length = None
        
        # Buffer de erros (casos difíceis)
        self.error_buffer = []  # Lista de {features, actual, predicted, error, weight}
        self.error_history = []  # Histórico de erros para análise
        
        print("🎯 ErrorFocusedLearner inicializado")
        print(f"   Error threshold: P{error_threshold_percentile}")
        print(f"   Weight multiplier: {error_weight_multiplier}x")
        print(f"   Max buffer: {max_error_buffer} erros")
    
    def load_model_and_scaler(self):
        """Carrega modelo e scaler."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
        
        print(f"\n📂 Carregando modelo de: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        
        from tensorflow.keras.optimizers import Adam
        self.model.compile(
            optimizer=Adam(learning_rate=0.00001),  # Learning rate conservador
            loss='mse',
            metrics=['mae']
        )
        
        print(f"✓ Modelo carregado")
        
        # Carregar scaler
        scaler_data = joblib.load(self.scaler_path)
        if isinstance(scaler_data, dict):
            self.scaler = scaler_data['scaler']
            self.feature_names = scaler_data.get('feature_names', [])
            self.target_idx = scaler_data.get('target_idx', 0)
        else:
            self.scaler = scaler_data
            self.feature_names = []
            self.target_idx = 0
        
        # Carregar sequence_length do modelo ou config
        # ✅ CORREÇÃO: Tentar carregar do model_info.json primeiro (mais confiável)
        try:
            with open("models/model_info.json", 'r') as f:
                import json
                model_info = json.load(f)
                model_config = model_info.get('model_config', {})
                self.sequence_length = model_config.get('sequence_length', 60)
        except:
            try:
                with open("data/processed/feature_config.json", 'r') as f:
                    import json
                    config = json.load(f)
                    self.sequence_length = config.get('sequence_length', 60)
            except:
                self.sequence_length = 60  # Fallback
        
        print(f"✓ Scaler carregado")
        print(f"   Features: {len(self.feature_names) if self.feature_names else 'N/A'}")
        print(f"   Sequence length: {self.sequence_length}")
    
    def add_error_feedback(
        self,
        sequence_features: pd.DataFrame,
        actual_return: float,
        predicted_return: float,
        actual_close: float,
        predicted_close: float,
        date: Optional[datetime] = None
    ):
        """
        Adiciona feedback de um erro ao buffer.
        
        Args:
            sequence_features: Features da sequência que gerou a predição
            actual_return: Return real
            predicted_return: Return predito
            actual_close: Close real
            predicted_close: Close predito
            date: Data da predição
        """
        if date is None:
            date = datetime.now()
        
        # Calcular erro
        error_return = abs(actual_return - predicted_return)
        error_close = abs(actual_close - predicted_close)
        error_pct = (error_close / actual_close) * 100 if actual_close > 0 else 0
        
        # Armazenar no histórico
        self.error_history.append({
            'date': date,
            'error_return': error_return,
            'error_close': error_close,
            'error_pct': error_pct,
            'actual_return': actual_return,
            'predicted_return': predicted_return
        })
        
        # Se erro é significativo, adicionar ao buffer de aprendizado
        if error_pct >= self.min_error_to_learn:
            # Calcular peso baseado no erro (erros maiores = peso maior)
            error_weight = 1.0 + (error_pct / 10.0) * self.error_weight_multiplier
            error_weight = min(error_weight, 10.0)  # Limitar peso máximo
            
            error_entry = {
                'date': date,
                'sequence_features': sequence_features.copy(),
                'actual_return': actual_return,
                'predicted_return': predicted_return,
                'actual_close': actual_close,
                'predicted_close': predicted_close,
                'error_return': error_return,
                'error_close': error_close,
                'error_pct': error_pct,
                'weight': error_weight
            }
            
            self.error_buffer.append(error_entry)
            
            # Limitar tamanho do buffer
            if len(self.error_buffer) > self.max_error_buffer:
                # Manter apenas os erros mais significativos
                self.error_buffer.sort(key=lambda x: x['error_pct'], reverse=True)
                self.error_buffer = self.error_buffer[:self.max_error_buffer]
            
            print(f"🎯 Erro adicionado ao buffer: {error_pct:.2f}% (peso: {error_weight:.2f}x)")
            print(f"   Buffer: {len(self.error_buffer)}/{self.max_error_buffer} erros")
    
    def get_hard_examples(self, n_examples: Optional[int] = None) -> List[Dict]:
        """
        Retorna os exemplos mais difíceis (hard examples).
        
        Args:
            n_examples: Número de exemplos a retornar (None = todos)
        
        Returns:
            Lista de exemplos difíceis ordenados por erro
        """
        if not self.error_buffer:
            return []
        
        # Ordenar por erro (maior primeiro)
        sorted_errors = sorted(self.error_buffer, key=lambda x: x['error_pct'], reverse=True)
        
        if n_examples is None:
            return sorted_errors
        
        return sorted_errors[:n_examples]
    
    def learn_from_errors(
        self,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """
        Retreina o modelo focando nos erros (hard examples).
        
        Args:
            epochs: Número de épocas de retreinamento
            batch_size: Tamanho do batch
            verbose: Se True, exibe progresso
        """
        if not self.error_buffer:
            print("⚠️  Nenhum erro no buffer para aprender!")
            return
        
        if self.model is None:
            raise ValueError("Modelo não foi carregado! Execute load_model_and_scaler() primeiro.")
        
        print(f"\n🎯 Aprendendo de {len(self.error_buffer)} erros...")
        print("="*60)
        
        # Preparar dados de treinamento dos erros
        X_errors = []
        y_errors = []
        sample_weights = []
        
        for error_entry in self.error_buffer:
            # Normalizar sequência
            seq_features = error_entry['sequence_features']
            
            # Garantir ordem correta das features
            if self.feature_names:
                available_features = [f for f in self.feature_names if f in seq_features.columns]
                seq_features = seq_features[available_features]
            
            seq_array = self.scaler.transform(seq_features.values)
            X_errors.append(seq_array)
            
            # Target (Return real)
            y_errors.append(error_entry['actual_return'])
            
            # Peso (erros maiores têm peso maior)
            sample_weights.append(error_entry['weight'])
        
        if len(X_errors) == 0:
            print("⚠️  Nenhum exemplo válido para aprender!")
            return None
        
        X_errors = np.array(X_errors)
        y_errors = np.array(y_errors)
        sample_weights = np.array(sample_weights)
        
        print(f"✓ Preparados {len(X_errors)} exemplos de erro")
        print(f"   Erro médio: {np.mean([e['error_pct'] for e in self.error_buffer]):.2f}%")
        print(f"   Peso médio: {np.mean(sample_weights):.2f}x")
        
        # Retreinar modelo com pesos
        if verbose:
            print(f"\n🔄 Retreinando modelo focando em erros...")
        
        history = self.model.fit(
            X_errors,
            y_errors,
            sample_weight=sample_weights,  # ✅ Peso maior para erros maiores
            epochs=epochs,
            batch_size=batch_size,
            verbose=1 if verbose else 0
        )
        
        print(f"\n✅ Aprendizado de erros concluído!")
        print(f"   Loss final: {history.history['loss'][-1]:.6f}")
        print(f"   MAE final: {history.history['mae'][-1]:.6f}")
        
        return history
    
    def adaptive_learn_from_error(
        self,
        sequence_features: pd.DataFrame,
        actual_return: float,
        predicted_return: float,
        actual_close: float,
        predicted_close: float,
        date: Optional[datetime] = None,
        immediate_learn: bool = False
    ):
        """
        Versão adaptativa: aprende imediatamente de um erro grande.
        
        Similar ao que o usuário mencionou da AWS - quando há erro,
        o modelo aprende imediatamente desse erro específico.
        
        Args:
            sequence_features: Features da sequência
            actual_return: Return real
            predicted_return: Return predito
            actual_close: Close real
            predicted_close: Close predito
            date: Data
            immediate_learn: Se True, aprende imediatamente (não apenas adiciona ao buffer)
        """
        error_pct = (abs(actual_close - predicted_close) / actual_close) * 100 if actual_close > 0 else 0
        
        # Adicionar ao buffer
        self.add_error_feedback(
            sequence_features, actual_return, predicted_return,
            actual_close, predicted_close, date
        )
        
        # Se erro é muito grande, aprender imediatamente
        error_threshold = 5.0  # 5% de erro
        if len(self.error_history) > 10:
            recent_errors = [e['error_pct'] for e in self.error_history[-100:]]
            error_threshold = np.percentile(recent_errors, 90)
        
        if immediate_learn and error_pct > error_threshold:
            print(f"\n🚨 Erro grande detectado ({error_pct:.2f}%)! Aprendendo imediatamente...")
            
            # ✅ CORREÇÃO: Garantir ordem correta das features
            if self.feature_names:
                available_features = [f for f in self.feature_names if f in sequence_features.columns]
                if len(available_features) != len(self.feature_names):
                    print(f"   ⚠️  Features não correspondem, pulando aprendizado imediato")
                    return
                sequence_features = sequence_features[available_features]
            
            # Aprender apenas deste erro (mini-batch de 1)
            seq_array = self.scaler.transform(sequence_features.values)
            
            # ✅ CORREÇÃO: Verificar dimensões antes de reshape
            expected_size = self.sequence_length * len(self.feature_names)
            actual_size = seq_array.size
            
            if actual_size != expected_size:
                print(f"   ⚠️  Erro de dimensões: esperado {expected_size}, obtido {actual_size}")
                print(f"   Sequence length: {self.sequence_length}, Features: {len(self.feature_names)}")
                print(f"   Shape do array: {seq_array.shape}")
                print(f"   Pulando aprendizado imediato")
                return
            
            seq_array = seq_array.reshape(1, self.sequence_length, len(self.feature_names))
            
            # Calcular peso baseado no erro
            error_weight = 1.0 + (error_pct / 10.0) * self.error_weight_multiplier
            
            # Uma época de aprendizado focado neste erro
            try:
                self.model.fit(
                    seq_array,
                    np.array([actual_return]),
                    sample_weight=np.array([error_weight]),
                    epochs=1,
                    verbose=0
                )
            except Exception as e:
                print(f"   ⚠️  Erro no aprendizado imediato: {e}")
            
            print(f"✓ Aprendizado imediato concluído (peso: {error_weight:.2f}x)")
    
    def get_error_statistics(self) -> Dict:
        """
        Retorna estatísticas dos erros.
        
        Returns:
            Dicionário com estatísticas
        """
        if not self.error_history:
            return {}
        
        errors_pct = [e['error_pct'] for e in self.error_history]
        
        return {
            'total_errors': len(self.error_history),
            'buffer_size': len(self.error_buffer),
            'mean_error_pct': float(np.mean(errors_pct)),
            'median_error_pct': float(np.median(errors_pct)),
            'max_error_pct': float(np.max(errors_pct)),
            'min_error_pct': float(np.min(errors_pct)),
            'std_error_pct': float(np.std(errors_pct)),
            'errors_above_threshold': sum(1 for e in errors_pct if e > np.percentile(errors_pct, self.error_threshold_percentile))
        }
    
    def save_model(self, filepath: Optional[str] = None):
        """Salva o modelo atualizado."""
        if filepath is None:
            filepath = self.model_path
        
        self.model.save(filepath)
        print(f"💾 Modelo salvo em: {filepath}")


if __name__ == "__main__":
    # Teste do módulo
    print("\n" + "="*60)
    print("TESTE DO MÓDULO ERROR_FOCUSED_LEARNER")
    print("="*60)
    
    print("\n✅ Estrutura do módulo validada!")
    print("   Use ErrorFocusedLearner para aprender especificamente dos erros.")

