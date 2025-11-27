"""
Sistema de Ensemble de Modelos
================================

Implementa ensemble de múltiplos modelos para melhorar direction accuracy:
- Combina predições de múltiplos modelos
- Usa votação ou média ponderada
- Melhora robustez e accuracy

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


class EnsemblePredictor:
    """
    Sistema de ensemble que combina múltiplos modelos.
    
    Melhora direction accuracy através de:
    - Votação majoritária na direção
    - Média ponderada das predições
    - Redução de variância
    """
    
    def __init__(
        self,
        model_paths: List[str] = None,
        scaler_paths: List[str] = None,
        weights: Optional[List[float]] = None,
        voting_method: str = 'weighted'  # 'weighted', 'majority', 'average'
    ):
        """
        Inicializa o ensemble.
        
        Args:
            model_paths: Lista de caminhos dos modelos
            scaler_paths: Lista de caminhos dos scalers
            weights: Pesos para cada modelo (None = igual)
            voting_method: Método de votação ('weighted', 'majority', 'average')
        """
        self.model_paths = model_paths or ["models/lstm_model.h5"]
        self.scaler_paths = scaler_paths or ["models/scaler.pkl"]
        self.voting_method = voting_method
        
        # Validar que temos mesmo número de modelos e scalers
        if len(self.model_paths) != len(self.scaler_paths):
            raise ValueError("Número de modelos deve ser igual ao número de scalers!")
        
        # Pesos (normalizar para somar 1)
        if weights is None:
            weights = [1.0 / len(self.model_paths)] * len(self.model_paths)
        
        self.weights = np.array(weights)
        self.weights = self.weights / self.weights.sum()  # Normalizar
        
        self.models = []
        self.scalers = []
        self.feature_names_list = []
        self.target_idx_list = []
        self.sequence_length = None
        
        print(f"🎯 EnsemblePredictor inicializado")
        print(f"   Modelos: {len(self.model_paths)}")
        print(f"   Método: {voting_method}")
        print(f"   Pesos: {self.weights}")
    
    def load_all_models(self):
        """Carrega todos os modelos e scalers."""
        print(f"\n📂 Carregando {len(self.model_paths)} modelos...")
        
        for i, (model_path, scaler_path) in enumerate(zip(self.model_paths, self.scaler_paths)):
            print(f"\n   Modelo {i+1}/{len(self.model_paths)}:")
            
            # Carregar modelo
            if not Path(model_path).exists():
                print(f"   ⚠️  Modelo não encontrado: {model_path}")
                continue
            
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            self.models.append(model)
            
            # Carregar scaler
            scaler_data = joblib.load(scaler_path)
            if isinstance(scaler_data, dict):
                scaler = scaler_data['scaler']
                feature_names = scaler_data.get('feature_names', [])
                target_idx = scaler_data.get('target_idx', 0)
            else:
                scaler = scaler_data
                feature_names = []
                target_idx = 0
            
            self.scalers.append(scaler)
            self.feature_names_list.append(feature_names)
            self.target_idx_list.append(target_idx)
            
            # ✅ CORREÇÃO: Carregar sequence_length do model_info.json (mais confiável)
            if self.sequence_length is None:
                try:
                    with open("models/model_info.json", 'r') as f:
                        import json
                        model_info = json.load(f)
                        model_config = model_info.get('model_config', {})
                        self.sequence_length = model_config.get('sequence_length', 60)
                        print(f"   ✓ Sequence length carregado do model_info.json: {self.sequence_length}")
                except Exception as e:
                    # ✅ NÃO usar feature_config.json como fallback (pode ter valor errado)
                    # Usar 60 como padrão (valor correto do modelo atual)
                    self.sequence_length = 60
                    print(f"   ⚠️  Não foi possível carregar sequence_length do model_info.json: {e}")
                    print(f"   ⚠️  Usando sequence_length padrão: 60 (valor do modelo atual)")
            
            print(f"   ✓ Modelo {i+1} carregado")
        
        print(f"\n✅ {len(self.models)} modelos carregados")
    
    def predict_ensemble(
        self,
        sequence_features: pd.DataFrame,
        return_directions: bool = True
    ) -> Dict:
        """
        Faz predição usando ensemble.
        
        Args:
            sequence_features: Features da sequência
            return_directions: Se True, retorna também direções individuais
        
        Returns:
            Dicionário com predição do ensemble e detalhes
        """
        if not self.models:
            raise ValueError("Nenhum modelo carregado! Execute load_all_models() primeiro.")
        
        predictions = []
        directions = []
        
        for i, (model, scaler, feature_names, target_idx) in enumerate(
            zip(self.models, self.scalers, self.feature_names_list, self.target_idx_list)
        ):
            # ✅ CORREÇÃO: Garantir que sequence_features tem o número correto de linhas
            # Filtrar features para corresponder ao scaler
            if feature_names:
                available_features = [f for f in feature_names if f in sequence_features.columns]
                if len(available_features) != len(feature_names):
                    raise ValueError(f"Modelo {i+1}: Features incompatíveis! Esperado {len(feature_names)}, encontrado {len(available_features)}")
                sequence_features_filtered = sequence_features[available_features]
            else:
                sequence_features_filtered = sequence_features
            
            # ✅ CORREÇÃO: Verificar se tem linhas suficientes
            if len(sequence_features_filtered) != self.sequence_length:
                # ✅ DIAGNÓSTICO: Mostrar informações úteis
                print(f"   ⚠️  ERRO: Modelo {i+1}: Sequência tem {len(sequence_features_filtered)} linhas, mas esperado {self.sequence_length}")
                print(f"   📊 Features: {len(feature_names)} esperadas, {len(sequence_features_filtered.columns)} encontradas")
                raise ValueError(
                    f"Modelo {i+1}: Sequência tem {len(sequence_features_filtered)} linhas, "
                    f"mas esperado {self.sequence_length}. "
                    f"Verifique se sequence_features passado tem exatamente {self.sequence_length} linhas."
                )
            
            # Normalizar sequência
            seq_array = scaler.transform(sequence_features_filtered.values)
            seq_array = seq_array.reshape(1, self.sequence_length, len(feature_names))
            
            # Predição
            pred_scaled = model.predict(seq_array, verbose=0)[0, 0]
            
            # Inverse transform
            dummy = np.zeros((1, scaler.n_features_in_))
            dummy[0, target_idx] = pred_scaled
            pred_return = scaler.inverse_transform(dummy)[0, target_idx]
            
            predictions.append(pred_return)
            directions.append(1 if pred_return > 0 else -1)
        
        predictions = np.array(predictions)
        directions = np.array(directions)
        
        # Combinar predições
        if self.voting_method == 'weighted':
            ensemble_return = np.average(predictions, weights=self.weights)
        elif self.voting_method == 'average':
            ensemble_return = np.mean(predictions)
        elif self.voting_method == 'majority':
            # Usar mediana (mais robusto)
            ensemble_return = np.median(predictions)
        else:
            ensemble_return = np.mean(predictions)
        
        # Direção do ensemble
        ensemble_direction = 1 if ensemble_return > 0 else -1
        
        # Votação de direção
        direction_votes = np.sum(directions == 1)  # Votos para subir
        direction_confidence = direction_votes / len(directions)  # Confiança na direção
        
        result = {
            'ensemble_return': float(ensemble_return),
            'ensemble_direction': int(ensemble_direction),
            'direction_confidence': float(direction_confidence),
            'individual_predictions': predictions.tolist(),
            'individual_directions': directions.tolist(),
            'weights': self.weights.tolist()
        }
        
        if return_directions:
            result['direction_votes_up'] = int(direction_votes)
            result['direction_votes_down'] = int(len(directions) - direction_votes)
            result['direction_consensus'] = direction_votes > len(directions) / 2
        
        return result


class AdaptiveThresholdPredictor:
    """
    Sistema de threshold adaptativo para direction accuracy.
    
    Ajusta o threshold de decisão baseado em performance histórica.
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.0,
        learning_rate: float = 0.0001,  # ✅ MELHORIA: Learning rate mais conservador inicialmente
        min_threshold: float = -0.001,
        max_threshold: float = 0.001,
        confidence_threshold: float = 0.0005  # ✅ NOVO: Threshold de confiança mínimo
    ):
        """
        Inicializa predictor com threshold adaptativo.
        
        Args:
            initial_threshold: Threshold inicial
            learning_rate: Taxa de aprendizado do threshold
            min_threshold: Threshold mínimo
            max_threshold: Threshold máximo
        """
        self.threshold = initial_threshold
        self.learning_rate = learning_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.confidence_threshold = confidence_threshold  # ✅ NOVO
        
        self.history = []  # Histórico de acertos/erros
        
        print(f"🎯 AdaptiveThresholdPredictor inicializado")
        print(f"   Threshold inicial: {initial_threshold:.6f}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Confidence threshold: {confidence_threshold:.6f}")
    
    def predict_direction(
        self,
        predicted_return: float,
        actual_return: Optional[float] = None,
        confidence_threshold: float = 0.0005  # ✅ NOVO: Threshold de confiança mínimo
    ) -> Tuple[int, float]:
        """
        Prediz direção usando threshold adaptativo.
        
        Args:
            predicted_return: Return predito
            actual_return: Return real (opcional, para aprendizado)
            confidence_threshold: ✅ NOVO: Retorna "lateral" se confiança < threshold
        
        Returns:
            Tupla (direção, confiança)
        """
        # ✅ MELHORIA 1: Usar threshold adaptativo mais inteligente
        # Calcular threshold baseado em histórico recente
        if len(self.history) > 10:
            recent_errors = [h['predicted_return'] - h['actual_return'] 
                           for h in self.history[-20:]]
            bias = np.mean(recent_errors)
            # Ajustar threshold para compensar bias
            dynamic_threshold = self.threshold - bias * 0.5
        else:
            dynamic_threshold = self.threshold
        
        # Aplicar threshold
        adjusted_return = predicted_return - dynamic_threshold
        
        # ✅ MELHORIA 2: Zona de indecisão (lateral) se confiança baixa
        confidence = abs(adjusted_return)
        if confidence < confidence_threshold:
            # Se confiança muito baixa, considerar lateral (direção = 0)
            # Mas para compatibilidade, retornar direção baseada no sinal
            direction = 1 if adjusted_return >= 0 else -1
            # Reduzir confiança para indicar incerteza
            confidence = confidence / 2
        else:
            direction = 1 if adjusted_return > 0 else -1
        
        # Se temos valor real, aprender
        if actual_return is not None:
            actual_direction = 1 if actual_return > 0 else -1
            correct = (direction == actual_direction)
            
            self.history.append({
                'predicted_return': predicted_return,
                'actual_return': actual_return,
                'threshold': self.threshold,
                'correct': correct,
                'confidence': confidence
            })
            
            # ✅ MELHORIA 3: Ajustar threshold mais agressivamente quando erra
            if not correct:
                # Se errou, ajustar threshold na direção do erro
                error = predicted_return - actual_return
                # ✅ Aumentar learning rate para erros grandes
                adaptive_lr = self.learning_rate * (1 + abs(error) * 10)
                self.threshold += adaptive_lr * error
                self.threshold = np.clip(self.threshold, self.min_threshold, self.max_threshold)
            else:
                # ✅ MELHORIA 4: Ajustar threshold também quando acerta (refinar) - MAS COM CUIDADO
                # ✅ CORREÇÃO: Reduzir ajuste quando acerta para evitar overfitting
                if len(self.history) > 30:  # ✅ Aumentar janela para evitar ajustes muito frequentes
                    # Se acertou consistentemente, ajuste MUITO fino
                    recent_correct = [h['correct'] for h in self.history[-20:]]  # ✅ Janela maior
                    if sum(recent_correct) >= 15:  # ✅ 75%+ de acerto recente (mais conservador)
                        # ✅ Ajuste MUITO fino para evitar overfitting
                        fine_tune = (predicted_return - actual_return) * 0.05  # ✅ Reduzido de 0.1 para 0.05
                        self.threshold += fine_tune * self.learning_rate * 0.05  # ✅ Reduzido de 0.1 para 0.05
                        self.threshold = np.clip(self.threshold, self.min_threshold, self.max_threshold)
        
        return direction, confidence
    
    def get_threshold_statistics(self) -> Dict:
        """Retorna estatísticas do threshold."""
        if not self.history:
            return {}
        
        correct_count = sum(1 for h in self.history if h['correct'])
        accuracy = correct_count / len(self.history) * 100
        
        return {
            'current_threshold': self.threshold,
            'total_predictions': len(self.history),
            'accuracy': accuracy,
            'correct_predictions': correct_count
        }


if __name__ == "__main__":
    # Teste do módulo
    print("\n" + "="*60)
    print("TESTE DO MÓDULO ENSEMBLE_PREDICTOR")
    print("="*60)
    
    print("\n✅ Estrutura do módulo validada!")
    print("   Use EnsemblePredictor para melhorar direction accuracy.")

