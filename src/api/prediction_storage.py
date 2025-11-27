"""
Sistema de Armazenamento de Predições
======================================

Armazena predições da API para validação posterior e aprendizado de erros.

Autor: Tech Challenge - Fase 04
Data: 2025-11-27
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional
import numpy as np


class PredictionStorage:
    """Gerencia armazenamento e validação de predições."""
    
    def __init__(self, storage_dir: str = "data/predictions"):
        """
        Inicializa o sistema de armazenamento.
        
        Args:
            storage_dir: Diretório para armazenar predições
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Arquivo para predições pendentes de validação
        self.pending_file = self.storage_dir / "pending_predictions.json"
        self.validated_file = self.storage_dir / "validated_predictions.json"
        
    def save_prediction(
        self,
        prediction_id: str,
        timestamp: datetime,
        predicted_price: float,
        predicted_return: float,
        direction: str,
        confidence_lower: float,
        confidence_upper: float,
        input_data_hash: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Salva uma predição pendente de validação.
        
        Args:
            prediction_id: ID único da predição
            timestamp: Timestamp da predição
            predicted_price: Preço predito
            predicted_return: Return predito
            direction: Direção prevista ('up', 'down', 'sideways')
            confidence_lower: Limite inferior do intervalo
            confidence_upper: Limite superior do intervalo
            input_data_hash: Hash dos dados de entrada (opcional)
            metadata: Metadados adicionais (opcional)
            
        Returns:
            True se salvou com sucesso
        """
        try:
            # Carregar predições pendentes existentes
            pending = self._load_pending()
            
            # Criar entrada
            entry = {
                "prediction_id": prediction_id,
                "timestamp": timestamp.isoformat(),
                "date": timestamp.date().isoformat(),
                "predicted_price": float(predicted_price),
                "predicted_return": float(predicted_return),
                "direction": direction,
                "confidence_lower": float(confidence_lower),
                "confidence_upper": float(confidence_upper),
                "validated": False,
                "input_data_hash": input_data_hash,
                "metadata": metadata or {}
            }
            
            # Adicionar à lista
            pending.append(entry)
            
            # Salvar
            self._save_pending(pending)
            
            return True
        except Exception as e:
            print(f"ERRO ao salvar predicao: {e}")
            return False
    
    def validate_prediction(
        self,
        prediction_id: str,
        actual_price: float,
        actual_date: Optional[date] = None
    ) -> Optional[Dict]:
        """
        Valida uma predição com o valor real.
        
        Args:
            prediction_id: ID da predição
            actual_price: Preço real observado
            actual_date: Data do valor real (opcional, usa hoje se None)
            
        Returns:
            Dicionário com resultado da validação ou None se não encontrado
        """
        try:
            # Carregar predições pendentes
            pending = self._load_pending()
            
            # Encontrar predição
            prediction = None
            idx = None
            for i, p in enumerate(pending):
                if p["prediction_id"] == prediction_id:
                    prediction = p
                    idx = i
                    break
            
            if prediction is None:
                return None
            
            # Calcular erros
            predicted_price = prediction["predicted_price"]
            error_absolute = abs(actual_price - predicted_price)
            error_percentage = (error_absolute / actual_price) * 100 if actual_price > 0 else 0
            
            # Calcular direção real
            if actual_date is None:
                actual_date = date.today()
            
            # Para calcular direção, precisamos do preço anterior
            # Por enquanto, vamos usar o último preço do input (se disponível)
            # Isso será melhorado quando tivermos acesso ao histórico completo
            
            # Criar resultado de validação
            validation_result = {
                **prediction,
                "validated": True,
                "validation_timestamp": datetime.now().isoformat(),
                "actual_price": float(actual_price),
                "actual_date": actual_date.isoformat(),
                "error_absolute": float(error_absolute),
                "error_percentage": float(error_percentage),
                "direction_correct": None,  # Será calculado quando tivermos preço anterior
                "is_error": error_percentage > 0.5  # Considera erro se > 0.5%
            }
            
            # Mover para validações
            validated = self._load_validated()
            validated.append(validation_result)
            self._save_validated(validated)
            
            # Remover das pendentes
            pending.pop(idx)
            self._save_pending(pending)
            
            return validation_result
            
        except Exception as e:
            print(f"ERRO ao validar predicao: {e}")
            return None
    
    def validate_by_date(
        self,
        target_date: date,
        actual_price: float,
        use_latest: bool = True
    ) -> List[Dict]:
        """
        Valida todas as predições de uma data específica.
        
        Args:
            target_date: Data das predições a validar
            actual_price: Preço real observado
            use_latest: Se True, valida apenas a última predição do dia
            
        Returns:
            Lista de validações realizadas
        """
        try:
            pending = self._load_pending()
            target_date_str = target_date.isoformat()
            
            # Filtrar predições da data
            predictions_for_date = [
                p for p in pending
                if p.get("date") == target_date_str and not p.get("validated", False)
            ]
            
            if not predictions_for_date:
                return []
            
            # Se use_latest, pegar apenas a última
            if use_latest:
                predictions_for_date = [max(predictions_for_date, key=lambda x: x["timestamp"])]
            
            # Validar cada uma
            validations = []
            for pred in predictions_for_date:
                result = self.validate_prediction(
                    pred["prediction_id"],
                    actual_price,
                    target_date
                )
                if result:
                    validations.append(result)
            
            return validations
            
        except Exception as e:
            print(f"ERRO ao validar por data: {e}")
            return []
    
    def get_errors_for_learning(
        self,
        min_error_pct: float = 0.5,
        max_errors: int = 1000
    ) -> List[Dict]:
        """
        Retorna erros para aprendizado.
        
        Args:
            min_error_pct: Erro mínimo percentual para considerar
            max_errors: Número máximo de erros a retornar
            
        Returns:
            Lista de erros formatados para aprendizado
        """
        try:
            validated = self._load_validated()
            
            # Filtrar apenas erros significativos
            errors = [
                v for v in validated
                if v.get("is_error", False) and v.get("error_percentage", 0) >= min_error_pct
            ]
            
            # Ordenar por erro (maior primeiro)
            errors.sort(key=lambda x: x.get("error_percentage", 0), reverse=True)
            
            # Limitar quantidade
            errors = errors[:max_errors]
            
            return errors
            
        except Exception as e:
            print(f"ERRO ao obter erros para aprendizado: {e}")
            return []
    
    def get_pending_predictions(self, date_filter: Optional[date] = None) -> List[Dict]:
        """Retorna predições pendentes de validação."""
        pending = self._load_pending()
        if date_filter:
            date_str = date_filter.isoformat()
            return [p for p in pending if p.get("date") == date_str]
        return pending
    
    def _load_pending(self) -> List[Dict]:
        """Carrega predições pendentes."""
        if self.pending_file.exists():
            try:
                with open(self.pending_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_pending(self, data: List[Dict]):
        """Salva predições pendentes."""
        with open(self.pending_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_validated(self) -> List[Dict]:
        """Carrega predições validadas."""
        if self.validated_file.exists():
            try:
                with open(self.validated_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_validated(self, data: List[Dict]):
        """Salva predições validadas."""
        with open(self.validated_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

