"""
Schemas Pydantic para API
==========================

Este módulo define os schemas (modelos de dados) para requests e responses da API.
Usa Pydantic para validação automática de dados.

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class StockDataPoint(BaseModel):
    """
    Modelo para um ponto de dados de ação.
    """
    date: str = Field(..., description="Data no formato YYYY-MM-DD")
    open: float = Field(..., description="Preço de abertura", gt=0)
    high: float = Field(..., description="Preço máximo", gt=0)
    low: float = Field(..., description="Preço mínimo", gt=0)
    close: float = Field(..., description="Preço de fechamento", gt=0)
    volume: float = Field(..., description="Volume negociado", ge=0)

    @validator('date')
    def validate_date(cls, v):
        """Valida formato da data."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError("Data deve estar no formato YYYY-MM-DD")

    @validator('high')
    def validate_high(cls, v, values):
        """Valida que high >= low."""
        if 'low' in values and v < values['low']:
            raise ValueError("Preço máximo deve ser >= preço mínimo")
        return v

    class Config:
        schema_extra = {
            "example": {
                "date": "2024-11-16",
                "open": 5730.50,
                "high": 5745.25,
                "low": 5720.00,
                "close": 5738.80,
                "volume": 1500000000
            }
        }


class PredictionRequest(BaseModel):
    """
    Modelo para request de predição.
    """
    data: List[StockDataPoint] = Field(
        ...,
        description="Lista de dados históricos (mínimo 60 dias)",
        min_items=60
    )

    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {
                        "date": "2024-11-01",
                        "open": 5730.50,
                        "high": 5745.25,
                        "low": 5720.00,
                        "close": 5738.80,
                        "volume": 1500000000
                    },
                    # ... mais 59 dias
                ]
            }
        }


class PredictionResponse(BaseModel):
    """
    Modelo para response de predição.
    """
    prediction: float = Field(..., description="Preço predito para o próximo dia")
    confidence_lower: Optional[float] = Field(None, description="Limite inferior do intervalo de confiança (95%)")
    confidence_upper: Optional[float] = Field(None, description="Limite superior do intervalo de confiança (95%)")
    model_version: str = Field(..., description="Versão do modelo")
    timestamp: str = Field(..., description="Timestamp da predição")

    class Config:
        schema_extra = {
            "example": {
                "prediction": 5745.32,
                "confidence_lower": 5720.15,
                "confidence_upper": 5770.50,
                "model_version": "1.0.0",
                "timestamp": "2024-11-16T10:30:00Z"
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Modelo para request de predição em batch.
    """
    sequences: List[List[StockDataPoint]] = Field(
        ...,
        description="Lista de sequências de dados históricos",
        min_items=1
    )

    @validator('sequences')
    def validate_sequences(cls, v):
        """Valida que todas as sequências têm pelo menos 60 pontos."""
        for i, seq in enumerate(v):
            if len(seq) < 60:
                raise ValueError(f"Sequência {i} tem menos de 60 pontos ({len(seq)})")
        return v


class BatchPredictionResponse(BaseModel):
    """
    Modelo para response de predição em batch.
    """
    predictions: List[float] = Field(..., description="Lista de predições")
    model_version: str = Field(..., description="Versão do modelo")
    timestamp: str = Field(..., description="Timestamp da predição")
    total_predictions: int = Field(..., description="Número total de predições")


class ModelInfo(BaseModel):
    """
    Modelo para informações do modelo.
    """
    model_version: str = Field(..., description="Versão do modelo")
    model_type: str = Field(..., description="Tipo do modelo")
    training_date: Optional[str] = Field(None, description="Data do treinamento")
    metrics: Optional[Dict[str, float]] = Field(None, description="Métricas de performance")
    sequence_length: int = Field(..., description="Comprimento da sequência de entrada")
    n_features: int = Field(..., description="Número de features")

    class Config:
        schema_extra = {
            "example": {
                "model_version": "1.0.0",
                "model_type": "LSTM",
                "training_date": "2024-11-16",
                "metrics": {
                    "MAE": 12.34,
                    "RMSE": 18.56,
                    "MAPE": 2.15,
                    "R2": 0.85,
                    "Direction_Accuracy": 62.5
                },
                "sequence_length": 60,
                "n_features": 10
            }
        }


class HealthResponse(BaseModel):
    """
    Modelo para response de health check.
    """
    status: str = Field(..., description="Status da API")
    model_loaded: bool = Field(..., description="Se o modelo está carregado")
    timestamp: str = Field(..., description="Timestamp do health check")
    version: str = Field(..., description="Versão da API")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": "2024-11-16T10:30:00Z",
                "version": "1.0.0"
            }
        }


class ErrorResponse(BaseModel):
    """
    Modelo para response de erro.
    """
    error: str = Field(..., description="Tipo do erro")
    message: str = Field(..., description="Mensagem de erro")
    timestamp: str = Field(..., description="Timestamp do erro")

    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Dados inválidos fornecidos",
                "timestamp": "2024-11-16T10:30:00Z"
            }
        }
