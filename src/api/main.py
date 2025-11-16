"""
API FastAPI para Predição de Ações com LSTM
============================================

Esta API fornece endpoints para predizer preços de ações usando um modelo LSTM treinado.

Endpoints:
- GET  /health          - Health check da API
- GET  /model/info      - Informações sobre o modelo
- POST /predict         - Fazer uma predição
- POST /predict/batch   - Fazer predições em batch

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
from typing import List, Dict, Any

# Imports dos schemas
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse,
    StockDataPoint
)

# Imports dos módulos do projeto
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.predictor import StockPredictor
from src.data.preprocessor import TimeSeriesPreprocessor
from src.data.feature_engineering import FeatureEngineer


# ============================================
# CONFIGURAÇÃO DA API
# ============================================

app = FastAPI(
    title="LSTM Stock Prediction API",
    description="API para predição de preços de ações usando modelo LSTM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS (permitir requests de qualquer origem)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# ESTADO GLOBAL DA API
# ============================================

class APIState:
    """Gerencia o estado global da API."""

    def __init__(self):
        self.predictor: StockPredictor = None
        self.preprocessor: TimeSeriesPreprocessor = None
        self.feature_engineer: FeatureEngineer = None
        self.model_info: Dict[str, Any] = {}
        self.model_loaded: bool = False

    def load_model(
        self,
        model_path: str = "models/lstm_model.h5",
        scaler_path: str = "models/scaler.pkl",
        info_path: str = "models/model_info.json"
    ):
        """Carrega o modelo e componentes necessários."""
        try:
            # Carregar predictor
            self.predictor = StockPredictor(model_path, scaler_path)
            self.predictor.load_all()

            # Carregar informações do modelo
            if Path(info_path).exists():
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)

            # Inicializar feature engineer
            self.feature_engineer = FeatureEngineer()

            # Inicializar preprocessor (não precisa de dados, só para processamento)
            self.preprocessor = TimeSeriesPreprocessor(
                sequence_length=60,  # Padrão
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15
            )

            self.model_loaded = True
            print("✅ Modelo carregado com sucesso!")

        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            self.model_loaded = False
            raise


# Instância global
api_state = APIState()


# ============================================
# EVENT HANDLERS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Executado quando a API inicia."""
    print("\n" + "="*60)
    print("🚀 INICIANDO API LSTM STOCK PREDICTION")
    print("="*60)

    try:
        api_state.load_model()
        print("✅ API pronta para receber requests!")
    except Exception as e:
        print(f"⚠️  API iniciada mas modelo não foi carregado: {e}")
        print("   Execute o treinamento primeiro: python scripts/train_model.py")

    print("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Executado quando a API é encerrada."""
    print("\n👋 Encerrando API...")


# ============================================
# EXCEPTION HANDLERS
# ============================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handler global para exceções."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


# ============================================
# ENDPOINTS
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Endpoint raiz."""
    return {
        "message": "LSTM Stock Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check da API.

    Retorna o status da API e se o modelo está carregado.
    """
    return HealthResponse(
        status="healthy" if api_state.model_loaded else "unhealthy",
        model_loaded=api_state.model_loaded,
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """
    Retorna informações sobre o modelo.

    Inclui métricas de performance, data de treinamento, etc.
    """
    if not api_state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não está carregado"
        )

    training_info = api_state.model_info.get('training_info', {})
    model_config = api_state.model_info.get('model_config', {})

    # Obter métricas do teste (se disponíveis)
    # TODO: Salvar métricas de teste no model_info.json
    metrics = {
        "MAE": 0.0,
        "RMSE": 0.0,
        "MAPE": 0.0,
        "R2": 0.0,
        "Direction_Accuracy": 0.0
    }

    return ModelInfo(
        model_version="1.0.0",
        model_type="LSTM",
        training_date=training_info.get('start_datetime'),
        metrics=metrics,
        sequence_length=model_config.get('sequence_length', 60),
        n_features=len(api_state.predictor.scaler.feature_names_in_) if hasattr(api_state.predictor.scaler, 'feature_names_in_') else 0
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Faz uma predição do preço de fechamento para o próximo dia.

    Requer os últimos 60 dias de dados (OHLCV).
    """
    if not api_state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não está carregado"
        )

    try:
        # Converter dados para DataFrame
        data_dicts = [point.dict() for point in request.data]
        df = pd.DataFrame(data_dicts)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Renomear colunas para uppercase (esperado pelo modelo)
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        # Criar features (se necessário)
        # TODO: Aplicar mesmas features usadas no treinamento

        # Normalizar dados
        data_scaled = api_state.predictor.scaler.transform(df.values)

        # Pegar últimos 60 dias
        if len(data_scaled) > 60:
            sequence = data_scaled[-60:]
        else:
            sequence = data_scaled

        # Fazer predição
        prediction = api_state.predictor.predict_single(sequence, return_scaled=False)

        # Calcular intervalo de confiança (opcional)
        # TODO: Implementar cálculo de confiança

        return PredictionResponse(
            prediction=float(prediction),
            confidence_lower=None,
            confidence_upper=None,
            model_version="1.0.0",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro ao processar predição: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Faz predições em batch para múltiplas sequências.

    Útil para processar múltiplas predições de uma vez.
    """
    if not api_state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não está carregado"
        )

    try:
        predictions = []

        for sequence_data in request.sequences:
            # Converter para DataFrame
            data_dicts = [point.dict() for point in sequence_data]
            df = pd.DataFrame(data_dicts)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            # Renomear colunas
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Normalizar
            data_scaled = api_state.predictor.scaler.transform(df.values)

            # Pegar últimos 60 dias
            if len(data_scaled) > 60:
                sequence = data_scaled[-60:]
            else:
                sequence = data_scaled

            # Predizer
            prediction = api_state.predictor.predict_single(sequence, return_scaled=False)
            predictions.append(float(prediction))

        return BatchPredictionResponse(
            predictions=predictions,
            model_version="1.0.0",
            timestamp=datetime.utcnow().isoformat() + "Z",
            total_predictions=len(predictions)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro ao processar predições em batch: {str(e)}"
        )


# ============================================
# MAIN (para desenvolvimento local)
# ============================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("🚀 INICIANDO API EM MODO DE DESENVOLVIMENTO")
    print("="*60)
    print("📝 Acesse a documentação em: http://localhost:8000/docs")
    print("="*60 + "\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
