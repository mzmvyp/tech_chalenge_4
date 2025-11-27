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

from fastapi import FastAPI, HTTPException, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
from typing import List, Dict, Any, Optional

# Imports dos schemas
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse,
    StockDataPoint,
    SimplePredictionRequest
)

# Imports dos módulos do projeto
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.predictor import StockPredictor
from src.data.preprocessor import TimeSeriesPreprocessor
from src.data.feature_engineering_stationary import create_stationary_features
from src.data.feature_selector import FeatureSelector
from src.monitoring.metrics import ModelMonitor
from src.api.prediction_storage import PredictionStorage
import uuid


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
        self.monitor: ModelMonitor = None
        self.feature_config: Dict[str, Any] = {}
        self.model_info: Dict[str, Any] = {}
        self.model_loaded: bool = False
        self.min_required_days: int = 90
        self.prediction_storage: PredictionStorage = PredictionStorage()

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

            # Carregar configuração de features (CORREÇÃO OPUS - CRÍTICO!)
            feature_config_path = Path("data/processed/feature_config.json")
            if feature_config_path.exists():
                with open(feature_config_path, 'r') as f:
                    self.feature_config = json.load(f)
                print(f"✓ Configuração de features carregada: {len(self.feature_config.get('features', []))} features")
            else:
                print("⚠️ WARNING: feature_config.json não encontrado. API pode não funcionar corretamente.")
                print("   Execute o treinamento primeiro: python scripts/train_model_stationary.py")

            # Inicializar preprocessor (não precisa de dados, só para processamento)
            sequence_length = self.feature_config.get('sequence_length', 60) if self.feature_config else 60
            self.preprocessor = TimeSeriesPreprocessor(
                sequence_length=sequence_length,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15
            )

            # Inicializar monitor (CORREÇÃO OPUS)
            self.monitor = ModelMonitor(log_dir="logs/monitoring")

            self.model_loaded = True
            print("✅ Modelo carregado com sucesso!")
            print("✅ Monitoramento ativado!")

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

        # ✅ CALCULAR MIN_REQUIRED DINAMICAMENTE baseado em config
        sequence_length = api_state.feature_config.get('sequence_length', 60)
        fe_config = api_state.feature_config.get('feature_engineering_config', {})

        # Encontrar maior janela rolling
        max_window = sequence_length  # Começa com sequence_length

        # Verificar volatility windows
        if fe_config.get('use_volatility'):
            # Volatility usa windows de 10 e 30 (hardcoded no FeatureEngineer)
            max_window = max(max_window, 30)

        # Verificar momentum windows
        if fe_config.get('use_momentum'):
            # Momentum usa windows de 5, 10, 20 (hardcoded)
            max_window = max(max_window, 20)

        # Verificar volume windows
        if fe_config.get('use_volume_features'):
            # Volume usa windows de 5, 20 (hardcoded)
            max_window = max(max_window, 20)

        # Verificar moving averages
        if fe_config.get('use_moving_averages'):
            # MA pode usar 5, 10, 20, 50 (hardcoded)
            max_window = max(max_window, 50)

        # MIN_REQUIRED = sequence_length + maior_janela
        api_state.min_required_days = sequence_length + max_window

        print(f"✓ Mínimo de dias necessários: {api_state.min_required_days}")
        print("✅ API pronta para receber requests!")

    except Exception as e:
        print(f"⚠️  API iniciada mas modelo não foi carregado: {e}")
        print("   Execute o treinamento primeiro: python scripts/train_model_stationary.py")
        # Fallback para valor padrão
        api_state.min_required_days = 90

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

    CORREÇÃO OPUS: Requer os últimos 90+ dias de dados (OHLCV) para criar features.
    Aplica as MESMAS features usadas no treinamento.
    """
    if not api_state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não está carregado"
        )

    import time
    start_time = time.time()  # CORREÇÃO OPUS: Monitoramento

    try:
        # ✅ USAR MIN_REQUIRED DINÂMICO (calculado no startup)
        if len(request.data) < api_state.min_required_days:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Mínimo de {api_state.min_required_days} dias de dados necessários. "
                       f"Recebido: {len(request.data)}"
            )

        # Converter dados para DataFrame
        data_dicts = [point.dict() for point in request.data]
        df = pd.DataFrame(data_dicts)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # Renomear colunas para uppercase
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        # Verificar se temos feature_config (CORREÇÃO OPUS - CRÍTICO!)
        if not api_state.feature_config:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Configuração de features não encontrada. Execute o treinamento primeiro."
            )

        # Criar VIX simulado se necessário
        df_vix = None
        if api_state.feature_config.get('use_vix', False):
            vix_value = 20.0
            df_vix = pd.DataFrame({
                'Open': [vix_value] * len(df),
                'High': [vix_value * 1.1] * len(df),
                'Low': [vix_value * 0.9] * len(df),
                'Close': [vix_value] * len(df),
                'Volume': [1000000] * len(df)
            }, index=df.index)
        
        # Criar features estacionárias (mesma função do treinamento)
        df_features = create_stationary_features(df, df_vix)
        
        # Aplicar FeatureSelector (mesmo do treinamento)
        selector = FeatureSelector(correlation_threshold=0.8)
        df_features = selector.select_features(df_features, verbose=False)

        # Verificar se temos features suficientes após criar rolling windows
        sequence_length = api_state.feature_config.get('sequence_length', 60)
        if len(df_features) < sequence_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Após criar features, apenas {len(df_features)} dias disponíveis. Mínimo: {sequence_length}"
            )

        # ✅ CORREÇÃO CRÍTICA: Usar features do scaler (fonte de verdade)
        # O scaler tem as features exatas usadas no treinamento (após FeatureSelector)
        if api_state.predictor.feature_names:
            expected_features = api_state.predictor.feature_names
        elif hasattr(api_state.predictor.scaler, 'feature_names_in_'):
            expected_features = list(api_state.predictor.scaler.feature_names_in_)
        else:
            expected_features = api_state.feature_config.get('features', [])
            if not expected_features:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Nao foi possivel determinar features esperadas. Execute o treinamento primeiro."
                )
        
        # Verificar se número de features corresponde
        if len(df_features.columns) != len(expected_features):
            missing_features = set(expected_features) - set(df_features.columns)
            extra_features = set(df_features.columns) - set(expected_features)
            
            if missing_features:
                # Adicionar features faltantes com valores padrão
                for feat in missing_features:
                    if feat == 'VIX' or 'VIX' in feat:
                        df_features[feat] = 20.0
                    else:
                        df_features[feat] = 0.0
            
            if extra_features:
                # Remover features extras (não esperadas pelo scaler)
                df_features = df_features.drop(columns=list(extra_features))

        # Reordenar colunas para match com o treinamento (CRÍTICO!)
        df_features = df_features[expected_features]

        # Normalizar dados
        data_scaled = api_state.predictor.scaler.transform(df_features.values)

        # Pegar últimos N dias para formar a sequência
        sequence = data_scaled[-sequence_length:]

        # Fazer predição (modelo prediz Return)
        prediction_return = api_state.predictor.predict_single(sequence, return_scaled=False)
        
        # Converter Return para Close
        last_close = float(df['Close'].iloc[-1])
        prediction_close = last_close * (1 + prediction_return)
        
        # Calcular direção
        if prediction_return > 0:
            direction = "up"
        elif prediction_return < 0:
            direction = "down"
        else:
            direction = "sideways"
        
        # Calcular intervalo de confiança
        if 'Return' in df_features.columns:
            recent_volatility = df_features['Return'].tail(30).std() * np.sqrt(1)
        else:
            recent_volatility = 0.01
        
        confidence_interval = float(prediction_close) * recent_volatility * 1.96

        # Gerar ID único e salvar predição
        prediction_id = str(uuid.uuid4())
        prediction_timestamp = datetime.utcnow()
        
        api_state.prediction_storage.save_prediction(
            prediction_id=prediction_id,
            timestamp=prediction_timestamp,
            predicted_price=float(prediction_close),
            predicted_return=float(prediction_return),
            direction=direction,
            confidence_lower=float(prediction_close - confidence_interval),
            confidence_upper=float(prediction_close + confidence_interval),
            metadata={
                "model_version": "1.0.0",
                "inference_time_ms": float((time.time() - start_time) * 1000)
            }
        )

        # Registrar no monitoramento
        inference_time = time.time() - start_time
        if api_state.monitor:
            api_state.monitor.log_prediction(
                input_shape=sequence.shape,
                prediction=float(prediction_close),
                inference_time=inference_time,
                timestamp=prediction_timestamp
            )

        return PredictionResponse(
            prediction=float(prediction_close),
            confidence_lower=float(prediction_close - confidence_interval),
            confidence_upper=float(prediction_close + confidence_interval),
            direction=direction,
            predicted_return=float(prediction_return),
            prediction_id=prediction_id,
            model_version="1.0.0",
            timestamp=prediction_timestamp.isoformat() + "Z",
            inference_time_ms=float((time.time() - start_time) * 1000)
        )

    except HTTPException:
        raise
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
# ENDPOINTS DE MONITORAMENTO (CORREÇÃO OPUS)
# ============================================

@app.get("/monitoring/stats", tags=["Monitoring"])
async def get_monitoring_stats(date: str = None):
    """
    Obtém estatísticas de monitoramento.

    Args:
        date: Data no formato YYYY-MM-DD (opcional, padrão: hoje)

    Returns:
        Estatísticas de predições do dia
    """
    if not api_state.monitor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Monitoramento não está habilitado"
        )

    try:
        stats = api_state.monitor.get_daily_stats(date)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter estatísticas: {str(e)}"
        )


@app.get("/monitoring/hourly", tags=["Monitoring"])
async def get_hourly_stats(date: str = None):
    """
    Obtém estatísticas horárias de monitoramento.

    Args:
        date: Data no formato YYYY-MM-DD (opcional, padrão: hoje)

    Returns:
        Estatísticas de predições por hora
    """
    if not api_state.monitor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Monitoramento não está habilitado"
        )

    try:
        stats = api_state.monitor.get_hourly_stats(date)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter estatísticas horárias: {str(e)}"
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


@app.post("/validate-prediction", tags=["Validation"], response_model=Dict[str, Any])
async def validate_prediction(
    prediction_id: str = Body(..., description="ID da predição a validar"),
    actual_price: float = Body(..., gt=0, description="Preço real observado"),
    actual_date: Optional[str] = Body(None, description="Data do valor real (YYYY-MM-DD)")
):
    """Valida uma predição com o valor real observado."""
    try:
        actual_date_obj = None
        if actual_date:
            actual_date_obj = datetime.strptime(actual_date, "%Y-%m-%d").date()
        
        result = api_state.prediction_storage.validate_prediction(
            prediction_id=prediction_id,
            actual_price=actual_price,
            actual_date=actual_date_obj
        )
        
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Predicao {prediction_id} nao encontrada"
            )
        
        return {
            "success": True,
            "prediction_id": prediction_id,
            "validation": result,
            "message": "Predicao validada com sucesso"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao validar predicao: {str(e)}"
        )


@app.post("/validate-by-date", tags=["Validation"], response_model=Dict[str, Any])
async def validate_by_date(
    target_date: str = Body(..., description="Data das predições (YYYY-MM-DD)"),
    actual_price: float = Body(..., gt=0, description="Preço real observado"),
    use_latest: bool = Body(True, description="Se True, valida apenas a última predição do dia")
):
    """Valida todas as predições de uma data específica."""
    try:
        target_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        
        validations = api_state.prediction_storage.validate_by_date(
            target_date=target_date_obj,
            actual_price=actual_price,
            use_latest=use_latest
        )
        
        return {
            "success": True,
            "target_date": target_date,
            "validations_count": len(validations),
            "validations": validations
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato de data invalido. Use YYYY-MM-DD: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao validar por data: {str(e)}"
        )


@app.post("/predict/simple", response_model=PredictionResponse, tags=["Prediction"])
async def predict_simple(request: SimplePredictionRequest):
    """
    Faz uma predição de forma SIMPLIFICADA - apenas forneça o símbolo da ação.
    
    A API busca os dados históricos automaticamente via yfinance.
    Muito mais fácil de usar que o endpoint /predict tradicional!
    
    Exemplo:
    {
      "symbol": "^GSPC",
      "days": 200
    }
    """
    if not api_state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo nao esta carregado"
        )
    
    import time
    start_time = time.time()
    
    try:
        # Buscar dados automaticamente
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days + 30)
        
        loader = DataLoader(
            symbol=request.symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            vix_symbol="^VIX" if request.symbol != "^VIX" else None
        )
        
        df_main, df_vix = loader.load_all_data()
        
        if len(df_main) < api_state.min_required_days:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dados insuficientes para {request.symbol}. Recebidos: {len(df_main)} dias, minimo: {api_state.min_required_days}"
            )
        
        # Converter para formato da API
        data_points = []
        for date, row in df_main.iterrows():
            data_points.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": float(row['Volume'])
            })
        
        # Criar DataFrame
        df = pd.DataFrame(data_points)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Criar VIX
        df_vix_api = None
        if api_state.feature_config.get('use_vix', False):
            if df_vix is not None and len(df_vix) > 0:
                vix_series = df_vix.iloc[:, 0] if isinstance(df_vix.columns, pd.MultiIndex) else df_vix['Close']
                df_vix_api = pd.DataFrame({
                    'Open': vix_series,
                    'High': vix_series * 1.1,
                    'Low': vix_series * 0.9,
                    'Close': vix_series,
                    'Volume': [1000000] * len(vix_series)
                }, index=df_vix.index)
            else:
                vix_value = 20.0
                df_vix_api = pd.DataFrame({
                    'Open': [vix_value] * len(df),
                    'High': [vix_value * 1.1] * len(df),
                    'Low': [vix_value * 0.9] * len(df),
                    'Close': [vix_value] * len(df),
                    'Volume': [1000000] * len(df)
                }, index=df.index)
        
        # Criar features
        df_features = create_stationary_features(df, df_vix_api)
        selector = FeatureSelector(correlation_threshold=0.8)
        df_features = selector.select_features(df_features, verbose=False)
        
        sequence_length = api_state.feature_config.get('sequence_length', 60)
        if len(df_features) < sequence_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Apos criar features, apenas {len(df_features)} dias disponiveis. Minimo: {sequence_length}"
            )
        
        # Usar features do scaler
        if api_state.predictor.feature_names:
            expected_features = api_state.predictor.feature_names
        elif hasattr(api_state.predictor.scaler, 'feature_names_in_'):
            expected_features = list(api_state.predictor.scaler.feature_names_in_)
        else:
            expected_features = api_state.feature_config.get('features', [])
            if not expected_features:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Nao foi possivel determinar features esperadas."
                )
        
        # Ajustar features
        if len(df_features.columns) != len(expected_features):
            missing_features = set(expected_features) - set(df_features.columns)
            extra_features = set(df_features.columns) - set(expected_features)
            
            if missing_features:
                for feat in missing_features:
                    if feat == 'VIX' or 'VIX' in feat:
                        df_features[feat] = 20.0
                    else:
                        df_features[feat] = 0.0
            
            if extra_features:
                df_features = df_features.drop(columns=list(extra_features))
        
        df_features = df_features[expected_features]
        
        # Normalizar e predizer
        data_scaled = api_state.predictor.scaler.transform(df_features.values)
        sequence = data_scaled[-sequence_length:]
        
        prediction_return = api_state.predictor.predict_single(sequence, return_scaled=False)
        last_close = float(df['Close'].iloc[-1])
        prediction_close = last_close * (1 + prediction_return)
        
        # Calcular direção
        if prediction_return > 0:
            direction = "up"
        elif prediction_return < 0:
            direction = "down"
        else:
            direction = "sideways"
        
        # Intervalo de confiança
        if 'Return' in df_features.columns:
            recent_volatility = df_features['Return'].tail(30).std() * np.sqrt(1)
        else:
            recent_volatility = 0.01
        
        confidence_interval = float(prediction_close) * recent_volatility * 1.96
        
        # Salvar predição
        prediction_id = str(uuid.uuid4())
        prediction_timestamp = datetime.utcnow()
        
        api_state.prediction_storage.save_prediction(
            prediction_id=prediction_id,
            timestamp=prediction_timestamp,
            predicted_price=float(prediction_close),
            predicted_return=float(prediction_return),
            direction=direction,
            confidence_lower=float(prediction_close - confidence_interval),
            confidence_upper=float(prediction_close + confidence_interval),
            metadata={
                "model_version": "1.0.0",
                "symbol": request.symbol,
                "inference_time_ms": float((time.time() - start_time) * 1000)
            }
        )
        
        # Monitoramento
        inference_time = time.time() - start_time
        if api_state.monitor:
            api_state.monitor.log_prediction(
                input_shape=sequence.shape,
                prediction=float(prediction_close),
                inference_time=inference_time,
                timestamp=prediction_timestamp
            )
        
        return PredictionResponse(
            prediction=float(prediction_close),
            confidence_lower=float(prediction_close - confidence_interval),
            confidence_upper=float(prediction_close + confidence_interval),
            direction=direction,
            predicted_return=float(prediction_return),
            prediction_id=prediction_id,
            model_version="1.0.0",
            timestamp=prediction_timestamp.isoformat() + "Z",
            inference_time_ms=float((time.time() - start_time) * 1000)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro ao processar predicao simplificada: {str(e)}"
        )
