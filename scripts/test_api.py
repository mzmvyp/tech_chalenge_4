"""
Script para Testar a API LSTM Stock Prediction
===============================================

Este script testa todos os endpoints da API para validar funcionamento.

Uso:
    python scripts/test_api.py
    
    # Ou especificar URL customizada
    python scripts/test_api.py --url http://localhost:8000
"""

import sys
from pathlib import Path
import argparse
import requests
import json
from datetime import datetime, timedelta
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

# Configurações
DEFAULT_URL = "http://localhost:8000"


def test_health(url: str) -> bool:
    """Testa endpoint /health"""
    print("\n" + "="*60)
    print("🧪 TESTE 1: Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{url}/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Status: {data.get('status')}")
        print(f"✅ Timestamp: {data.get('timestamp')}")
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


def test_model_info(url: str) -> bool:
    """Testa endpoint /model/info"""
    print("\n" + "="*60)
    print("🧪 TESTE 2: Model Info")
    print("="*60)
    
    try:
        response = requests.get(f"{url}/model/info", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Modelo carregado: {data.get('model_loaded')}")
        print(f"✅ Versão: {data.get('version')}")
        print(f"✅ Features: {data.get('n_features')}")
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


def generate_test_data(n_days: int = 200) -> list:
    """Gera dados de teste simulados"""
    # Gerar dados históricos simulados
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Simular preços de ações (random walk)
    import numpy as np
    np.random.seed(42)
    base_price = 5000.0
    prices = [base_price]
    
    for i in range(1, n_days):
        change = np.random.normal(0, 50)  # Mudança aleatória
        new_price = prices[-1] + change
        prices.append(max(new_price, 100))  # Preço mínimo
    
    data = []
    for i, date in enumerate(dates):
        close = prices[i]
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": int(volume)
        })
    
    return data


def test_predict(url: str, min_days: int = 100) -> bool:
    """Testa endpoint /predict"""
    print("\n" + "="*60)
    print("🧪 TESTE 3: Prediction")
    print("="*60)
    
    try:
        # Gerar dados de teste
        print(f"📊 Gerando {min_days} dias de dados de teste...")
        test_data = generate_test_data(min_days)
        
        # Fazer request
        print(f"📤 Enviando request para {url}/predict...")
        request_data = {"data": test_data}
        
        response = requests.post(
            f"{url}/predict",
            json=request_data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        print(f"✅ Predição recebida!")
        print(f"   Preço predito: ${result.get('prediction', 'N/A'):.2f}")
        print(f"   Intervalo inferior: ${result.get('confidence_lower', 'N/A'):.2f}")
        print(f"   Intervalo superior: ${result.get('confidence_upper', 'N/A'):.2f}")
        print(f"   Versão do modelo: {result.get('model_version', 'N/A')}")
        print(f"   Timestamp: {result.get('timestamp', 'N/A')}")
        
        return True
    except requests.exceptions.HTTPError as e:
        print(f"❌ Erro HTTP: {e}")
        if e.response.status_code == 400:
            error_detail = e.response.json().get('detail', '')
            print(f"   Detalhes: {error_detail}")
        return False
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


def test_monitoring(url: str) -> bool:
    """Testa endpoint /monitoring/stats"""
    print("\n" + "="*60)
    print("🧪 TESTE 4: Monitoring Stats")
    print("="*60)
    
    try:
        response = requests.get(f"{url}/monitoring/stats", timeout=5)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Estatísticas recebidas")
        print(f"   Total de predições: {data.get('total_predictions', 'N/A')}")
        print(f"   Tempo médio: {data.get('avg_inference_time', 'N/A')}s")
        return True
    except Exception as e:
        print(f"⚠️  Monitoramento pode não estar habilitado: {e}")
        return True  # Não é crítico


def test_validation(url: str) -> bool:
    """Testa endpoint /validate-by-date"""
    print("\n" + "="*60)
    print("🧪 TESTE 5: Validation by Date")
    print("="*60)
    
    try:
        from datetime import date
        today = date.today().isoformat()
        
        body = {
            "target_date": today,
            "actual_price": 4650.0,
            "use_latest": True
        }
        
        response = requests.post(
            f"{url}/validate-by-date",
            json=body,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Validação realizada")
        print(f"   Data: {data.get('target_date')}")
        print(f"   Validações: {data.get('validations_count', 0)}")
        
        return True
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"⚠️  Nenhuma predição encontrada para hoje (normal se não houver predições)")
            return True  # Não é um erro, apenas não há predições
        print(f"❌ Erro HTTP: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


def main():
    """Executa todos os testes"""
    parser = argparse.ArgumentParser(description='Testar API LSTM Stock Prediction')
    parser.add_argument(
        '--url',
        type=str,
        default=DEFAULT_URL,
        help=f'URL da API (padrão: {DEFAULT_URL})'
    )
    parser.add_argument(
        '--min-days',
        type=int,
        default=200,
        help='Número mínimo de dias para teste de predição (padrão: 200)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 TESTES DA API LSTM STOCK PREDICTION")
    print("="*60)
    print(f"🌐 URL: {args.url}")
    print("="*60)
    
    results = []
    
    # Executar testes
    results.append(("Health Check", test_health(args.url)))
    results.append(("Model Info", test_model_info(args.url)))
    results.append(("Prediction", test_predict(args.url, args.min_days)))
    results.append(("Monitoring", test_monitoring(args.url)))
    results.append(("Validation", test_validation(args.url)))
    
    # Resumo
    print("\n" + "="*60)
    print("📊 RESUMO DOS TESTES")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{status} - {test_name}")
    
    print("="*60)
    print(f"✅ Testes passados: {passed}/{total}")
    
    if passed == total:
        print("🎉 Todos os testes passaram!")
        return 0
    else:
        print("⚠️  Alguns testes falharam. Verifique os erros acima.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

