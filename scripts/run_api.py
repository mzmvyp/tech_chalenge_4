"""
Script para Executar a API
===========================

Este script inicia o servidor FastAPI para servir o modelo LSTM.

Uso:
    python scripts/run_api.py

    Ou em modo de desenvolvimento (com reload):
    python scripts/run_api.py --dev

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import sys
from pathlib import Path
import argparse

# Adicionar diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent))

import uvicorn


def main():
    """Inicia o servidor da API."""

    parser = argparse.ArgumentParser(description='Executar API FastAPI')
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Executar em modo de desenvolvimento (com reload)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host para bind (padrão: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Porta para bind (padrão: 8000)'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🚀 INICIANDO API LSTM STOCK PREDICTION")
    print("="*60)
    print(f"🌐 Host: {args.host}")
    print(f"🔌 Porta: {args.port}")
    print(f"🔧 Modo: {'Desenvolvimento (reload)' if args.dev else 'Produção'}")
    print(f"\n📝 Documentação disponível em:")
    print(f"   http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs")
    print(f"\n📊 Health check:")
    print(f"   http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/health")
    print("="*60 + "\n")

    # Configurações do uvicorn
    uvicorn_config = {
        "app": "src.api.main:app",
        "host": args.host,
        "port": args.port,
        "log_level": "info",
    }

    # Adicionar reload apenas em modo dev
    if args.dev:
        uvicorn_config["reload"] = True
        uvicorn_config["reload_dirs"] = ["src"]

    # Iniciar servidor
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
