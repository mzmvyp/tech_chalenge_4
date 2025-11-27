"""
Script para Configurar Agendamento de Retreinamento
====================================================

Este script ajuda a configurar o retreinamento automático periódico:
- Windows: Task Scheduler
- Linux/Mac: Cron
- Docker: Cron dentro do container

Uso:
    python scripts/setup_retrain_scheduler.py --platform windows
    python scripts/setup_retrain_scheduler.py --platform linux
    python scripts/setup_retrain_scheduler.py --platform docker
"""

import sys
from pathlib import Path
import argparse
import platform

sys.path.append(str(Path(__file__).parent.parent))


def generate_windows_task():
    """Gera script para Windows Task Scheduler."""
    script_content = f"""@echo off
REM Script para retreinamento periódico (Windows Task Scheduler)
REM Execute este script diariamente via Task Scheduler

cd /d "{Path.cwd()}"

REM Ativar ambiente virtual (se existir)
if exist "venv\\Scripts\\activate.bat" (
    call venv\\Scripts\\activate.bat
)

REM Executar retreinamento incremental com backtest automático
python scripts/periodic_retrain.py --mode incremental --run-backtest --backtest-tests 100

REM Log
echo %date% %time% - Retreinamento e backtest executados >> logs\\retrain_scheduler.log
"""
    
    script_path = Path("scripts/retrain_daily.bat")
    script_path.write_text(script_content, encoding='utf-8')
    
    print("\n[OK] Script Windows criado: scripts/retrain_daily.bat")
    print("\n[INFO] Para configurar no Task Scheduler:")
    print("   1. Abrir 'Agendador de Tarefas' (Task Scheduler)")
    print("   2. Criar Tarefa Básica")
    print("   3. Nome: 'LSTM Retrain Daily'")
    print("   4. Gatilho: Diariamente às 02:00")
    print("   5. Ação: Iniciar programa")
    print(f"   6. Programa: {script_path.absolute()}")
    print("\n   Ou execute manualmente:")
    print(f"   {script_path.absolute()}")


def generate_linux_cron():
    """Gera entrada para crontab do Linux/Mac."""
    script_path = Path("scripts/retrain_daily.sh")
    
    script_content = f"""#!/bin/bash
# Script para retreinamento periódico (Linux/Mac Cron)
# Execute este script diariamente via cron

cd "{Path.cwd()}"

# Ativar ambiente virtual (se existir)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Executar retreinamento incremental com backtest automático
python scripts/periodic_retrain.py --mode incremental --run-backtest --backtest-tests 100

# Log
echo "$(date) - Retreinamento e backtest executados" >> logs/retrain_scheduler.log
"""
    
    script_path.write_text(script_content, encoding='utf-8')
    
    # Tornar executável
    import os
    os.chmod(script_path, 0o755)
    
    cron_entry = f"0 2 * * * {script_path.absolute()}\n"
    
    print("\n[OK] Script Linux/Mac criado: scripts/retrain_daily.sh")
    print("\n[INFO] Para configurar no crontab:")
    print("   1. Editar crontab: crontab -e")
    print("   2. Adicionar linha:")
    print(f"   {cron_entry.strip()}")
    print("   (Isso executa diariamente às 02:00)")
    print("\n   Ou execute manualmente:")
    print(f"   bash {script_path.absolute()}")


def generate_docker_cron():
    """Gera Dockerfile com cron para retreinamento automático."""
    dockerfile_content = """# Dockerfile com Cron para Retreinamento Automático
FROM python:3.10-slim

# Instalar cron
RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar código
COPY . .

# Instalar dependências
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Criar script de retreinamento com backtest
RUN echo '#!/bin/bash\ncd /app && python scripts/periodic_retrain.py --mode incremental --run-backtest --backtest-tests 100 >> logs/retrain.log 2>&1' > /app/retrain.sh && \
    chmod +x /app/retrain.sh

# Configurar cron (executar diariamente às 02:00)
RUN echo '0 2 * * * /app/retrain.sh' | crontab -

# Criar diretórios
RUN mkdir -p models logs outputs data/raw data/processed

# Expor porta
EXPOSE 8000

# Iniciar cron e API
CMD service cron start && python scripts/run_api.py
"""
    
    dockerfile_path = Path("docker/Dockerfile.with_cron")
    dockerfile_path.write_text(dockerfile_content, encoding='utf-8')
    
    print("\n[OK] Dockerfile com cron criado: docker/Dockerfile.with_cron")
    print("\n[INFO] Para usar:")
    print("   docker build -f docker/Dockerfile.with_cron -t lstm-api-cron .")
    print("   docker run -p 8000:8000 -v $(pwd)/models:/app/models lstm-api-cron")


def main():
    """Gera scripts de agendamento."""
    parser = argparse.ArgumentParser(description='Configurar Agendamento de Retreinamento')
    parser.add_argument(
        '--platform',
        type=str,
        choices=['windows', 'linux', 'mac', 'docker', 'all'],
        default='all',
        help='Plataforma para gerar scripts'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("CONFIGURACAO DE RETREINAMENTO AUTOMATICO")
    print("="*60)
    
    if args.platform in ['windows', 'all']:
        generate_windows_task()
    
    if args.platform in ['linux', 'mac', 'all']:
        generate_linux_cron()
    
    if args.platform in ['docker', 'all']:
        generate_docker_cron()
    
    print("\n" + "="*60)
    print("[OK] CONFIGURACAO CONCLUIDA!")
    print("="*60)
    print("\n[DICA] Teste o retreinamento manualmente primeiro:")
    print("   python scripts/periodic_retrain.py --mode incremental")


if __name__ == "__main__":
    main()

