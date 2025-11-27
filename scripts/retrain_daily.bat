@echo off
REM Script para retreinamento periódico (Windows Task Scheduler)
REM Execute este script diariamente via Task Scheduler

cd /d "C:\Users\Willian\python_projects\tecchallenge_4"

REM Ativar ambiente virtual (se existir)
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Executar retreinamento incremental com backtest automático
python scripts/periodic_retrain.py --mode incremental --run-backtest --backtest-tests 100

REM Log
echo %date% %time% - Retreinamento e backtest executados >> logs\retrain_scheduler.log
