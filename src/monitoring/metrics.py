"""
Módulo de Monitoramento e Métricas
===================================

Implementa rastreamento de performance do modelo em produção.

CORREÇÃO OPUS: Adicionado monitoramento completo de predições.

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class ModelMonitor:
    """Monitor de performance do modelo em produção."""

    def __init__(self, log_dir: str = "logs/monitoring"):
        """
        Inicializa o monitor.

        Args:
            log_dir: Diretório para salvar logs de monitoramento
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = []

        print(f"📊 ModelMonitor inicializado")
        print(f"   Log dir: {self.log_dir}")

    def log_prediction(
        self,
        input_shape: tuple,
        prediction: float,
        inference_time: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registra métricas de uma predição.

        Args:
            input_shape: Shape da entrada (sequence_length, features)
            prediction: Valor predito
            inference_time: Tempo de inferência em segundos
            timestamp: Timestamp da predição (opcional)
            metadata: Metadados adicionais (opcional)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        metric = {
            'timestamp': timestamp.isoformat(),
            'input_shape': list(input_shape),
            'prediction': float(prediction),
            'inference_time_ms': float(inference_time * 1000),
        }

        # Adicionar metadados se fornecidos
        if metadata:
            metric['metadata'] = metadata

        self.metrics.append(metric)

        # Salvar em arquivo diário
        date_str = timestamp.strftime("%Y-%m-%d")
        log_file = self.log_dir / f"predictions_{date_str}.jsonl"

        with open(log_file, 'a') as f:
            json.dump(metric, f)
            f.write('\n')

    def get_daily_stats(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtém estatísticas do dia.

        Args:
            date: Data no formato YYYY-MM-DD (opcional, padrão: hoje)

        Returns:
            Dicionário com estatísticas
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        log_file = self.log_dir / f"predictions_{date}.jsonl"

        if not log_file.exists():
            return {
                'date': date,
                'message': 'Sem dados para esta data',
                'total_predictions': 0
            }

        metrics = []
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))

        if not metrics:
            return {
                'date': date,
                'message': 'Sem métricas',
                'total_predictions': 0
            }

        inference_times = [m['inference_time_ms'] for m in metrics]
        predictions = [m['prediction'] for m in metrics]

        return {
            'date': date,
            'total_predictions': len(metrics),
            'inference_time': {
                'mean_ms': sum(inference_times) / len(inference_times),
                'max_ms': max(inference_times),
                'min_ms': min(inference_times),
            },
            'predictions': {
                'mean': sum(predictions) / len(predictions),
                'max': max(predictions),
                'min': min(predictions),
            }
        }

    def get_hourly_stats(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtém estatísticas por hora.

        Args:
            date: Data no formato YYYY-MM-DD (opcional, padrão: hoje)

        Returns:
            Dicionário com estatísticas por hora
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        log_file = self.log_dir / f"predictions_{date}.jsonl"

        if not log_file.exists():
            return {'date': date, 'message': 'Sem dados para esta data'}

        metrics = []
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    metrics.append(json.loads(line))

        if not metrics:
            return {'date': date, 'message': 'Sem métricas'}

        # Agrupar por hora
        hourly_stats = {}
        for m in metrics:
            timestamp = datetime.fromisoformat(m['timestamp'].replace('Z', '+00:00'))
            hour = timestamp.strftime("%H:00")

            if hour not in hourly_stats:
                hourly_stats[hour] = {
                    'count': 0,
                    'inference_times': [],
                    'predictions': []
                }

            hourly_stats[hour]['count'] += 1
            hourly_stats[hour]['inference_times'].append(m['inference_time_ms'])
            hourly_stats[hour]['predictions'].append(m['prediction'])

        # Calcular estatísticas por hora
        result = {'date': date, 'hourly': {}}
        for hour, data in hourly_stats.items():
            result['hourly'][hour] = {
                'count': data['count'],
                'avg_inference_ms': sum(data['inference_times']) / len(data['inference_times']),
                'avg_prediction': sum(data['predictions']) / len(data['predictions'])
            }

        return result

    def clear_old_logs(self, days_to_keep: int = 30):
        """
        Remove logs antigos.

        Args:
            days_to_keep: Número de dias para manter (padrão: 30)
        """
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        removed_count = 0
        for log_file in self.log_dir.glob("predictions_*.jsonl"):
            # Extrair data do nome do arquivo
            try:
                file_date_str = log_file.stem.replace('predictions_', '')
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d")

                if file_date < cutoff_date:
                    log_file.unlink()
                    removed_count += 1
            except Exception as e:
                print(f"⚠️ Erro ao processar {log_file}: {e}")

        print(f"🗑️  {removed_count} arquivos de log removidos (> {days_to_keep} dias)")


if __name__ == "__main__":
    # Teste do módulo
    print("\n" + "="*60)
    print("TESTE DO MÓDULO MONITORING")
    print("="*60)

    # Criar monitor
    monitor = ModelMonitor(log_dir="logs/monitoring_test")

    # Simular predições
    for i in range(10):
        monitor.log_prediction(
            input_shape=(60, 10),
            prediction=4500 + i * 10,
            inference_time=0.05 + i * 0.001,
            metadata={'test': True}
        )

    # Obter estatísticas
    stats = monitor.get_daily_stats()
    print("\n📊 Estatísticas do dia:")
    print(json.dumps(stats, indent=2))

    print("\n✅ Teste concluído com sucesso!")
