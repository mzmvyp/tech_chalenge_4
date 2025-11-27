"""
Módulo de Coleta de Dados
==========================

Este módulo é responsável por baixar dados históricos de ações usando a API do Yahoo Finance.
Implementa validações e tratamento de erros para garantir qualidade dos dados.

IMPORTANTE: Este módulo apenas COLETA os dados brutos. O preprocessamento é feito em módulo separado.

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """Classe para carregar dados históricos de ações."""

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1d',
        vix_symbol: Optional[str] = None
    ):
        """
        Inicializa o carregador de dados.

        Args:
            symbol: Símbolo do ativo (ex: '^GSPC' para S&P 500)
            start_date: Data de início no formato 'YYYY-MM-DD'
            end_date: Data de fim no formato 'YYYY-MM-DD'
            interval: Intervalo dos dados ('1d', '1h', etc)
            vix_symbol: Símbolo do VIX (opcional)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.vix_symbol = vix_symbol

        print(f"DataLoader inicializado:")
        print(f"   Symbol: {symbol}")
        print(f"   Período: {start_date} a {end_date}")
        print(f"   Intervalo: {interval}")
        if vix_symbol:
            print(f"   VIX Symbol: {vix_symbol}")

    def download_data(self, symbol: str) -> pd.DataFrame:
        """
        Baixa dados históricos de um símbolo específico.

        Args:
            symbol: Símbolo do ativo

        Returns:
            DataFrame com os dados históricos

        Raises:
            ValueError: Se não conseguir baixar os dados
        """
        print(f"\nBaixando dados de {symbol}...")

        try:
            df = yf.download(
                symbol,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                progress=False
            )

            if df.empty:
                raise ValueError(f"Nenhum dado foi baixado para {symbol}")

            print(f"OK: {len(df)} registros baixados de {symbol}")
            print(f"  Período real: {df.index[0].date()} a {df.index[-1].date()}")
            print(f"  Colunas: {list(df.columns)}")

            return df

        except Exception as e:
            raise ValueError(f"Erro ao baixar dados de {symbol}: {str(e)}")

    def validate_data(self, df: pd.DataFrame, name: str = "Dataset") -> pd.DataFrame:
        """
        Valida e limpa os dados baixados.

        Args:
            df: DataFrame com os dados
            name: Nome do dataset (para logging)

        Returns:
            DataFrame validado e limpo

        Raises:
            ValueError: Se os dados forem inválidos
        """
        print(f"\nValidando {name}...")

        # Verificar se está vazio
        if df.empty:
            raise ValueError(f"{name} está vazio")

        # Verificar valores nulos
        null_counts = df.isnull().sum()
        if null_counts.any():
            print(f"AVISO: Valores nulos encontrados em {name}:")
            for col, count in null_counts[null_counts > 0].items():
                print(f"     {col}: {count} ({count/len(df)*100:.2f}%)")

            # Remover linhas com valores nulos
            df_clean = df.dropna()
            print(f"  Removidas {len(df) - len(df_clean)} linhas com valores nulos")
            df = df_clean

        # Verificar valores negativos em Volume
        if 'Volume' in df.columns:
            # ✅ CORREÇÃO: Garantir que neg_volume seja um número (não Series)
            neg_volume = int((df['Volume'] < 0).sum())
            if neg_volume > 0:
                print(f"AVISO: {neg_volume} valores negativos em Volume (corrigindo para 0)")
                df.loc[df['Volume'] < 0, 'Volume'] = 0

        # Verificar ordem temporal
        if not df.index.is_monotonic_increasing:
            print("AVISO: Dados fora de ordem temporal (reordenando)")
            df = df.sort_index()

        # Verificar duplicatas de índice
        duplicates = df.index.duplicated().sum()
        if duplicates > 0:
            print(f"AVISO: {duplicates} indices duplicados (removendo)")
            df = df[~df.index.duplicated(keep='first')]

        print(f"OK: {name} validado: {len(df)} registros validos")

        return df

    def load_main_data(self) -> pd.DataFrame:
        """
        Carrega e valida os dados principais (OHLCV).

        Returns:
            DataFrame com dados principais validados
        """
        df = self.download_data(self.symbol)
        
        # ✅ CORREÇÃO: Remover MultiIndex se existir
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)  # Remove segundo nível do MultiIndex
        
        df = self.validate_data(df, f"{self.symbol}")
        return df

    def load_vix_data(self) -> Optional[pd.DataFrame]:
        """
        Carrega dados do VIX se especificado.

        Returns:
            DataFrame com dados do VIX ou None
        """
        if self.vix_symbol is None:
            return None

        try:
            df_vix = self.download_data(self.vix_symbol)
            
            # ✅ CORREÇÃO: Remover MultiIndex se existir
            if isinstance(df_vix.columns, pd.MultiIndex):
                df_vix.columns = df_vix.columns.droplevel(1)  # Remove segundo nível do MultiIndex
            
            df_vix = self.validate_data(df_vix, "VIX")

            # Manter apenas a coluna Close do VIX
            df_vix = df_vix[['Close']].rename(columns={'Close': 'VIX'})

            return df_vix

        except Exception as e:
            print(f"AVISO: Erro ao carregar VIX: {e}")
            print("   Continuando sem dados do VIX...")
            return None

    def load_all_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Carrega todos os dados necessários.

        Returns:
            Tupla (dados_principais, dados_vix)
        """
        print("\n" + "="*60)
        print("INICIANDO COLETA DE DADOS")
        print("="*60)

        # Carregar dados principais
        df_main = self.load_main_data()

        # Carregar VIX se especificado
        df_vix = self.load_vix_data()

        print("\n" + "="*60)
        print("OK: COLETA DE DADOS CONCLUIDA")
        print("="*60)
        print(f"Total de registros: {len(df_main)}")
        print(f"Período: {df_main.index[0].date()} a {df_main.index[-1].date()}")

        return df_main, df_vix

    def save_raw_data(
        self,
        df_main: pd.DataFrame,
        df_vix: Optional[pd.DataFrame],
        output_dir: str = "data/raw"
    ):
        """
        Salva os dados brutos em arquivos CSV.

        Args:
            df_main: DataFrame com dados principais
            df_vix: DataFrame com dados do VIX (opcional)
            output_dir: Diretório de saída
        """
        # Criar diretório se não existir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Salvar dados principais
        main_path = Path(output_dir) / f"{self.symbol.replace('^', '')}_raw.csv"
        df_main.to_csv(main_path)
        print(f"\nDados principais salvos em: {main_path}")

        # Salvar VIX se disponível
        if df_vix is not None:
            vix_path = Path(output_dir) / "VIX_raw.csv"
            df_vix.to_csv(vix_path)
            print(f"Dados VIX salvos em: {vix_path}")

    @staticmethod
    def load_from_csv(
        symbol: str,
        output_dir: str = "data/raw"
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Carrega dados brutos salvos anteriormente.

        Args:
            symbol: Símbolo do ativo
            output_dir: Diretório onde estão os dados

        Returns:
            Tupla (dados_principais, dados_vix)
        """
        # Carregar dados principais
        main_path = Path(output_dir) / f"{symbol.replace('^', '')}_raw.csv"
        if not main_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {main_path}")

        df_main = pd.read_csv(main_path, index_col=0, parse_dates=True)
        print(f"Dados principais carregados de: {main_path}")

        # Carregar VIX se disponível
        vix_path = Path(output_dir) / "VIX_raw.csv"
        df_vix = None
        if vix_path.exists():
            df_vix = pd.read_csv(vix_path, index_col=0, parse_dates=True)
            print(f"Dados VIX carregados de: {vix_path}")

        return df_main, df_vix


if __name__ == "__main__":
    # Teste do módulo
    print("\n" + "="*60)
    print("TESTE DO MÓDULO DATA_LOADER")
    print("="*60)

    # Configurações de teste
    loader = DataLoader(
        symbol="^GSPC",
        start_date="2019-01-01",
        end_date="2024-11-01",
        interval="1d",
        vix_symbol="^VIX"
    )

    # Carregar todos os dados
    df_main, df_vix = loader.load_all_data()

    # Exibir informações
    print("\nInformacoes dos dados principais:")
    print(df_main.info())
    print("\n📈 Primeiras linhas:")
    print(df_main.head())
    print("\n📉 Últimas linhas:")
    print(df_main.tail())

    if df_vix is not None:
        print("\nInformacoes do VIX:")
        print(df_vix.info())
        print("\n📈 Primeiras linhas do VIX:")
        print(df_vix.head())

    # Salvar dados brutos
    loader.save_raw_data(df_main, df_vix)

    print("\nOK: Teste concluido com sucesso!")
