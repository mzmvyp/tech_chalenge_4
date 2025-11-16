"""
Testes para o módulo data_loader
=================================

Autor: Tech Challenge - Fase 04
Data: 2024-11-16
"""

import pytest
import pandas as pd
import numpy as np
from src.data.data_loader import DataLoader


class TestDataLoader:
    """Testes para a classe DataLoader."""

    def test_init(self):
        """Testa inicialização do DataLoader."""
        loader = DataLoader(
            symbol="^GSPC",
            start_date="2020-01-01",
            end_date="2020-12-31",
            interval="1d"
        )

        assert loader.symbol == "^GSPC"
        assert loader.start_date == "2020-01-01"
        assert loader.end_date == "2020-12-31"
        assert loader.interval == "1d"

    def test_validate_data_removes_nulls(self):
        """Testa que validate_data remove valores nulos."""
        # Criar dados com NaN
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'Open': np.random.randn(10),
            'High': np.random.randn(10),
            'Low': np.random.randn(10),
            'Close': np.random.randn(10),
            'Volume': np.random.randint(1000, 10000, 10),
        }, index=dates)

        # Adicionar NaN
        df.loc[dates[5], 'Close'] = np.nan

        loader = DataLoader("^GSPC", "2020-01-01", "2020-12-31")
        df_clean = loader.validate_data(df, "Test")

        # Deve ter removido 1 linha
        assert len(df_clean) == 9
        assert df_clean.isnull().sum().sum() == 0

    def test_validate_data_removes_duplicates(self):
        """Testa que validate_data remove duplicatas de índice."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'Close': np.random.randn(10),
        }, index=dates)

        # Adicionar duplicata
        df = pd.concat([df, df.iloc[[0]]])

        loader = DataLoader("^GSPC", "2020-01-01", "2020-12-31")
        df_clean = loader.validate_data(df, "Test")

        # Deve ter removido duplicata
        assert df_clean.index.duplicated().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
