from __future__ import annotations
import duckdb
import pandas as pd
import datetime as dt
from dataclasses import dataclass
from typing import Optional


@dataclass
class PolygonCfg:
    api_key: str
    cache_path: str = "duckdb/polygon.duckdb"
    table: str = "daily_bars"


class PolygonLoader:
    """Real data loader for Polygon.io integration.
    
    This is a skeleton implementation that can be extended to pull real market data
    from Polygon.io and cache it in DuckDB for fast access.
    """
    
    def __init__(self, cfg: PolygonCfg):
        self.cfg = cfg
        self.con = duckdb.connect(cfg.cache_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        self.con.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.cfg.table} (
            date DATE,
            ticker VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            PRIMARY KEY (date, ticker)
        )
        """)

    def ingest_daily(self, tickers: list[str], start: str, end: str) -> None:
        """Ingest daily bars from Polygon.io and store in DuckDB.
        
        Args:
            tickers: List of ticker symbols to fetch
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
        """
        # TODO: Implement Polygon.io REST API calls
        # 1. Call Polygon aggregates/v2 endpoint for each ticker
        # 2. Parse response and upsert into DuckDB
        # 3. Handle rate limiting and pagination
        # 4. Add proper error handling and retries
        pass

    def load_close_panel(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        """Load close prices as a panel DataFrame.
        
        Args:
            tickers: List of ticker symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with date index, ticker columns, close prices as values
        """
        q = f"""
        SELECT date, ticker, close
        FROM {self.cfg.table}
        WHERE ticker IN ({','.join([f"'{t}'" for t in tickers])})
          AND date BETWEEN '{start}' AND '{end}'
        ORDER BY date
        """
        df = self.con.execute(q).fetchdf()
        if df.empty:
            raise ValueError(f"No data found for tickers {tickers} between {start} and {end}")
        
        df["date"] = pd.to_datetime(df["date"])
        return df.pivot(index="date", columns="ticker", values="close").sort_index()

    def get_available_tickers(self, start: Optional[str] = None, end: Optional[str] = None) -> list[str]:
        """Get list of available tickers in the database.
        
        Args:
            start: Optional start date filter
            end: Optional end date filter
            
        Returns:
            List of available ticker symbols
        """
        where_clause = ""
        if start and end:
            where_clause = f"WHERE date BETWEEN '{start}' AND '{end}'"
        elif start:
            where_clause = f"WHERE date >= '{start}'"
        elif end:
            where_clause = f"WHERE date <= '{end}'"
            
        q = f"SELECT DISTINCT ticker FROM {self.cfg.table} {where_clause} ORDER BY ticker"
        result = self.con.execute(q).fetchdf()
        return result["ticker"].tolist()

    def get_date_range(self, ticker: str) -> tuple[dt.date, dt.date]:
        """Get the available date range for a specific ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Tuple of (min_date, max_date)
        """
        q = f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM {self.cfg.table} WHERE ticker = '{ticker}'"
        result = self.con.execute(q).fetchdf()
        if result.empty or result.iloc[0]["min_date"] is None:
            raise ValueError(f"No data found for ticker {ticker}")
        
        return result.iloc[0]["min_date"], result.iloc[0]["max_date"]
