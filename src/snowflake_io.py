"""Snowflake database connection and query helpers for self-storage analysis."""
import snowflake.connector
from snowflake.connector import DictCursor
import pandas as pd
from typing import Optional, Union, Dict, Any
from contextlib import contextmanager
import logging
import time

from .config import snowflake_config

logger = logging.getLogger(__name__)


class SnowflakeConnection:
    """Manages Snowflake database connections and queries."""
    
    def __init__(self, config=None, max_retries: int = 3, retry_delay: int = 5):
        self.config = config or snowflake_config
        self.config.validate()
        self._connection = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def connect(self) -> snowflake.connector.SnowflakeConnection:
        """Establish connection to Snowflake with retry logic."""
        if self._connection is None or self._connection.is_closed():
            last_error = None
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Connecting to Snowflake (attempt {attempt + 1}/{self.max_retries})")
                    self._connection = snowflake.connector.connect(
                        **self.config.get_connection_params(),
                        login_timeout=120,
                        network_timeout=7200,
                        socket_timeout=7200,
                        client_session_keep_alive=True,
                    )
                    cursor = self._connection.cursor()
                    cursor.execute("ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 7200")
                    cursor.close()
                    return self._connection
                except snowflake.connector.errors.OperationalError as e:
                    last_error = e
                    if attempt < self.max_retries - 1:
                        logger.warning(f"Connection failed, retrying in {self.retry_delay}s... ({e})")
                        time.sleep(self.retry_delay)
                    else:
                        raise
        return self._connection
    
    def close(self):
        """Close the connection."""
        if self._connection and not self._connection.is_closed():
            self._connection.close()
            logger.info("Snowflake connection closed")
    
    @contextmanager
    def cursor(self, dict_cursor: bool = False):
        """Context manager for database cursors."""
        conn = self.connect()
        cursor_class = DictCursor if dict_cursor else None
        cursor = conn.cursor(cursor_class)
        try:
            yield cursor
        finally:
            cursor.close()
    
    def fetch_df(self, query: str, params: Optional[dict] = None) -> pd.DataFrame:
        """Execute query and return results as DataFrame."""
        conn = self.connect()
        logger.info(f"Executing query ({len(query)} chars)")
        start = time.time()
        
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            df = cursor.fetch_pandas_all()
            elapsed = time.time() - start
            logger.info(f"Query returned {len(df):,} rows in {elapsed:.1f}s")
            cursor.close()
            return df
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def execute(self, query: str, params: Optional[dict] = None):
        """Execute a query without returning results."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
        finally:
            cursor.close()
