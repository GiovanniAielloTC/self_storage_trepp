"""Configuration management for Snowflake connection."""
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class SnowflakeConfig:
    """Snowflake connection configuration."""
    
    user: str = None
    password: str = None
    account: str = None
    warehouse: str = None
    database: str = None
    schema: str = None
    role: Optional[str] = None
    query_timeout: int = 300
    
    def __post_init__(self):
        """Load from environment variables if not set."""
        self.user = self.user or os.getenv('SNOWFLAKE_USER')
        self.password = self.password or os.getenv('SNOWFLAKE_PASSWORD')
        self.account = self.account or os.getenv('SNOWFLAKE_ACCOUNT')
        self.warehouse = self.warehouse or os.getenv('SNOWFLAKE_WAREHOUSE')
        self.database = self.database or os.getenv('SNOWFLAKE_DATABASE')
        self.schema = self.schema or os.getenv('SNOWFLAKE_SCHEMA')
        self.role = self.role or os.getenv('SNOWFLAKE_ROLE')
        self.query_timeout = int(os.getenv('SNOWFLAKE_QUERY_TIMEOUT', 300))
    
    def validate(self) -> bool:
        """Validate that required configuration is present."""
        required = ['user', 'password', 'account', 'warehouse', 'database', 'schema']
        missing = [f for f in required if not getattr(self, f)]
        if missing:
            raise ValueError(f"Missing required Snowflake configuration: {', '.join(missing)}")
        return True
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for snowflake.connector.connect()."""
        params = {
            'user': self.user,
            'password': self.password,
            'account': self.account,
            'warehouse': self.warehouse,
            'database': self.database,
            'schema': self.schema,
        }
        if self.role:
            params['role'] = self.role
        return params


# Default configuration instance
snowflake_config = SnowflakeConfig()
