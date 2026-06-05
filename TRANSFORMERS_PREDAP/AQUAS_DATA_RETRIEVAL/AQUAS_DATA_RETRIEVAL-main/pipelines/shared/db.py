"""Database connection utilities for SQL Server/Synapse."""
import pyodbc


def get_connection(
    db_server: str,
    db_database: str,
    auth_mode: str = "ActiveDirectoryIntegrated",
    timeout: int = 60,
) -> pyodbc.Connection:
    """
    Create a connection to SQL Server or Azure Synapse.
    
    Args:
        db_server: Server address (e.g., 'server.sql.azuresynapse.net')
        db_database: Database name
        auth_mode: Authentication mode (default: ActiveDirectoryIntegrated)
        timeout: Connection timeout in seconds
        
    Returns:
        pyodbc.Connection: Database connection
    """
    connection_string = (
        "Driver={ODBC Driver 18 for SQL Server};"
        f"Server={db_server};"
        "Port=1433;"
        f"Database={db_database};"
        f"Authentication={auth_mode};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
    )
    return pyodbc.connect(connection_string, timeout=timeout)
