"""Implementation of DataConnector class for fetching data from various sources."""
import sqlite3
from typing import Union, Dict, Optional
from pathlib import Path
import pandas as pd

required_modules = {
    "google_auth": "google-auth==2.25.2",
    "redshift_connector": "redshift_connector==2.0.918",
    "bigquery": "google-cloud-bigquery==3.14.1",
    "mysql_connector": "mysql-connector-python==8.2.0",
    "snowflake_connector": "snowflake-connector-python==3.6.0",
    "pandas_gbq": "pandas-gbq==0.20.0",
    "postgres": "psycopg2-binary==2.9.9",
}


class MissingModuleError(Exception):
    """Exception raised when a required module is missing."""

    def __init__(self, modules: Dict[str, str]):
        self.required_modules = modules
        super().__init__(self._format_message())

    def _format_message(self):
        missing_modules = ", ".join(
            f"{mod}: {ver}" for mod, ver in self.required_modules.items()
        )
        return (
            f"Missing required module versions: {missing_modules}. Please install them."
        )


class DataConnector:
    """Implementation of the DataConnector class for fetching data from various sources."""

    def fetch_dataframe_from_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Fetches data from a CSV file and returns a pandas DataFrame.

        Parameters:
        - file_path (Path): Path to the CSV file.

        Raises:
        - RuntimeError: If an error occurs while reading the CSV file.

        Returns:
        - DataFrame: Pandas DataFrame containing the data from the CSV file.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while reading CSV file '{file_path}': {e}"
            ) from e

    def fetch_dataframe_from_excel(
        self, file_path: Path, sheet_name: Union[int, str] = 0
    ) -> pd.DataFrame:
        """
        Fetches data from an Excel file and returns a pandas DataFrame.

        Parameters:
        - file_path (Path): Path to the Excel file.
        - sheet_name (Union[int, str], optional): Sheet name or index. Defaults to 0.

        Raises:
        - RuntimeError: If an error occurs while reading the Excel file.

        Returns:
        - DataFrame: Pandas DataFrame containing the data from the Excel file.
        """
        try:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while reading Excel file '{file_path}': {e}"
            ) from e

    def fetch_dataframe_from_redshift(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        schema: str,
        table: str,
        port: int = 5439,
    ) -> pd.DataFrame:
        """
        Fetches data from a Redshift database table into a pandas DataFrame.

        Parameters:
        - host (str): Redshift cluster endpoint address.
        - database (str): Database name.
        - user (str): Username for the Redshift cluster.
        - password (str): Password for the Redshift cluster.
        - schema (str): Schema where the table resides.
        - table (str): Table name to query data from.
        - port (int, optional): Port number. Defaults to 5439.

        Returns:
        - DataFrame: Pandas DataFrame containing the data from the Redshift table.

        Raises:
        - MissingModuleError: Raised when the 'redshift_connector' module is missing.
        - RuntimeError: Raised if an error occurs during connection or fetching data.
        """
        try:
            import redshift_connector
        except ImportError as exc:
            module_key = "redshift_connector"
            raise MissingModuleError(
                {module_key: required_modules[module_key]}
            ) from exc

        try:
            with redshift_connector.connect(
                host=host, database=database, user=user, password=password, port=port
            ) as conn:
                cursor = conn.cursor()

                full_table_name = f'"{schema}"."{table}"'
                query = f"SELECT * FROM {full_table_name};"
                cursor.execute(query)

                dataframe = cursor.fetch_dataframe()
                return dataframe

        except redshift_connector.InterfaceError as e:
            raise RuntimeError(
                f"Error occurred while connecting to Redshift database: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while fetching data from Redshift table: {e}"
            ) from e

    def fetch_dataframe_from_postgres(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        schema: str,
        table: str,
        port: int = 5432,
    ) -> pd.DataFrame:
        """
        Fetches data from a PostgreSQL database table into a pandas DataFrame.

        Parameters:
        - host (str): PostgreSQL server host.
        - database (str): Database name.
        - user (str): User name used to authenticate.
        - password (str): Password used to authenticate.
        - schema (str): Schema in which the target table resides.
        - table (str): Table from which to fetch the data.
        - port (int, optional): Port number. Defaults to 5432.

        Returns:
        - DataFrame: Pandas DataFrame containing the data from the PostgreSQL table.

        Raises:
        - MissingModuleError: Raised when the 'psycopg2' module is missing.
        - RuntimeError: Raised if an error occurs during connection or data fetching.
        """
        try:
            import psycopg2
        except ImportError as exc:
            raise MissingModuleError(
                {"postgres": required_modules["postgres"]}
            ) from exc

        try:
            with psycopg2.connect(
                host=host, database=database, user=user, password=password, port=port
            ) as connection:
                cursor = connection.cursor()

                from psycopg2.extensions import AsIs

                cursor.execute("SELECT * FROM %s.%s;", (AsIs(schema), AsIs(table)))

                table_contents = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                table_df = pd.DataFrame(table_contents, columns=column_names)
                return table_df

        except psycopg2.Error as e:
            raise RuntimeError(
                "Unable to connect to the PostgreSQL database. Please ensure the database details are correct."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while fetching data from PostgreSQL table: {e}"
            ) from e

    def fetch_dataframe_from_bigquery(
        self,
        dataset: str,
        table: str,
        project_id: str,
        credentials_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Fetches data from a BigQuery table into a pandas DataFrame.

        Parameters:
        - dataset (str): BigQuery dataset name.
        - table (str): BigQuery table name.
        - project_id (str): Google Cloud project ID.
        - credentials_path (Optional[Path]): Path to the service account key file. If
          None, the default credentials will be used.

        Returns:
        - DataFrame: Pandas DataFrame containing data from the BigQuery table.

        Raises:
        - MissingModuleError: Raised when the required modules are missing.
        - RuntimeError: Raised if an error occurs during data fetching.
        """
        try:
            from google.oauth2 import service_account
            import pandas_gbq
        except ImportError as exc:
            raise MissingModuleError(
                {
                    "google_auth": required_modules["google_auth"],
                    "pandas_gbq": required_modules["pandas_gbq"],
                }
            ) from exc

        sql_query = f"""
        SELECT * 
        FROM `{project_id}.{dataset}.{table}`
        """

        try:
            if credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                return pandas_gbq.read_gbq(
                    sql_query, project_id=project_id, credentials=credentials
                )
            else:
                return pandas_gbq.read_gbq(sql_query, project_id=project_id)

        except Exception as e:
            raise RuntimeError(
                f"Error occurred while fetching data from BigQuery: {e}"
            ) from e

    def fetch_dataframe_from_snowflake(
        self,
        user: str,
        password: str,
        account: str,
        warehouse: str,
        database: str,
        schema: str,
        table: str,
    ) -> pd.DataFrame:
        """
        Fetches data from a Snowflake table into a pandas DataFrame.

        Parameters:
        - user (str): Username for the Snowflake database.
        - password (str): Password for the Snowflake database.
        - account (str): Account identifier for the Snowflake database.
        - warehouse (str): Warehouse name to use for the Snowflake session.
        - database (str): Database name to connect to.
        - schema (str): Schema name where the table is located.
        - table (str): Table name to fetch data from.

        Returns:
        - DataFrame: Pandas DataFrame containing data from the Snowflake table.

        Raises:
        - MissingModuleError: Raised when the 'snowflake-connector-python' module is missing.
        - RuntimeError: Raised if an error occurs during data fetching or connection.
        """
        try:
            import snowflake.connector
        except ImportError as exc:
            raise MissingModuleError(
                {"snowflake_connector": required_modules["snowflake_connector"]}
            ) from exc

        try:
            with snowflake.connector.connect(
                user=user,
                password=password,
                account=account,
                warehouse=warehouse,
                database=database,
                schema=schema,
            ) as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM IDENTIFIER(%(database)s) IDENTIFIER(%(schema)s) IDENTIFIER(%(table)s);"
                cursor.execute(
                    query, {"database": database, "schema": schema, "table": table}
                )
                table_contents = cursor.fetchall()

                column_names = [desc[0] for desc in cursor.description]
                table_df = pd.DataFrame(table_contents, columns=column_names)
                return table_df

        except snowflake.connector.Error as e:
            raise RuntimeError(
                f"Error occurred while connecting or fetching data from Snowflake: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while fetching data from Snowflake table: {e}"
            ) from e

    def fetch_dataframe_from_mysql(
        self, user: str, password: str, host: str, db_name: str, table: str
    ) -> pd.DataFrame:
        """
        Fetches data from a MySQL table into a pandas DataFrame.

        Parameters:
        - user (str): Username for authenticating with MySQL.
        - password (str): Password for authenticating with MySQL.
        - host (str): Host address of the MySQL server.
        - db_name (str): Name of the database to use.
        - table (str): Table name to fetch data from.

        Returns:
        - DataFrame: Pandas DataFrame containing data from the MySQL table.

        Raises:
        - MissingModuleError: If the mysql-connector-python module is missing.
        - RuntimeError: If an error occurs during connection or data fetching.
        """
        try:
            from mysql.connector import connect, Error as MySQLError
        except ImportError as exc:
            raise MissingModuleError(
                {"mysql_connector": required_modules["mysql_connector"]}
            ) from exc

        try:
            connection_config = {
                "user": user,
                "password": password,
                "host": host,
                "database": db_name,
            }
            with connect(**connection_config) as conn:
                cursor = conn.cursor()
                # Parameterized queries should be used to prevent SQL injection
                cursor.execute("SELECT * FROM %s;", (table,))

                columns = [desc[0] for desc in cursor.description]
                df = pd.DataFrame(cursor.fetchall(), columns=columns)

                return df

        except MySQLError as e:
            raise RuntimeError(
                f"Error occurred while connecting to MySQL database: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while fetching data from MySQL table: {e}"
            ) from e

    def fetch_dataframe_from_sqlite(self, db_path: Path, table: str) -> pd.DataFrame:
        """
        Fetches data from an SQLite table into a pandas DataFrame.

        Parameters:
        - db_path (Path): Path to the SQLite database file.
        - table (str): Table name to fetch data from.

        Returns:
        - DataFrame: Pandas DataFrame containing data from the SQLite table.

        Raises:
        - RuntimeError: If an error occurs during connection or data fetching.
        """
        try:
            if not table.isidentifier():
                raise ValueError(f"Invalid table name: {table}")

            with sqlite3.connect(str(db_path)) as conn:
                query = f"SELECT * FROM `{table}`"
                df = pd.read_sql_query(query, conn)
            return df

        except sqlite3.Error as e:
            raise RuntimeError(
                f"Error occurred while connecting to SQLite database: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while fetching data from SQLite table: {e}"
            ) from e
