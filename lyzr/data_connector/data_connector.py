from typing import Optional, Union
import pandas as pd
import redshift_connector
import psycopg2
import pandas_gbq
from google.cloud import bigquery
from google.oauth2 import service_account
import snowflake.connector
import mysql.connector
import sqlite3


class DataConnector:
    def __init__(self):
        pass

    def fetch_dataframe_from_csv(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while reading csv file: {str(e)}")

    def fetch_dataframe_from_excel(self, file_path: str, sheet_name: Union[int, str] = 0) -> Optional[pd.DataFrame]:
        try:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while reading excel file: {str(e)}")

    def fetch_dataframe_from_redshift(self, host: str, database: str, user: str, password: str, schema: str, table: str, port: int = 5439) -> pd.DataFrame:
        try:
            with redshift_connector.connect(
                host=host,
                database=database,
                port=port,
                user=user,
                password=password
            ) as conn:

                cursor = conn.cursor()

                full_table_name = f'"{database}"."{schema}"."{table}"'
                cursor.execute(f'SELECT * FROM {full_table_name};')

                dataframe: pd.DataFrame = cursor.fetch_dataframe()
                return dataframe

        except redshift_connector.InterfaceError as e:
            raise RuntimeError(
                f"Error occured while connecting to Redshift database: {str(e)}")
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while fetching data from RedShift table: {str(e)}")

    def fetch_dataframe_from_postgres(self, host: str, database: str, user: str, password: str, schema: str, table: str, port: int = 5432) -> pd.DataFrame:
        try:
             with psycopg2.connect(
                host=host, database=database, port=port, user=user, password=password) as connection:
            
                cursor = connection.cursor()
                cursor.execute(f"SELECT * FROM {schema}.{table};")
                table_contents = cursor.fetchall()

                column_names = [desc[0] for desc in cursor.description]
                table_df = pd.DataFrame(table_contents, columns=column_names)
                return table_df
             
        except psycopg2.Error:
            raise RuntimeError(
                f"Unable to connect to PostgreSQL database. Please ensure the database details are correct.")
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while fetching data from PostgreSQL table: {str(e)}")

    def fetch_dataframe_from_bigquery(self, dataset: str, table: str, project_id: str, credentials_path: str = None) -> pd.DataFrame:
        try:
            sql_query = f"""
            SELECT * 
            FROM `{project_id}.{dataset}.{table}`
            """

            if credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                return pandas_gbq.read_gbq(sql_query, project_id=project_id, credentials=credentials)
            else:
                return pandas_gbq.read_gbq(sql_query, project_id=project_id)

        except Exception as e:
            raise RuntimeError(
                f"Error occurred while fetching data from BigQuery: {str(e)}")

    def fetch_dataframe_from_snowflake(self, user: str, password: str, account: str,  warehouse: str, database: str, schema: str, table: str) -> pd.DataFrame:
        try:
            with snowflake.connector.connect(
                user=user,
                password=password,
                account=account,
                warehouse=warehouse,
                database=database,
                schema=schema,
                table=table,
            ) as conn:

                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table};")
                table_contents = cursor.fetchall()

                column_names = [desc[0] for desc in cursor.description]
                table_df = pd.DataFrame(table_contents, columns=column_names)
                return table_df
            
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while fetching data from Snowflake table: {str(e)}")

    def fetch_dataframe_from_mysql(self, user: str, password: str, host: str, db_name: str, table: str) -> pd.DataFrame:
        try:
            with mysql.connector.connect(
                user=user, password=password, host=host) as conn:

                cursor = conn.cursor()
                cursor.execute(f"USE {db_name}")
                cursor.execute(f"SELECT * FROM {table}")

                columns = [desc[0] for desc in cursor.description]
                df = pd.DataFrame(cursor.fetchall(), columns=columns)

                return df

        except mysql.connector.Error as e:
            raise RuntimeError(
                f"Error occurred while connecting to MySQL database: {str(e)}")
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while fetching data from MySQL table: {str(e)}")

    def fetch_dataframe_from_sqlite(self, db_path: str, table: str) -> pd.DataFrame:
        try:
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            return df
        except sqlite3.Error as e:
            raise RuntimeError(
                f"Error occurred while connecting to SQLite database: {str(e)}")
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while fetching data from SQLite table: {str(e)}")
        
    