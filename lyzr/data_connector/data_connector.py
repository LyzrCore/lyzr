from typing import Optional, Union
import pandas as pd
import redshift_connector
import psycopg2
import pandas_gbq
from google.cloud import bigquery
from google.oauth2 import service_account

class DataConnector:
    def __init__(self):
        pass

    def fetch_dataframe_from_csv(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise RuntimeError(f"Error occurred while reading csv file: {str(e)}")


    def fetch_dataframe_from_excel(self, file_path: str, sheet_name: Union[int, str] = 0) -> Optional[pd.DataFrame]:
        try:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            raise RuntimeError(f"Error occurred while reading excel file: {str(e)}")


    def fetch_dataframe_from_redshift(self, host: str, database: str, user:str, password: str, schema: str, table:str, port: int=5439) -> pd.DataFrame:            
        try:   
            conn = redshift_connector.connect(
                host=host,
                database=database,
                port=port,
                user=user,
                password=password
            )

            cursor = conn.cursor()

            full_table_name = f'"{database}"."{schema}"."{table}"'
            cursor.execute(f'SELECT * FROM {full_table_name};')
        
            dataframe: pd.DataFrame = cursor.fetch_dataframe()
            return dataframe

        except redshift_connector.InterfaceError as e:
            raise RuntimeError(f"Error occured while connecting to Redshift database: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error occurred while fetching data from RedShift table: {str(e)}")    
            

    def fetch_dataframe_from_postgres(self, host: str, database: str, user: str, password: str, schema: str, table: str, port: int = 5432) -> pd.DataFrame:
        try:
            connection = psycopg2.connect(host=host, database=database, port=port, user=user, password=password)
            cursor = connection.cursor()
            cursor.execute(f"SELECT * FROM {schema}.{table};")
            table_contents = cursor.fetchall()
            
            column_names = [desc[0] for desc in cursor.description]
            table_df = pd.DataFrame(table_contents, columns=column_names)
            return table_df
        except psycopg2.Error:
            raise RuntimeError(f"Unable to connect to PostgreSQL database. Please ensure the database details are correct.")
        except Exception as e:
            raise RuntimeError(f"Error occurred while fetching data from PostgreSQL table: {str(e)}")
        
    

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
            raise RuntimeError(f"Error occurred while fetching data from BigQuery: {str(e)}")