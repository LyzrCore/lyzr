from typing import Optional, Union
import redshift_connector
import pandas as pd
import psycopg2

class Connector:
    def __init__(self):
        pass

    def from_csv(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error occurred while reading csv file: {str(e)}")
            return None

    def from_excel(self, file_path: str, sheet_name: Union[int, str] = 0) -> Optional[pd.DataFrame]:
        try:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"Error occurred while reading excel file: {str(e)}")
            return None


    def from_redshift(self, host: str, database: str, user:str, password: str, schema: str, table:str, port: int=5439) -> pd.DataFrame:            
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
            print(f"Error occured while connecting to Redshift database: {str(e)}")
        except Exception as e:
            print(f"Error occurred while fetching data from RedShift table: {str(e)}")    
            

    def from_postgres(self, host: str, database: str, user: str, password: str, schema: str, table: str, port: int = 5432) -> pd.DataFrame:
        try:
            connection = psycopg2.connect(host=host, database=database, port=port, user=user, password=password)
            cursor = connection.cursor()
            cursor.execute(f"SELECT * FROM {schema}.{table};")
            table_contents = cursor.fetchall()
            
            column_names = [desc[0] for desc in cursor.description]
            table_df = pd.DataFrame(table_contents, columns=column_names)
            return table_df
        except psycopg2.Error:
            print(f"Unable to connect to PostgreSQL database. Please ensure the database details are correct.")
            return None
        except Exception as e:
            print(f"Error occurred while fetching data from PostgreSQL table: {str(e)}")
            return None