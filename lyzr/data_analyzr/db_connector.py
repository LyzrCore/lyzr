# standard library imports
import os
import sqlite3
import requests
from typing import Union
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlparse

# third-party imports
import pandas as pd

# local imports
from lyzr.base.errors import (
    ImproperlyConfigured,
    ValidationError,
    DependencyError,
)


required_modules = {
    "redshift_connector": "redshift_connector==2.0.918",
    "mysql.connector": "mysql-connector-python==8.2.0",
    "snowflake.connector": "snowflake-connector-python==3.6.0",
    "psycopg2": "psycopg2-binary==2.9.9",
}


@dataclass
class TrainingPlanItem:
    item_type: str
    item_group: str
    item_name: str
    item_value: str

    def __str__(self):
        if self.item_type == self.ITEM_TYPE_SQL:
            return f"Train on SQL: {self.item_group} {self.item_name}"
        elif self.item_type == self.ITEM_TYPE_DDL:
            return f"Train on DDL: {self.item_group} {self.item_name}"
        elif self.item_type == self.ITEM_TYPE_IS:
            return f"Train on Information Schema: {self.item_group} {self.item_name}"

    ITEM_TYPE_SQL = "sql"
    ITEM_TYPE_DDL = "ddl"
    ITEM_TYPE_IS = "is"


class TrainingPlan:
    _plan: list[TrainingPlanItem]

    def __init__(self, plan: list[TrainingPlanItem]):
        self._plan = plan

    def __str__(self):
        return "\n".join(self.get_summary())

    def __repr__(self):
        return self.__str__()

    def get_summary(self) -> list[str]:
        return [f"{item}" for item in self._plan]

    def remove_item(self, item: str):
        for plan_item in self._plan:
            if str(plan_item) == item:
                self._plan.remove(plan_item)
                break


def import_modules(modules: dict[str, str]):
    imported_modules = []
    for mod, ver in modules.items():
        try:
            globals()[mod] = __import__(mod)
        except ImportError:
            raise DependencyError({mod: ver})
        imported_modules.append(globals()[mod])
    return imported_modules


class DatabaseConnector:
    def __init__(self):
        """Parent class for all database connectors."""

    def fetch_dataframes_dict(self, **kwargs):
        raise NotImplementedError("You need to connect to a database first.")

    def fetch_dataframes_info_dict(self, **kwargs):
        raise NotImplementedError("You need to connect to a database first.")

    def run_sql(self, **kwargs):
        raise NotImplementedError("You need to connect to a database first.")

    def get_dbschema(self):
        raise NotImplementedError("You need to connect to a database first.")

    def get_schema_names(self):
        raise NotImplementedError("You need to connect to a database first.")

    @staticmethod
    def get_connector(db_type):
        if db_type == "redshift":
            return RedshiftConnector
        elif db_type == "postgres":
            return PostgresConnector
        elif db_type == "sqlite":
            return SQLiteConnector
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def get_default_training_plan(self):
        db_schema = self.get_dbschema()
        # For each of the following, we look at the df columns to see if there's a match:
        database_column = db_schema.columns[
            db_schema.columns.str.lower().str.contains("database")
            | db_schema.columns.str.lower().str.contains("table_catalog")
        ].to_list()[0]
        database_list = db_schema[database_column].unique().tolist()
        schema_column = db_schema.columns[
            db_schema.columns.str.lower().str.contains("table_schema")
        ].to_list()[0]
        schema_list = db_schema[schema_column].unique().tolist()
        table_column = db_schema.columns[
            db_schema.columns.str.lower().str.contains("table_name")
        ].to_list()[0]
        table_list = db_schema[table_column].unique().tolist()
        column_column = db_schema.columns[
            db_schema.columns.str.lower().str.contains("column_name")
        ].to_list()[0]
        data_type_column = db_schema.columns[
            db_schema.columns.str.lower().str.contains("data_type")
        ].to_list()[0]

        plan = TrainingPlan([])
        for database in database_list:
            for table_schema in schema_list:
                for table in table_list:
                    df_columns_filtered_to_table = db_schema.query(
                        f'{database_column} == "{database}" and {schema_column} == "{table_schema}" and {table_column} == "{table}"'
                    )
                    doc = f"The following columns are in the {table} table in the {database} database:\n\n"
                    doc += df_columns_filtered_to_table[
                        [
                            database_column,
                            schema_column,
                            table_column,
                            column_column,
                            data_type_column,
                        ]
                    ].to_markdown()
                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_IS,
                            item_group=f"{database}.{table_schema}",
                            item_name=table,
                            item_value=doc,
                        )
                    )

                    doc = f"Following are the first five rows of the {table} table in the {database} database:\n\n"
                    doc += (
                        self.run_sql(
                            f"SELECT * FROM {database}.{table_schema}.{table} LIMIT 5"
                        )
                        .fillna("nan")
                        .to_markdown(
                            floatfmt=".2f",
                            index=False,
                        )
                    )
                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_IS,
                            item_group=f"{database}.{table_schema}",
                            item_name=table,
                            item_value=doc,
                        )
                    )

        return plan


class PostgresConnector(DatabaseConnector):

    def __init__(
        self,
        host: str,
        port: Union[int, str],
        database: str,
        user: str,
        password: str,
        schema: Union[list, str] = None,
        tables: list[str] = None,
    ):
        self.host = host or os.getenv("POSTGRES_HOST")
        self.port = port or os.getenv("POSTGRES_PORT")
        self.database = database or os.getenv("POSTGRES_DB")
        self.user = user or os.getenv("POSTGRES_USER")
        self.password = password or os.getenv("POSTGRES_PASSWORD")
        if not all([self.host, self.port, self.database, self.user, self.password]):
            raise ImproperlyConfigured(
                "Please provide all required environment variables for Postgres connection: POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD"
            )
        self.schema = schema or os.getenv("POSTGRES_SCHEMA") or None
        if isinstance(self.schema, str):
            self.schema = [self.schema]
        self.tables = tables or os.getenv("POSTGRES_TABLES") or None
        (self.psycopg2,) = import_modules({"psycopg2": required_modules["psycopg2"]})
        try:
            self.conn = self.psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
            )
        except self.psycopg2.Error as e:
            self.conn = None
            raise ValidationError(
                f"Error occurred while connecting to PostgreSQL database: {e}"
            ) from e

        if self.schema is None:
            self.schema = self.get_schema_names()

    def fetch_dataframes_dict(
        self,
        schema: Union[str, list] = None,
        tables: list[str] = None,
    ) -> dict[pd.DataFrame]:
        schema = schema or self.schema or None
        if isinstance(schema, str):
            schema = [schema]
        if schema is None:
            schema = self.get_schema_names()
        tables = tables or self.tables
        if self.conn is None:
            raise RuntimeError("No connection to the database.")
        try:
            cursor = self.conn.cursor()
            as_is = self.psycopg2.extensions.AsIs

            # Fetch all table names in the schema
            if tables is None:
                query = "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA in (%s);"
                schema_names = ", ".join([f"'{s}'" for s in schema])
                cursor.execute(query, (as_is(schema_names),))
                tables = cursor.fetchall()
                tables = list({f"'{table[0]}'.'{table[1]}'" for table in tables})

            # Fetch all table contents and store in a dictionary as pandas dataframes
            dataframes = {tablename: "SELECT * FROM %s;" for tablename in tables}
            for tablename, query in dataframes.items():
                cursor.execute(query, (as_is(tablename),))
                dataframes[tablename] = pd.DataFrame(
                    cursor.fetchall(), columns=[desc[0] for desc in cursor.description]
                )
            # return the dataframes and the connection object
            return dataframes
        except self.psycopg2.Error as e:
            self.conn.rollback()
            raise ValidationError(
                f"Error occurred while connecting to PostgreSQL database: {e}"
            ) from e
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(
                f"Error occurred while fetching data from PostgreSQL table: {e}"
            ) from e

    def get_schema_names(self):
        all_schema_names = self.run_sql(
            "SELECT table_schema FROM information_schema.columns;"
        ).table_schema.unique()
        schema_names = [
            schema
            for schema in all_schema_names
            if schema != "information_schema" and not schema.startswith("pg_")
        ]
        if len(schema_names) == 0:
            raise ValidationError("No schema found in the database.")
        return schema_names

    def get_dbschema(self):
        schema_names = ", ".join([f"'{s}'" for s in self.schema])
        return self.run_sql(
            f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA in ({schema_names});"
        )

    def run_sql(self, sql: str) -> Union[pd.DataFrame, None]:
        if self.conn is None:
            raise RuntimeError("No connection to the database.")
        try:
            cs = self.conn.cursor()
            cs.execute(sql)
            results = cs.fetchall()
            # Create a pandas dataframe from the results
            df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
            return df

        except self.psycopg2.Error as e:
            self.conn.rollback()
            raise ValidationError(
                f"Error occurred while connecting to PostgreSQL database: {e}"
            ) from e

        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(
                f"Error occurred while executing query on PostgreSQL table: {e}"
            ) from e


class RedshiftConnector(DatabaseConnector):

    def __init__(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int,
        schema: str,
        tables: list[str] = None,
    ):
        self.host = host or os.getenv("REDSHIFT_HOST") or 5439
        self.port = port or os.getenv("REDSHIFT_PORT")
        self.database = database or os.getenv("REDSHIFT_DB")
        self.user = user or os.getenv("REDSHIFT_USER")
        self.password = password or os.getenv("REDSHIFT_PASSWORD")
        if not all([self.host, self.port, self.database, self.user, self.password]):
            raise ImproperlyConfigured(
                "Please provide all required environment variables for Redshift connection: REDSHIFT_HOST, REDSHIFT_PORT, REDSHIFT_DB, REDSHIFT_USER, REDSHIFT_PASSWORD"
            )
        self.schema = schema or os.getenv("REDSHIFT_SCHEMA") or None
        if isinstance(self.schema, str):
            self.schema = [self.schema]
        self.tables = tables or os.getenv("REDSHIFT_TABLES") or None
        (self.redshift_connector,) = import_modules(
            {"redshift_connector": required_modules["redshift_connector"]}
        )
        try:
            self.conn = self.redshift_connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
            )
        except self.redshift_connector.InterfaceError as e:
            self.conn = None
            raise ValidationError(
                f"Error occurred while connecting to Redshift database: {e}"
            ) from e

        if self.schema is None:
            self.schema = self.get_schema_names()

    def fetch_dataframes_dict(
        self,
        schema: str = None,
        tables: list[str] = None,
    ) -> dict[pd.DataFrame]:
        schema = schema or self.schema or ["public"]
        tables = tables or self.tables
        if self.conn is None:
            raise RuntimeError("No connection to the database.")
        try:
            cursor = self.conn.cursor()
            # Fetch all table names in the schema
            if tables is None:
                schema_names = ", ".join([f"'{s}'" for s in schema])
                query = f"SELECT schemaname, tablename FROM pg_table_def WHERE schemaname in ({schema_names});"
                cursor.execute(query)
                tables = cursor.fetchall()
                tables = list({f"'{table[0]}'.'{table[1]}'" for table in tables})

            # Fetch all table contents and store in a dictionary as pandas dataframes
            dataframes = {
                tablename: f"SELECT * FROM {tablename};" for tablename in tables
            }
            for tablename, query in dataframes.items():
                cursor.execute(query)
                dataframes[tablename] = pd.DataFrame(
                    cursor.fetchall(), columns=[desc[0] for desc in cursor.description]
                )
            return dataframes
        except self.redshift_connector.InterfaceError as e:
            self.conn.rollback()
            raise ValidationError(
                f"Error occurred while connecting to Redshift database: {e}"
            ) from e
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(
                f"Error occurred while fetching data from Redshift table: {e}"
            ) from e

    def get_schema_names(self):
        all_schema_names = self.run_sql(
            "SELECT table_schema FROM information_schema.columns;"
        ).table_schema.unique()
        schema_names = [
            schema
            for schema in all_schema_names
            if schema != "information_schema" and not schema.startswith("pg_")
        ]
        if len(schema_names) == 0:
            raise ValidationError("No schema found in the database.")
        return schema_names

    def get_dbschema(self):
        schema_names = ", ".join([f"'{s}'" for s in self.schema])
        return self.run_sql(
            f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA in ({schema_names});"
        )

    def run_sql(self, sql: str) -> Union[pd.DataFrame, None]:
        if self.conn is None:
            raise RuntimeError("No connection to the database.")
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            # Create a pandas dataframe from the results
            df = pd.DataFrame(results, columns=[desc[0] for desc in cursor.description])
            return df
        except self.redshift_connector.InterfaceError as e:
            self.conn.rollback()
            raise ValidationError(
                f"Error occurred while connecting to Redshift database: {e}"
            ) from e
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(
                f"Error occurred while executing query on Redshift table: {e}"
            ) from e


class SQLiteConnector(DatabaseConnector):
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("SQLITE_DB_PATH")
        self.db_path = SQLiteConnector._download_db(self.db_path)
        self.conn = None
        try:
            if self.db_path is not None:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        except sqlite3.Error as e:
            raise ValidationError(
                f"Error occurred while connecting to SQLite database: {e}"
            ) from e

    @staticmethod
    def _download_db(url: str) -> Union[str, None]:
        url = urlparse(url).path
        if os.path.exists(url):
            return url
        try:
            filepath = os.path.basename(url)
            response = requests.get(url)
            response.raise_for_status()  # Check that the request was successful
            with open(filepath, "wb") as f:
                f.write(response.content)
            return filepath
        except Exception:
            return url

    def create_database(self, db_path: str, df_dict: dict[pd.DataFrame]):
        self.db_path = urlparse(db_path).path or self.db_path or ":memory:"

        parent = Path(self.db_path).parent
        os.makedirs(parent, exist_ok=True)

        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            for name, df in df_dict.items():
                df = df.rename(
                    columns=dict(
                        zip(
                            df.columns,
                            [col.replace(" ", "_").lower() for col in df.columns],
                        )
                    )
                )
                df.to_sql(name, con=self.conn, index=False, if_exists="replace")
            return self.conn
        except sqlite3.Error as e:
            raise ValidationError(
                f"Error occurred while creating SQLite database: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while executing query on SQLite database: {e}"
            ) from e

    def fetch_dataframes_dict(self, tables: list[str] = None) -> dict[pd.DataFrame]:
        if self.conn is None:
            raise RuntimeError("No connection to the database.")
        try:
            if tables is None:
                cursor = self.conn.cursor()
                cursor.execute("SELECT name FROM sqlite_schema WHERE type='table';")
                tables = cursor.fetchall()
                tables = [table[0] for table in tables]
            dataframes = {
                table: pd.read_sql(f"SELECT * FROM {table};", con=self.conn)
                for table in tables
            }
            return dataframes
        except sqlite3.Error as e:
            raise ValidationError(
                f"Error occurred while fetching data from SQLite database: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while executing query on SQLite database: {e}"
            ) from e

    def run_sql(self, sql: str) -> Union[pd.DataFrame, None]:
        if self.conn is None:
            raise RuntimeError("No connection to the database.")
        try:
            df = pd.read_sql(sql, con=self.conn)
            return df
        except sqlite3.Error as e:
            raise ValidationError(
                f"Error occurred while executing query on SQLite database: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while executing query on SQLite database: {e}"
            ) from e

    def get_default_training_plan(self):

        tables_schema = self.run_sql("SELECT * FROM sqlite_master WHERE type='table';")
        plan = TrainingPlan([])

        for table_idx in tables_schema.index:
            table_info = tables_schema.loc[table_idx]

            table_columns = self.run_sql(
                f"PRAGMA table_info('{table_info['tbl_name']}')"
            )
            doc = (
                f"The following columns are in the {table_info['tbl_name']} table:\n\n"
            )
            doc += table_columns[["name", "type"]].to_markdown()

            plan._plan.append(
                TrainingPlanItem(
                    item_type=TrainingPlanItem.ITEM_TYPE_IS,
                    item_group=table_info["name"],
                    item_name=table_info["tbl_name"],
                    item_value=doc,
                )
            )

            doc = f"Following are the first five rows of the {table_info['tbl_name']} table:\n\n"
            doc += (
                self.run_sql(f"SELECT * FROM '{table_info['tbl_name']}' LIMIT 5")
                .fillna("nan")
                .to_markdown(
                    floatfmt=".2f",
                    index=False,
                )
            )
            plan._plan.append(
                TrainingPlanItem(
                    item_type=TrainingPlanItem.ITEM_TYPE_IS,
                    item_group=f"{table_info['name']}.public",
                    item_name=table_info["tbl_name"],
                    item_value=doc,
                )
            )
        return plan
