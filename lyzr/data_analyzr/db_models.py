"""
Pydantic models for database configurations.
"""

# standart-library imports
import warnings
from enum import Enum
from typing import Annotated, Union

# third-party imports
import pandas as pd
from pydantic import BaseModel, Field, Discriminator, Tag, AliasChoices, ConfigDict


warnings.filterwarnings("ignore", category=UserWarning)


class SupportedDBs(str, Enum):
    files = "files"
    redshift = "redshift"
    postgres = "postgres"
    sqlite = "sqlite"


class DataFile(BaseModel):
    name: str
    value: Union[str, pd.DataFrame]
    kwargs: Union[dict, None] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FilesConfig(BaseModel):
    datasets: list[DataFile]
    db_path: Union[str, None] = Field(default=None)


class RedshiftConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str
    schema: Union[list, None] = Field(
        default_factory=list, validation_alias=AliasChoices("lschema", "schema")
    )
    tables: Union[list, None] = Field(default_factory=list)


class PostgresConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str
    schema: Union[list, None] = Field(
        default_factory=list, validation_alias=AliasChoices("lschema", "schema")
    )
    tables: Union[list, None] = Field(default_factory=list)


class SQLiteConfig(BaseModel):
    db_path: str


DataConfig = Annotated[
    Union[
        Annotated[FilesConfig, Tag(SupportedDBs.files)],
        Annotated[RedshiftConfig, Tag(SupportedDBs.redshift)],
        Annotated[PostgresConfig, Tag(SupportedDBs.postgres)],
        Annotated[SQLiteConfig, Tag(SupportedDBs.sqlite)],
    ],
    Discriminator(lambda x: x["db_type"]),
]
"""
Union type for database configurations of supported types.

This type is used to validate and discriminate between different
database configurations based on the `db_type` field:
    - FilesConfig: Configuration model for file-based databases.
    - RedshiftConfig: Configuration model for Redshift databases.
    - PostgresConfig: Configuration model for Postgres databases.
    - SQLiteConfig: Configuration model for SQLite databases.

Usage:
    TypeAdapter(DataConfig).validate_python(db_config)
"""
