# standart-library imports
import warnings
from enum import Enum
from typing import Annotated, Union

# third-party imports
from pydantic import BaseModel, Field, Discriminator, Tag, AliasChoices


warnings.filterwarnings("ignore", category=UserWarning)


class SupportedDBs(str, Enum):
    files = "files"
    redshift = "redshift"
    postgres = "postgres"
    sqlite = "sqlite"


class FilesConfig(BaseModel):
    datasets: dict
    files_kwargs: Union[dict, None] = Field(default_factory=dict)
    db_path: Union[str, None] = Field(default=None)


class RedshiftConfig(BaseModel):
    name: str
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
    name: str
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


class VectorStoreConfig(BaseModel):
    path: Union[str, None] = Field(
        default=None,
        validation_alias=AliasChoices("vector_store_path", "vstore_path", "path"),
    )
    remake_store: Union[bool, None] = Field(
        default=False, validation_alias=AliasChoices("remake", "remake_store")
    )
