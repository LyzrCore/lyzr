"""
Pydantic models for database configurations.
"""

# standart-library imports
import warnings
from aenum import Enum
from typing import Union

# third-party imports
import pandas as pd
from pydantic import BaseModel, Field, AliasChoices, ConfigDict


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
        default=None, validation_alias=AliasChoices("lschema", "schema")
    )
    tables: Union[list, None] = Field(default=None)


class PostgresConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str
    schema: Union[list, None] = Field(
        default=None, validation_alias=AliasChoices("lschema", "schema")
    )
    tables: Union[list, None] = Field(default=None)


class SQLiteConfig(BaseModel):
    db_path: str


class DynamicConfigUnion:
    """
    A class to manage dynamic configuration types.
    It allows for the registration of configuration types and
    creation of instances based on the discriminator key.
    To be used as a mutable Union type for validation with Pydantic models.

    Procedure:
    1. Create an instance of the `DynamicConfigUnion` class.
    2. Register configuration types using the `add_config_type` method.
    3. Validate the data using the `validate` method.
    4. The `validate` method returns an instance of the appropriate configuration type.

    Usage:
        config = DynamicConfigUnion()
        config.add_config_type("db_type", DBConfig)
        config.validate({"db_type": "db_type", "key": "value"})
    """

    def __init__(self, key_name: str = None):
        self._config_types: dict[str, BaseModel] = {}
        self._discriminator_key = key_name or "db_type"

    def add_config_type(self, name, config_type: BaseModel):
        if name not in self._config_types:
            self._config_types[name] = config_type
        elif config_type != self._config_types[name]:
            raise ValueError(
                f"Config type with name '{name}' is already registered with {self._config_types[name]}"
            )

    def _create_instance(self, name, **kwargs) -> BaseModel:
        if name not in self._config_types:
            raise ValueError(f"Config type with name '{name}' is not registered")
        config_type = self._config_types[name]
        return config_type(**kwargs)

    def validate(self, data: dict) -> BaseModel:
        if self._discriminator_key not in data:
            raise ValueError(
                f"Data must contain the discriminator key '{self._discriminator_key}'"
            )
        name = data[self._discriminator_key]
        return self._create_instance(name, **data)


DataConfig = DynamicConfigUnion()
DataConfig.add_config_type(SupportedDBs.files, FilesConfig)
DataConfig.add_config_type(SupportedDBs.redshift, RedshiftConfig)
DataConfig.add_config_type(SupportedDBs.postgres, PostgresConfig)
DataConfig.add_config_type(SupportedDBs.sqlite, SQLiteConfig)

"""
Mutable union type for database configurations of supported types.

This type is used to validate and discriminate between different
database configurations based on the `db_type` field:
    - FilesConfig: Configuration model for file-based databases.
    - RedshiftConfig: Configuration model for Redshift databases.
    - PostgresConfig: Configuration model for Postgres databases.
    - SQLiteConfig: Configuration model for SQLite databases.

Usage:
    config = DataConfig.validate({"db_type": "files", "datasets": [...], "db_path": "path/to/db"})
    assert isinstance(config, FilesConfig)
"""
