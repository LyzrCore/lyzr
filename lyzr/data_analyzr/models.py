# standart-library imports
import logging
import warnings
from enum import Enum
from typing import Annotated, Union

# third-party imports
from pydantic import BaseModel, Field, Discriminator, Tag

# local imports
from lyzr.base import LiteLLM
from lyzr.base.errors import MissingValueError
from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore

warnings.filterwarnings("ignore", category=UserWarning)


class AnalysisTypes(str, Enum):
    sql = "sql"
    ml = "ml"
    skip = "skip"


class ParamsDict(BaseModel):
    max_retries: int = Field(default=3)
    time_limit: int = Field(default=30)
    auto_train: bool = Field(default=True)


class FactoryBaseClass:
    llm: LiteLLM
    context: str
    logger: logging.Logger
    vector_store: ChromaDBVectorStore
    analysis_output: dict
    code: str
    additional_kwargs: dict
    max_retries: int
    time_limit: int
    params: ParamsDict
    steps: list

    def __init__(
        self,
        llm: LiteLLM,
        logger: logging.Logger,
        context: str,
        vector_store: ChromaDBVectorStore,
        max_retries: int,
        time_limit: int,
        auto_train: bool,
        llm_kwargs: dict,
    ) -> None:
        self.llm = llm
        model_kwargs = dict(seed=123, temperature=0.1, top_p=0.5)
        model_kwargs.update(llm_kwargs)
        self.llm.set_model_kwargs(model_kwargs=model_kwargs)
        self.context = context
        self.logger = logger
        self.vector_store = vector_store
        self.params = ParamsDict(
            max_retries=3 if max_retries is None else max_retries,
            time_limit=30 if time_limit is None else time_limit,
            auto_train=True if auto_train is None else auto_train,
        )
        self.code = None
        self.analysis_output = None
        self.analysis_guide = None
        self.steps = None
        if self.vector_store is None:
            raise MissingValueError("vector_store")

    def run_analysis(self, user_input: str, **kwargs):
        raise NotImplementedError


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
    schema: Union[list, None] = Field(default_factory=list, validation_alias="lschema")
    tables: Union[list, None] = Field(default_factory=list)


class PostgresConfig(BaseModel):
    name: str
    host: str
    port: int
    user: str
    password: str
    database: str
    schema: Union[list, None] = Field(default_factory=list, validation_alias="lschema")
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
    path: str = Field(default=None, validation_alias="vector_store_path")
    remake_store: bool = Field(default=False, validation_alias="remake")


class OutputTypes(str, Enum):
    visualisation = "visualisation"
    visualisations = "visualisation"
    vizualisation = "visualisation"
    vizualisations = "visualisation"
    visualization = "visualisation"
    visualizations = "visualisation"
    image = "visualisation"
    images = "visualisation"
    insights = "insights"
    insight = "insights"
    recommendations = "recommendations"
    recommendation = "recommendations"
    tasks = "tasks"
    task = "tasks"
