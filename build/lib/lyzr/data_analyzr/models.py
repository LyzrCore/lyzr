# standart-library imports
import logging
import warnings
from enum import Enum
from typing import Union

# third-party imports
from pydantic import BaseModel, Field

# local imports
from lyzr.base import LiteLLM

warnings.filterwarnings("ignore", category=UserWarning)


class AnalysisTypes(str, Enum):
    sql = "sql"
    ml = "ml"
    skip = "skip"


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


class ParamsDict(BaseModel):
    max_retries: Union[int, None] = Field(default=None)
    time_limit: Union[int, None] = Field(default=None)
    auto_train: Union[bool, None] = Field(default=True)


class FactoryBaseClass:
    from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore

    llm: LiteLLM
    context: str
    logger: logging.Logger
    vector_store: ChromaDBVectorStore
    output: dict
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
            max_retries=max_retries,
            time_limit=time_limit,
            auto_train=True if auto_train is None else auto_train,
        )
        self.code = None
        self.output = None
        self.guide = None
        if self.vector_store is None:
            raise ValueError("Vector store is required.")

    def run_analysis(self, user_input: str, **kwargs):
        raise NotImplementedError
