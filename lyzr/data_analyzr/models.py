"""
Base classes and configuration models for the DataAnalyzr class.
"""

# standart-library imports
import json
import logging
import warnings
from enum import Enum
from typing import Union, Any

# third-party imports
from aenum import Enum as AEnum
from pydantic import BaseModel, Field, AliasChoices, ConfigDict, model_validator

# local imports
from lyzr.base import LiteLLM

warnings.filterwarnings("ignore", category=UserWarning)


class AnalysisTypes(str, AEnum):
    """A mutable enumeration of analysis types."""

    sql = "sql"
    ml = "ml"
    skip = "skip"


class OutputTypes(str, Enum):
    """An enumeration of output types."""

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
    """Configuration model for `params` attribute of factory classes and the DataAnalyzr class."""

    max_retries: Union[int, None] = Field(default=None)
    time_limit: Union[int, None] = Field(default=None)
    auto_train: Union[bool, None] = Field(default=True)


class ContextDict(BaseModel):
    """
    A class to manage and validate context data for analysis, visualisation, insights, recommendations, and tasks.

    Attributes:
        analysis (Union[str, None]): The analysis context, with alias choices for validation.
        visualisation (Union[str, None]): The visualisation context, with alias choices for validation.
        insights (Union[str, None]): The insights context, with alias choices for validation.
        recommendations (Union[str, None]): The recommendations context, with alias choices for validation.
        tasks (Union[str, None]): The tasks context, with alias choices for validation.
        model_config (ConfigDict): Configuration for model validation, ensuring assignment validation.

    Methods:
        validate_context(cls, data):
            Class method to validate and format context data before model instantiation.
        validate_from_string(self, context_str: str):
            Validates and formats context data from a string.
        validate_from_dict(self, context_dict: dict):
            Validates and formats context data from a dictionary.
        validate(self, context=None, **kwargs):
            Validates and formats context data from either a string, dictionary, or keyword arguments.
        reset_values(self):
            Resets all context fields to empty strings.
    """

    analysis: Union[str, None] = Field(
        default_factory=str, validation_alias=AliasChoices("analysis", "analyses")
    )
    visualisation: Union[str, None] = Field(
        default_factory=str,
        validation_alias=AliasChoices(
            "visualisation",
            "visualisations",
            "visualization",
            "visualizations",
            "image",
        ),
    )
    insights: Union[str, None] = Field(
        default_factory=str, validation_alias=AliasChoices("insights", "insight")
    )
    recommendations: Union[str, None] = Field(
        default_factory=str,
        validation_alias=AliasChoices("recommendations", "recommendation"),
    )
    tasks: Union[str, None] = Field(
        default_factory=str, validation_alias=AliasChoices("tasks", "task")
    )
    model_config = ConfigDict(validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def validate_context(cls, data):
        if isinstance(data, str):
            data = data.strip() + "\n\n" if data.strip() != "" else ""
            return {fname: data for fname in cls.model_fields}
        if not isinstance(data, dict):
            return data
        for key, value in data.items():
            if isinstance(value, str) and value.strip() != "":
                data[key] = value.strip() + "\n\n"
        return data

    def validate_from_string(self, context_str: str):
        if not isinstance(context_str, str):
            return self
        context_str = context_str.strip()
        if context_str != "":
            context_str = context_str + "\n\n"
        for key, value in self:
            if value is None or not isinstance(value, str) or value.strip() == "":
                setattr(self, key, context_str)
            else:
                setattr(self, key, value.strip() + "\n\n")
        return self

    def validate_from_dict(self, context_dict: dict):
        if isinstance(context_dict, dict):
            context_dict = ContextDict.model_validate_json(json.dumps(context_dict))
        if not isinstance(context_dict, ContextDict):
            return self
        for key, value in context_dict:
            if isinstance(value, str) and value.strip() != "":
                setattr(self, key, value.strip() + "\n\n")
        return self

    def validate(self, context=None, **kwargs):
        if isinstance(context, str):
            self = self.validate_from_string(context)
        if isinstance(context, (dict, ContextDict)):
            self = self.validate_from_dict(context)
        return self.validate_from_dict(kwargs)

    def reset_values(self):
        for key, _ in self:
            setattr(self, key, "")
        return self


class VectorStoreConfig(BaseModel):
    """Configuration model for vector store initialisation settings."""

    path: Union[str, None] = Field(
        default=None,
        validation_alias=AliasChoices("vector_store_path", "vstore_path", "path"),
    )
    remake_store: Union[bool, None] = Field(
        default=False, validation_alias=AliasChoices("remake", "remake_store")
    )


class FactoryBaseClass:
    """
    A base class for creating factory objects that interact with a language model, logging system, and vector store.

    Attributes:
        llm (LiteLLM): The language model instance.
        context (str): The context or environment in which the factory operates.
        logger (logging.Logger): The logger instance for logging messages.
        vector_store (ChromaDBVectorStore): The vector store for managing and retrieving vectors.
        output (Any): The output generated by the factory.
        code (str): The code to be executed or processed.
        params (ParamsDict): The parameters dictionary containing configuration settings.

    Methods:
        __init__(llm: LiteLLM, logger: logging.Logger, context: str, vector_store: ChromaDBVectorStore,
            max_retries: int, time_limit: int, auto_train: bool, llm_kwargs: dict) -> None:
            Initializes the factory with the given parameters and sets up the language model and vector store.

        generate_output(**kwargs):
            Abstract method to generate output. Must be implemented by subclasses.

        get_prompt_messages(**kwargs) -> list:
            Abstract method to get prompt messages. Must be implemented by subclasses.

        extract_and_execute_code(llm_response: str):
            Abstract method to extract and execute code from the language model response. Must be implemented by subclasses.

        auto_train(user_input: str, code: str, **kwargs) -> None:
            Abstract method to perform auto-training. Must be implemented by subclasses.
    """

    from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore

    llm: LiteLLM
    context: str
    logger: logging.Logger
    vector_store: ChromaDBVectorStore
    output: Any
    code: str
    params: ParamsDict

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

    def generate_output(self, **kwargs):
        raise NotImplementedError

    def get_prompt_messages(self, **kwargs) -> list:
        raise NotImplementedError

    def extract_and_execute_code(self, llm_response: str):
        raise NotImplementedError

    def auto_train(self, user_input: str, code: str, **kwargs) -> None:
        raise NotImplementedError
