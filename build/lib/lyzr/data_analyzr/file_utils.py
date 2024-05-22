# standard library imports
import os
import io
import pickle
import logging
from typing import Union, Optional

# third-party imports
import pandas as pd
from pydantic import BaseModel

# local imports
from lyzr.data_analyzr.db_models import (
    SupportedDBs,
    FilesConfig,
    RedshiftConfig,
    PostgresConfig,
    SQLiteConfig,
    VectorStoreConfig,
)
from lyzr.data_analyzr.db_connector import (
    DatabaseConnector,
    SQLiteConnector,
    TrainingPlan,
    TrainingPlanItem,
)
from lyzr.data_analyzr.models import AnalysisTypes
from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore
from lyzr.data_analyzr.utils import deterministic_uuid, translate_string_name


def get_db_details(
    analysis_type: AnalysisTypes,
    db_type: SupportedDBs,
    db_config: BaseModel,
    vector_store_config: VectorStoreConfig,
    logger: logging.Logger,
):
    df_dict = None
    connector = None
    vector_store = None
    training_plan = None
    # Read given datasets
    if db_type is SupportedDBs.files:
        assert isinstance(
            db_config, FilesConfig
        ), f"Expected FilesConfig, got {type(db_config)}"
        df_dict = get_dict_of_files(db_config.datasets, db_config.files_kwargs)
        logger.info(
            "Following datasets read successfully:\n"
            + "\n".join([f"{df} with shape {df_dict[df].shape}" for df in df_dict])
            + "\n"
        )
    else:
        assert isinstance(
            db_config, (RedshiftConfig, PostgresConfig, SQLiteConfig)
        ), f"Expected RedshiftConfig, PostgresConfig or SQLiteConfig, got {type(db_config)}"
        connector = DatabaseConnector.get_connector(db_type)(**db_config.model_dump())
    # Ensure correct format of data (pandas DataFrame or sql connector) depending on analysis_type
    if analysis_type is AnalysisTypes.ml and df_dict is None:
        df_dict = connector.fetch_dataframes_dict()
        connector = None
    if df_dict is not None:
        df_dict = {translate_string_name(k): v for k, v in df_dict.items()}
    if analysis_type is AnalysisTypes.sql and connector is None:
        connector = SQLiteConnector()
        connector.create_database(
            db_path=(
                db_config.db_path
                if db_config.db_path is not None
                else f"./sqlite/{deterministic_uuid()}.db"
            ),
            df_dict=df_dict,
        )
        df_dict = None
    # Create training plan and vector store
    training_plan = make_training_plan(analysis_type, db_type, df_dict, connector)
    if vector_store_config.path is None:
        uuid = deterministic_uuid(
            [
                analysis_type.value,
                db_type.value,
                " ".join([f"{k}: {v}" for k, v in db_config.model_dump().items()]),
            ]
        )
        vector_store_config.path = f"./vector_store/{uuid}"
    vector_store = ChromaDBVectorStore(
        path=vector_store_config.path,
        remake_store=vector_store_config.remake_store,
        training_plan=training_plan,
        logger=logger,
    )
    return connector, df_dict, vector_store


def make_training_plan(
    db_scope: AnalysisTypes,
    db_type: SupportedDBs,
    df_dict: dict,
    connector: DatabaseConnector,
) -> TrainingPlan:
    training_plan = None
    if db_scope is AnalysisTypes.ml or (
        (db_scope is AnalysisTypes.skip) and (db_type is SupportedDBs.files)
    ):
        training_plan = TrainingPlan([])
        for name, df in df_dict.items():
            assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"
            doc = f"The following columns are in the {name} dataframe:\n\n"
            buffer = io.StringIO()
            df.info(buf=buffer)
            doc += buffer.getvalue()
            training_plan._plan.append(
                TrainingPlanItem(
                    item_type=TrainingPlanItem.ITEM_TYPE_IS,
                    item_group=name,
                    item_name=name,
                    item_value=doc,
                )
            )
            doc = f"Following are the first five rows of the {name} dataframe:\n\n"
            doc += df.head().to_markdown()
            training_plan._plan.append(
                TrainingPlanItem(
                    item_type=TrainingPlanItem.ITEM_TYPE_IS,
                    item_group=name,
                    item_name=name,
                    item_value=doc,
                )
            )
    elif db_scope is AnalysisTypes.sql or (
        (db_scope is AnalysisTypes.skip) and (db_type is not SupportedDBs.files)
    ):
        training_plan = connector.get_default_training_plan()
    assert isinstance(
        training_plan, TrainingPlan
    ), "Unable to create training plan from given data."
    return training_plan


def get_dict_of_files(datasets: dict, kwargs) -> dict[str, pd.DataFrame]:
    kwargs_list = get_list_of_kwargs(datasets, kwargs)
    datasets_dict = {}
    for idx, (name, data) in enumerate(datasets.items()):
        read_datasets = read_file_or_folder(name, data, kwargs_list[idx])
        for df_name in read_datasets:
            datasets_dict.update(
                {translate_string_name(df_name): read_datasets[df_name]}
            )
    return datasets_dict


def read_file_or_folder(
    name: str, filepath: Union[str, pd.DataFrame], kwargs
) -> dict[str, pd.DataFrame]:
    if isinstance(filepath, pd.DataFrame):
        return {name: filepath}
    if os.path.isfile(filepath):
        return {name: read_file(filepath, **kwargs)}
    elif os.path.isdir(filepath):
        return {
            f"{name} {os.path.splitext(file)[0]}": read_file(
                os.path.join(filepath, file), **kwargs
            )
            for file in os.listdir(filepath)
            if os.path.isfile(
                os.path.join(filepath, file)
            )  # do not read from nested directories
        }

    else:
        raise ValueError(
            f"Filepath '{filepath}' is not a valid file or directory. Please provide a valid filepath."
        )


def get_list_of_kwargs(datasets: dict, kwargs: Union[dict, list]) -> list[dict]:
    if isinstance(kwargs, list) and len(kwargs) == len(datasets):
        return kwargs
    kwargs_list = [{} for _ in range(len(datasets))]
    for key, value in kwargs.items():
        if isinstance(value, list) and len(value) == len(
            datasets
        ):  # {"key": [value1, value2, value3]} -> [{"key": value1}, {"key": value2}, {"key": value3}]
            for i, v in enumerate(value):
                kwargs_list[i][key] = v
        else:  # {"key": value} -> [{"key": value}, {"key": value}, {"key": value}]
            for i in range(len(datasets)):
                kwargs_list[i][key] = value
    return kwargs_list


def read_file(
    filepath: str, encoding: Optional[str] = "utf-8", **kwargs
) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise ValueError(
            f"File '{filepath}' not found. Please provide a valid filepath."
        )
    file_extension = filepath.split(".")[-1]
    try:
        if file_extension == "csv":
            return pd.read_csv(filepath, encoding=encoding, **kwargs)
        elif file_extension == "tsv":
            return pd.read_csv(filepath, sep="\t", encoding=encoding, **kwargs)
        elif file_extension == "txt":
            with open(filepath, "r") as f:
                return f.read(encoding=encoding, **kwargs)
        elif file_extension == "json":
            return pd.read_json(filepath, encoding=encoding, **kwargs)
        elif file_extension in ["xlsx", "xls"]:
            return pd.read_excel(filepath, encoding=encoding, **kwargs)
        elif file_extension == "pkl":
            with open(filepath, "rb") as f:
                return pickle.load(f, encoding=encoding, **kwargs)
        else:
            raise ValueError(
                f"File extension '{file_extension}' not supported. Please provide a csv or pkl file."
            )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File '{filepath}' not found. Please provide a valid filepath."
        )
    except UnicodeDecodeError:
        raise UnicodeDecodeError(
            f"File '{filepath}' could not be decoded. Please provide a file with utf-8 encoding."
            "If the file is not encoded in utf-8, please provide the encoding as a parameter: file_kwargs={'encoding': 'utf-8'}"
        )
