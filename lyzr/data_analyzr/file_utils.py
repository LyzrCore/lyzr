# standard library imports
import os
import logging
from typing import Union

# third-party imports
import pandas as pd
from pydantic import BaseModel

# local imports
from lyzr.base.file_utils import read_file
from lyzr.data_analyzr.db_connector import (
    DatabaseConnector,
    SQLiteConnector,
)
from lyzr.data_analyzr.utils import deterministic_uuid
from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore
from lyzr.data_analyzr.models import (
    SupportedDBs,
    VectorStoreConfig,
    AnalysisTypes,
)


def get_db_details(
    db_scope: AnalysisTypes,
    db_type: SupportedDBs,
    db_config: BaseModel,
    vector_store_config: VectorStoreConfig,
    logger: logging.Logger,
):
    df_dict = None
    connector = None
    vector_store = None
    if db_type is SupportedDBs.files:
        logger.info(
            "Reading the following datasets:\n"
            + "\n".join(
                [f"{df} from {db_config.datasets[df]}" for df in db_config.datasets]
            )
            + "\n"
        )
        df_dict = get_dict_of_files(db_config.datasets, db_config.files_kwargs)
        logger.info(
            "Datasets read successfully:\n"
            + "\n".join([f"{df} with shape {df_dict[df].shape}" for df in df_dict])
            + "\n"
        )
    else:
        connector = DatabaseConnector.get_connector(db_type.value)(
            **db_config.model_dump()
        )

    if db_scope is AnalysisTypes.ml and df_dict is None:
        df_dict = connector.fetch_dataframes_dict()
    if df_dict is not None:
        df_keys = list(df_dict.keys())
        for key in df_keys:
            k_new = key.lower().replace(" ", "_")
            df_dict[k_new] = df_dict.pop(key)
    if db_scope is AnalysisTypes.sql and connector is None:
        connector = SQLiteConnector()
        connector.create_database(
            db_path=(
                db_config.db_path
                if db_config.db_path is not None
                else f"./sqlite/{deterministic_uuid()}.db"
            ),
            df_dict=df_dict,
        )
    if vector_store_config.path is None:
        uuid = deterministic_uuid(
            [
                db_scope.value,
                db_type.value,
                " ".join([f"{k}: {v}" for k, v in db_config.model_dump().items()]),
            ]
        )
        vector_store_config.path = f"./vector_store/{uuid}"
    vector_store = ChromaDBVectorStore(
        path=vector_store_config.path,
        remake_store=vector_store_config.remake_store,
        connector=connector,
        logger=logger,
    )
    return connector, df_dict, vector_store


def get_dict_of_files(datasets: dict, kwargs) -> dict[pd.DataFrame]:
    # kwargs = {"encoding": "utf-8", "sep": ["\t", ","]}
    kwargs_list = get_list_of_kwargs(datasets, kwargs)
    datasets_dict = {}
    for idx, (name, data) in enumerate(datasets.items()):
        datas = read_file_or_folder(name, data, kwargs_list[idx])
        # print(type(datas), len(datas), datas.keys())
        for d in datas:
            datasets_dict.update({d: datas[d]})
    return datasets_dict


def read_file_or_folder(
    name: str, filepath: Union[str, pd.DataFrame], kwargs
) -> dict[pd.DataFrame]:
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
