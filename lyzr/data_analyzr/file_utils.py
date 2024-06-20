"""
Utility functions for handling data, including reading files and directories.
"""

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
    DataFile,
    FilesConfig,
    RedshiftConfig,
    PostgresConfig,
    SQLiteConfig,
    DataConfig,
)
from lyzr.data_analyzr.db_connector import (
    DatabaseConnector,
    SQLiteConnector,
    TrainingPlan,
    TrainingPlanItem,
)
from lyzr.data_analyzr.models import AnalysisTypes
from lyzr.data_analyzr.models import VectorStoreConfig
from lyzr.data_analyzr.vector_store_utils import ChromaDBVectorStore
from lyzr.data_analyzr.utils import deterministic_uuid, translate_string_name


def get_db_details(
    analysis_type: AnalysisTypes,
    db_type: SupportedDBs,
    db_config: BaseModel,
    vector_store_config: VectorStoreConfig,
    logger: logging.Logger,
) -> tuple[DatabaseConnector, dict[str, pd.DataFrame], ChromaDBVectorStore]:
    """
    Retrieve database details including dataframes, connector, and vector store based on the provided configurations.

    Args:
        analysis_type (AnalysisTypes): The type of analysis to be performed.
        db_type (SupportedDBs): The type of database being used.
        db_config (BaseModel): Configuration for the database, must be an instance of
            FilesConfig, RedshiftConfig, PostgresConfig, or SQLiteConfig.
        vector_store_config (VectorStoreConfig): Configuration for the vector store.
        logger (logging.Logger): Logger instance for logging information.

    Returns:
        tuple: A tuple containing:
            - connector: The database connector instance.
            - df_dict (dict): A dictionary of dataframes read from the datasets.
            - vector_store (ChromaDBVectorStore): The vector store instance.

    Raises:
        AssertionError: If db_config is not an instance of the expected configuration class based on db_type.
    """
    df_dict = None
    connector = None
    vector_store = None
    training_plan = None
    # Read given datasets
    if db_type is SupportedDBs.files:
        assert isinstance(
            db_config, FilesConfig
        ), f"Expected FilesConfig, got {type(db_config)}"
        df_dict = get_dict_of_files(db_config.datasets)
        logger.info(
            "Following datasets read successfully:\n"
            + "\n".join([f"{df} with shape {df_dict[df].shape}" for df in df_dict])
            + "\n"
        )
    else:
        accepted_db_types = tuple(
            [
                elem
                for elem in DataConfig._config_types.values()
                if elem is not FilesConfig
            ]
        )
        assert isinstance(
            db_config, accepted_db_types
        ), f"Expected one of {accepted_db_types}, got {type(db_config)}"
        connector = DatabaseConnector.get_connector(db_type)(**db_config.model_dump())
    df_dict, connector = ensure_correct_data_format(
        analysis_type=analysis_type,
        db_config=db_config,
        df_dict=df_dict,
        connector=connector,
    )
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


def ensure_correct_data_format(
    analysis_type: AnalysisTypes,
    db_config: Union[FilesConfig, RedshiftConfig, PostgresConfig, SQLiteConfig],
    df_dict: Optional[dict[str, pd.DataFrame]] = None,
    connector: Optional[DatabaseConnector] = None,
) -> tuple[dict[str, pd.DataFrame], DatabaseConnector]:
    """
    Ensure the correct format of data based on the analysis type and provided configurations.

    This function adjusts the format of the data (either as pandas DataFrames or a SQL connector)
    depending on the specified analysis type.
    - For pythonic analysis, it ensures data is in the form of pandas DataFrames.
    - For SQL analysis, it ensures data is accessible via a SQL connector.

    Args:
        analysis_type (AnalysisTypes): The type of analysis to be performed (pythonic, SQL, or neither).
        db_config (Union[FilesConfig, RedshiftConfig, PostgresConfig, SQLiteConfig]): Configuration for the database.
        df_dict (Optional[dict[str, pd.DataFrame]], optional): A dictionary of dataframes. Defaults to None.
        connector (Optional[DatabaseConnector], optional): A database connector instance. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - df_dict (Optional[dict[str, pd.DataFrame]]): The dictionary of dataframes, if applicable.
            - connector (Optional[DatabaseConnector]): The database connector instance, if applicable.
    """
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
    return df_dict, connector


def make_training_plan(
    analysis_type: AnalysisTypes,
    db_type: SupportedDBs,
    df_dict: dict,
    connector: DatabaseConnector,
) -> TrainingPlan:
    """
    Create a training plan based on the analysis type, database type, and provided data.

    This function generates a training plan for pythonic or SQL analysis based on the specified
    analysis type and database type. It processes dataframes or uses a database connector to
    create the appropriate training plan.

    Args:
        analysis_type (AnalysisTypes): The type of analysis to be performed (Pythonic, SQL, or skip).
        db_type (SupportedDBs): The type of database being used (files, Redshift, Postgres, SQLite).
        df_dict (dict): A dictionary of dataframes to be used for the training plan.
        connector (DatabaseConnector): The database connector instance for SQL databases.

    Returns:
        TrainingPlan: The generated training plan based on the provided data and configurations.

    Raises:
        AssertionError: If the dataframes in df_dict are not instances of pandas DataFrame.
        AssertionError: If the training plan could not be created from the given data.
    """
    training_plan = None
    if analysis_type is AnalysisTypes.ml or (
        (analysis_type is AnalysisTypes.skip) and (db_type is SupportedDBs.files)
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
    elif analysis_type is AnalysisTypes.sql or (
        (analysis_type is AnalysisTypes.skip) and (db_type is not SupportedDBs.files)
    ):
        training_plan = connector.get_default_training_plan()
    assert isinstance(
        training_plan, TrainingPlan
    ), "Unable to create training plan from given data."
    return training_plan


def get_dict_of_files(datasets: list[DataFile]) -> dict[str, pd.DataFrame]:
    """
    Retrieve a dictionary of dataframes from a list of data files.

    Args:
        datasets (list[DataFile]): A list of DataFile objects, each containing the name, value,
            and keyword arguments for reading the file.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where the keys are the formatted names of the data
            files and the values are the corresponding dataframes.

    Precedure:
    - Process a list of data files
    - Read each file, folder, or dataframe
    - Store the resulting dataframes in a dictionary
    - The keys of the dictionary are formatted to
        - replaces spaces and punctuation with underscores (_)
        - converts all characters to lowercase
    """
    datasets_dict = {}
    for data in datasets:
        datasets_dict.update(read_file_or_folder(data.name, data.value, data.kwargs))
    df_names = list(datasets_dict.keys())
    for name in df_names:
        new_df_name = translate_string_name(name)
        datasets_dict[new_df_name] = datasets_dict.pop(name)
        rename_cols = {
            col: translate_string_name(col)
            for col in datasets_dict[new_df_name].columns
        }
        datasets_dict[new_df_name].rename(columns=rename_cols, inplace=True)
    return datasets_dict


def read_file_or_folder(
    name: str, filepath: Union[str, pd.DataFrame], kwargs
) -> dict[str, pd.DataFrame]:
    """
    Reads a file or all files in a directory and returns their contents as a dictionary of DataFrames.

    This function handles both individual files and directories:
    - If filepath is a DataFrame, it returns a dictionary with the given name as the key.
    - If filepath is a file, it reads the file and returns a dictionary with the given name as the key.
    - If filepath is a directory, it reads all files in the directory (excluding nested directories) and
      returns a dictionary with keys derived from the filenames.

    Args:
        name (str): The base name to use for the dictionary keys.
        filepath (Union[str, pd.DataFrame]): The path to the file or directory, or a DataFrame.
        kwargs: Additional keyword arguments to pass to the `read_file` function.

    Returns:
        dict[str, pd.DataFrame]: A dictionary where keys are derived from the provided name and filenames, and values are DataFrames.

    Raises:
        ValueError: If filepath is neither a valid file nor a directory.
    """
    if isinstance(filepath, pd.DataFrame):
        return {name: filepath}
    if os.path.isfile(filepath):
        return {name: read_file(filepath, **kwargs)}
    elif os.path.isdir(filepath):
        return {
            f"{name}_{os.path.splitext(file)[0]}": read_file(
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


def read_file(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Read a file and return its contents as a pandas DataFrame.

    This function reads a file from the specified filepath and returns its contents as a pandas DataFrame.
    Supported file formats include CSV, TSV, JSON, Excel, TXT, and pickle files.

    Args:
        filepath (str): The path to the file to be read.
        encoding (Optional[str], optional): The encoding of the file. Defaults to "utf-8".
        **kwargs: Additional keyword arguments to pass to the pandas read function.

    Returns:
        pd.DataFrame: The contents of the file as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file is not found at the specified filepath.
        UnicodeDecodeError: If the file could not be decoded with the specified encoding.
        ValueError:
            - If the file does not exist
            - If the file extension is not supported
            - If the file could not be read as a DataFrame.
    """
    if not os.path.exists(filepath):
        raise ValueError(
            f"File '{filepath}' not found. Please provide a valid filepath."
        )
    file_extension = filepath.split(".")[-1]
    try:
        if file_extension == "csv" or file_extension == "txt":
            df = pd.read_csv(filepath, **kwargs)
        elif file_extension == "tsv":
            df = pd.read_csv(filepath, sep="\t", **kwargs)
        elif file_extension == "json":
            df = pd.read_json(filepath, **kwargs)
        elif file_extension in ["xlsx", "xls"]:
            df = pd.read_excel(filepath, **kwargs)
        elif file_extension == "pkl":
            with open(filepath, "rb") as f:
                df = pickle.load(f, **kwargs)
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
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"File '{filepath}' could not be read as a DataFrame. Please provide a valid file."
        )
    return df
