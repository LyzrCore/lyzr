"""
With DataAnalyzr, you can analyze any dataframe 
derive actionable insights and create 
visually compelling and Intuitive visualizations.
"""
import os
import io
import re
import sys
import shutil
import warnings
import datetime
from pathlib import Path
from types import TracebackType
from contextlib import AbstractContextManager
from typing import Optional, Union, Type

import numpy as np
from PIL import Image
import pandas as pd
from pandas.errors import EmptyDataError

from lyzr.base.prompt import Prompt
from lyzr.base.llms import LLM, get_model
from lyzr.base.errors import MissingValueError
from lyzr.base.file_utils import read_file

PATTERN_PYTHON_CODE_BLOCK = r"```python\n(.*?)\n```"

warnings.filterwarnings("ignore")


class CapturePrints(AbstractContextManager):
    """
    A context manager for capturing output to stdout.

    This class can be used to capture prints and other output
    that normally go to stdout. It can be useful whenever
    output to stdout needs to be captured or suppressed.
    """

    def __enter__(self) -> "CapturePrints":
        self._old_stdout = sys.stdout
        sys.stdout = self._mystdout = io.StringIO()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[Exception],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        sys.stdout = self._old_stdout
        # Can decide to handle exceptions here or let them propagate
        return None  # Returning None means any exceptions will propagate

    def get_value(self) -> str:
        """Retrieve the captured stdout content."""
        return self._mystdout.getvalue()


class DataAnalyzr:
    """The DataAnalyzr Class for analyzing dataframes."""

    def __init__(
        self,
        df: Union[str, pd.DataFrame] = None,
        api_key: Optional[str] = None,
        model: Optional[LLM] = None,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        user_input: Optional[str] = None,
        seed: int = None,
    ):
        self.model = (
            model
            or os.environ.get("LLM_MODEL")
            or get_model(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                model_type=model_type or os.environ.get("MODEL_TYPE", "openai"),
                model_name=model_name or os.environ.get("MODEL_NAME", "gpt-3.5-turbo"),
            )
        )
        # Process the dataframe parameter
        if df is None:
            df = os.environ.get(
                "DATAFRAME"
            )  # Try getting from environment if not provided
        if df is None:
            # If df is still None, neither a DataFrame nor a valid path were provided
            raise MissingValueError(["dataframe"])
        elif isinstance(df, str):
            self.df = self._clean_df(read_file(df))
        elif isinstance(df, pd.DataFrame):
            # You might want to check whether df is empty
            if df.empty:
                raise EmptyDataError("The provided DataFrame is empty.")
            self.df = self._clean_df(df)
        else:
            raise ValueError("df must be a path to a file or a pd.DataFrame object.")

        if isinstance(df, str):
            self.df = self._clean_df(read_file(df))

        self.user_input = user_input
        self.seed = seed

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataframe.

        Parameters:
        - df (pd.DataFrame) : Input Dataframe to be cleaned.

        Returns:
        - pd.DataFrame: cleaned dataframe with no columns having >50% missing values,
        no unnamed columns, no duplicates, and NaNs replaced in numerical/categorical columns.
        """
        # Removing columns having more than 50% of missing values
        df = df[df.columns[df.isnull().mean() < 0.5]]

        # Getting the columns which are categorical
        cat_columns = df.select_dtypes(include=["object"]).columns

        # Getting the columns which are numerical
        num_columns = df.select_dtypes(include=[np.number]).columns

        # Replacing missing categorical values with the most frequent value(mode)
        df[cat_columns] = df[cat_columns].apply(lambda x: x.fillna(x.mode()[0]))

        # Replacing missing numerical values with the mean
        df[num_columns] = df[num_columns].apply(lambda x: x.fillna(x.mean()))

        # Removing "Unnamed:" columns if any
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # Removing duplicates
        df = df.drop_duplicates(keep="first")

        return df

    def _get_analysis_steps(self, user_input: str = None) -> str:
        """
        Get the steps to perform the analysis.

        Parameters:
        - user_input (str):
            The user input based on which analysis steps are generated. (default: None)

        Raises:
        - MissingValueError: If user input is missing.

        Returns:
        - str: The analysis steps based on user input and dataframe.
        """
        self.user_input = user_input or self.user_input
        if self.user_input is None:
            raise MissingValueError(["user_input"])

        self.model.set_messages(
            [
                {
                    "prompt": Prompt("system_analysis_steps_pt"),
                    "role": "system",
                },
                {
                    "prompt": Prompt("analysis_steps_pt").format(
                        user_input=self.user_input,
                        df_head=self.df.head(5),
                        df_columns=self.df.columns.tolist(),
                    ),
                    "role": "user",
                },
            ]
        )

        steps = self.model.run(temperature=0.1).choices[0].message.content

        return steps

    def _get_analysis_code(self, instructions: str, user_input: str = None) -> str:
        """
        Get the Python code to perform the analysis.

        Parameters:
        - instructions (str): The instructions to perform the analysis.
        - user_input (str):
            The user input based on which analysis code is generated. (default: None)

        Raises:
        - MissingValueError: If user input is missing.

        Returns:
        - str: The analysis Python code based on user input and dataframe.
        """
        self.user_input = user_input or self.user_input
        if self.user_input is None:
            raise MissingValueError(["user_input"])

        self.model.set_messages(
            [
                {
                    "prompt": Prompt("system_code_pt"),
                    "role": "system",
                },
                {
                    "prompt": Prompt("analysis_code_pt").format(
                        user_input=self.user_input,
                        instructions=instructions,
                        df_head=self.df.head(5),
                        df_columns=self.df.columns.tolist(),
                    ),
                    "role": "user",
                },
            ]
        )
        model_response = self.model.run(temperature=0.1).choices[0].message.content

        python_code_blocks = re.findall(
            PATTERN_PYTHON_CODE_BLOCK, model_response, re.DOTALL
        )

        try:
            python_code = python_code_blocks[0]
        except IndexError:
            python_code = model_response

        return python_code

    def _get_visualization_steps(self, user_input: str = None) -> str:
        """
        Get the steps to perform the visualization.

        Parameters:
        - user_input (str):
            The user input based on which visualization steps are generated. (default: None)

        Raises:
        - MissingValueError: If user input is missing.

        Returns:
        - str: The visualization steps based on user input and dataframe.
        """
        self.user_input = user_input or self.user_input
        if self.user_input is None:
            raise MissingValueError(["user_input"])

        self.model.set_messages(
            [
                {
                    "prompt": Prompt("system_visualization_pt"),
                    "role": "system",
                },
                {
                    "prompt": Prompt("visualization_steps_pt").format(
                        user_input=self.user_input,
                        df_head=self.df.head(5),
                        df_columns=self.df.columns.tolist(),
                    ),
                    "role": "user",
                },
            ]
        )

        steps = self.model.run(temperature=0.1).choices[0].message.content

        return steps

    def _get_visualiztion_code(self, instructions: str, user_input: str = None) -> str:
        """
        Get the Python code to generate the visualization.

        Parameters:
        - instructions (str): The instructions to generate the visualization.
        - user_input (str): The user input based on which visualization code is generated. (default: None)

        Raises:
        - MissingValueError: If user input is missing.

        Returns:
        - str: The visualization Python code based on user input and dataframe.
        """
        self.user_input = user_input or self.user_input
        if self.user_input is None:
            raise MissingValueError(["user_input"])

        self.model.set_messages(
            [
                {
                    "prompt": Prompt("system_vis_code_pt"),
                    "role": "system",
                },
                {
                    "prompt": Prompt("visualization_code_pt").format(
                        user_input=self.user_input,
                        df_head=self.df.head(5),
                        df_columns=self.df.columns.tolist(),
                        instructions=instructions,
                    ),
                    "role": "user",
                },
            ]
        )

        model_response = self.model.run(temperature=0.1).choices[0].message.content

        python_code_blocks = re.findall(
            PATTERN_PYTHON_CODE_BLOCK, model_response, re.DOTALL
        )

        try:
            python_code = python_code_blocks[0]
        except IndexError:
            python_code = model_response

        return python_code

    def _correct_code(self, python_code: str, error_message: str) -> str:
        """
        Correct the Python code based on the error message.

        Parameters:
        - python_code (str): The Python code to be corrected.
        - error_message (str): The error message based on which the code is corrected.

        Returns:
        - str: The corrected Python code.
        """
        corrected_python_code = ""

        self.model.set_messages(
            [
                {
                    "prompt": Prompt("system_correct_code_pt"),
                    "role": "system",
                },
                {
                    "prompt": Prompt("correct_code_pt").format(
                        python_code=python_code,
                        error_message=error_message,
                        df_columns=self.df.columns.tolist(),
                        df_head=self.df.head(5),
                        user_input=self.user_input,
                    ),
                    "role": "user",
                },
            ]
        )
        model_response = self.model.run(temperature=0.1).choices[0].message.content

        python_code_blocks = re.findall(
            PATTERN_PYTHON_CODE_BLOCK, model_response, re.DOTALL
        )

        try:
            corrected_python_code = python_code_blocks[0]
        except IndexError:
            corrected_python_code = model_response

        return corrected_python_code

    def _get_analysis_output(self, user_input: str = None) -> str:
        """
        Get the output of the analysis.

        Parameters:
        - user_input (str):
            The user input based on which analysis output is generated. (default: None)

        Raises:
        - MissingValueError: If user input is missing.

        Returns:
        - str: The analysis output after running the analysis python code.
        """
        self.user_input = user_input or self.user_input
        if self.user_input is None:
            raise MissingValueError(["user_input"])
        steps = self._get_analysis_steps()

        analysis_python_code = self._get_analysis_code(steps)

        with CapturePrints() as c:
            exec_scope = {"df": self.df}
            try:
                exec(analysis_python_code, exec_scope)
            except Exception as e:
                error_message = str(e)
                print(f"Error. {type(e).__name__}: {error_message}")
                analysis_python_code = self._correct_code(
                    analysis_python_code, error_message
                )
                exec(analysis_python_code, exec_scope)

        output = c.get_value()
        return output

    def _load_images_in_current_directory(self) -> dict[str, bytes]:
        """
        Load all PNG images in the current directory and convert them to a dictionary

        Returns:
        - Dict[str, bytes]:
            A dictionary where the key is the filename and the value is byte data of the image.
        """
        current_directory = os.getcwd()
        image_data = {}

        for filename in os.listdir(current_directory):
            if filename.lower().endswith(".png"):
                image_path = os.path.join(current_directory, filename)
                with Image.open(image_path) as img:
                    byte_arr = io.BytesIO()
                    img.save(byte_arr, format="PNG")
                    image_data[filename] = byte_arr.getvalue()

        return image_data

    def _move_visualization_files(
        self, source: Path, destination: Path, file_type: str
    ) -> None:
        """
        Move generated visualization files of a given type,
        from a source directory to a destination directory.

        Parameters:
        - source (Path): The source directory.
        - destination (Path): The destination directory.
        - file_type (str): The type of files to be moved.

        Post-condition:
        - The files of the specified type are moved from source to destination.
        - The files are renamed to include the current datetime in their names for uniqueness.
        """
        if not destination.exists():
            destination.mkdir(parents=True)
        for file_name in source.glob(f"*.{file_type}"):
            base = file_name.stem
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            new_file_name = f"{base}-{now}.{file_type}"
            new_file_path = destination / new_file_name
            shutil.move(str(file_name), str(new_file_path))

    def analysis_insights(self, user_input: str = None) -> str:
        """
        Get insights from the analysis in three bullet points.

        Parameters:
        - user_input (str): The user input based on which insights are generated. (default: None)

        Raises:
        - MissingValueError: If user input is missing.

        Returns:
        - str: The analysis insights derived from the dataframe based on user input.
        """
        self.user_input = user_input or self.user_input
        if self.user_input is None:
            raise MissingValueError(["user_input"])

        analysis_output = self._get_analysis_output()

        if len(analysis_output) > 6000:
            analysis_output = analysis_output[:3000] + "..." + analysis_output[-3000:]

        prompt_messages = [
            {
                "prompt": Prompt("system_analysis_pt"),
                "role": "system",
            },
            {
                "prompt": Prompt("analysis_output_pt").format(
                    user_input=self.user_input, analysis_output=analysis_output
                ),
                "role": "user",
            },
        ]

        self.model.set_messages(prompt_messages)

        analysis = (
            self.model.run(
                temperature=0.3, top_p=1, frequency_penalty=0, presence_penalty=0
            )
            .choices[0]
            .message.content
        )

        return analysis

    def visualizations(
        self,
        user_input: str = None,
        dir_path: Path = Path("./generated_plots"),
    ) -> list[Image.Image]:
        """
        Get visualizations of the analysis and
        save the generated plot images to the specified directory.

        Parameters:
        - user_input (str):
            The user input data based on which visualizations are created. (default: None)
        - dir_path (Path):
            The directory path to save the visualizations (default: "./generated_plots")

        Raises:
        - MissingValueError: If user input is not provided.

        Returns:
        - List[Image.Image]: A list of PIL Image objects representing the saved visualizations.
        """
        self.user_input = user_input or self.user_input
        if self.user_input is None:
            raise MissingValueError(["user_input"])
        visualization_steps = self._get_visualization_steps()
        visualization_python_code = self._get_visualiztion_code(visualization_steps)

        exec_scope = {"df": self.df}
        try:
            exec(visualization_python_code, exec_scope)
        except Exception as e:
            error_message = str(e)
            visualization_python_code = self._correct_code(
                visualization_python_code, error_message
            )
            exec(visualization_python_code, exec_scope)

        image_list = self._load_images_in_current_directory()
        self._move_visualization_files(Path("./"), dir_path, "png")
        return image_list

    def dataset_description(self) -> str:
        """
        Generate a brief description of the dataset currently in use.

        Returns:
        - str: A string providing a description of the dataset.
        """
        self.model.set_messages(
            [
                {
                    "prompt": Prompt("system_describe_dataset_pt"),
                    "role": "system",
                },
                {
                    "prompt": Prompt("describe_dataset_pt").format(
                        df_head=self.df.head(5),
                    ),
                    "role": "user",
                },
            ]
        )

        description = (
            self.model.run(
                temperature=1, top_p=0.3, frequency_penalty=0.7, presence_penalty=0.3
            )
            .choices[0]
            .message.content
        )

        return description

    def ai_queries_df(self, dataset_description: Optional[str] = None) -> str:
        """
        Returns AI-generated queries for data analysis related to the dataset.

        Parameters:
        - dataset_description (str, optional):
            A description of the dataset. If not provided, it will be generated.

        Returns:
        - str: Queries for data analysis related to the dataset.
        """
        if dataset_description is None:
            dataset_description = self.dataset_description()

        self.model.set_messages(
            [
                {
                    "prompt": Prompt("system_ai_queries_pt"),
                    "role": "system",
                },
                {
                    "prompt": Prompt("ai_queries_pt").format(
                        dataset_description=dataset_description,
                        df_head=self.df.head(5),
                    ),
                    "role": "user",
                },
            ]
        )

        ai_queries_df = (
            self.model.run(
                temperature=1, top_p=0.3, frequency_penalty=0.7, presence_penalty=0.3
            )
            .choices[0]
            .message.content
        )

        return ai_queries_df

    def analysis_recommendation(
        self, user_input: Optional[str] = None, number_of_recommendations: int = 4
    ) -> str:
        """
        Get recommendation on what analysis to perform on the dataset.

        Parameters:
        - user_input (str, optional): The user input data. Defaults to None.
        - number_of_recommendations (int, optional):
            The number of recommendations to return. Defaults to 4.

        Returns:
        - str: Recommendations for analysis.
        """
        formatted_user_input: str = (
            Prompt("formatted_user_input_pt").format(user_input=user_input)
            if user_input is not None
            else ""
        )

        self.model.set_messages(
            [
                {
                    "prompt": Prompt("system_recommendations_pt").format(
                        number_of_recommendations=number_of_recommendations,
                    ),
                    "role": "system",
                },
                {
                    "prompt": Prompt("analysis_recommendations_pt").format(
                        number_of_recommendations=number_of_recommendations,
                        df_head=self.df.head(5),
                        df_columns=self.df.columns.tolist(),
                        formatted_user_input=formatted_user_input,
                    ),
                    "role": "system",
                },
            ]
        )

        recommendations = (
            self.model.run(temperature=0.2, seed=self.seed).choices[0].message.content
        )
        return recommendations

    def recommendations(
        self,
        insights: Optional[str] = None,
        user_input: Optional[str] = None,
        schema: Optional[list] = None,
    ) -> str:
        """
        Get recommendations based on the analysis insights.

        Parameters:
        - insights (str, optional): The analysis insights. Defaults to None.
        - user_input (str, optional): The user input. Defaults to None.
        - schema (list, optional):
            The schema for the recommendations. Defaults to a predefined schema.

        Raises:
        - MissingValueError: If user_input is not provided.

        Returns:
        - str: Recommendations as a string.
        """
        self.user_input = user_input or self.user_input

        if self.user_input is None:
            raise MissingValueError(["user_input"])

        if insights is None:
            insights = self.analysis_insights(user_input=user_input)

        schema = schema or [
            {
                "Recommendation": "string",
                "Basis of the Recommendation": "string",
                "Impact if implemented": "string",
            },
            {
                "Recommendation": "string",
                "Basis of the Recommendation": "string",
                "Impact if implemented": "string",
            },
            {
                "Recommendation": "string",
                "Basis of the Recommendation": "string",
                "Impact if implemented": "string",
            },
        ]

        self.model.set_messages(
            [
                {
                    "prompt": Prompt("system_recommendation_pt").format(schema=schema),
                    "role": "system",
                },
                {
                    "prompt": Prompt("recommendation_pt").format(
                        user_input=user_input, insights=insights
                    ),
                    "role": "user",
                },
            ]
        )

        recommendations = (
            self.model.run(
                temperature=1, top_p=0.3, frequency_penalty=0.7, presence_penalty=0.3
            )
            .choices[0]
            .message.content
        )

        return recommendations

    def tasks(
        self,
        user_input: Optional[str] = None,
        insights: Optional[str] = None,
        recommendations: Optional[str] = None,
    ) -> str:
        """
        Generate tasks based on the given user input, analysis insights, and recommendations.

        Parameters:
        - user_input (Optional[str]):
            The user input data. If None, the class's user_input attribute is used.
        - insights (Optional[str]):
            The analysis insights. If None, generated by analysis_insights method.
        - recommendations (Optional[str]):
            The analysis recommendations. If None, generated by recommendations method.

        Raises:
        - MissingValueError: If user_input is not provided.

        Returns:
        - str: The generated tasks.
        """
        self.user_input = user_input or self.user_input
        if self.user_input is None:
            raise MissingValueError(["user_input"])

        if insights is None:
            insights = self.analysis_insights(user_input=user_input)

        if recommendations is None:
            recommendations = self.recommendations(
                insights=insights, user_input=user_input
            )

        self.model.set_messages(
            [
                {
                    "prompt": Prompt("system_tasks_pt"),
                    "role": "system",
                },
                {
                    "prompt": Prompt("task_pt").format(
                        user_input=user_input,
                        insights=insights,
                        recommendations=recommendations,
                    ),
                    "role": "user",
                },
            ]
        )

        tasks = (
            self.model.run(
                temperature=1, top_p=0.3, frequency_penalty=0.7, presence_penalty=0.3
            )
            .choices[0]
            .message.content
        )

        return tasks
