from openai import OpenAI
from PIL import Image
import pandas as pd
import shutil
import glob
import ast
import sys
import re
import io
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import plotly

import warnings

from typing import Optional, Union
from lyzr.base.prompt import Prompt
from lyzr.base.llms import LLM, get_model
from lyzr.base.errors import MissingValueError
from lyzr.base.file_utils import read_file
from datetime import datetime

warnings.filterwarnings("ignore")


class CapturePrints:
    def __enter__(self):
        self.old_stdout = sys.stdout
        sys.stdout = self.mystdout = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout

    def getvalue(self):
        return self.mystdout.getvalue()


class DataAnalyzr:
    def __init__(
        self,
        df: Union[str, pd.DataFrame] = None,
        api_key: Optional[str] = None,
        model: Optional[LLM] = None,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        user_input: Optional[str] = None,
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
        df = df or os.environ.get("DATAFRAME")
        if df is None:
            raise MissingValueError(["dataframe"])
        if isinstance(df, str):
            self.df = self.cleandf(read_file(df))

        self.user_input = user_input

    def cleandf(self, df):
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

    def getrecommendations(self, number_of_recommendations=4):
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
                    ),
                    "role": "system",
                },
            ]
        )

        steps = self.model.run(temperature=0.2).choices[0].message.content
        return steps

    def getanalysissteps(self, user_input=None):
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

        try:
            result_list = ast.literal_eval(steps)
            if isinstance(result_list, list) and all(
                isinstance(item, str) for item in result_list
            ):
                steps = result_list
        except Exception:
            pass

        return steps

    def getanalysiscode(self, instructions, user_input=None):
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

        pattern = r"```python\n(.*?)\n```"
        python_code_blocks = re.findall(pattern, model_response, re.DOTALL)

        try:
            python_code = python_code_blocks[0]
        except Exception:
            python_code = model_response

        return python_code

    def getvisualizationsteps(self, user_input=None):
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

        try:
            result_list = ast.literal_eval(steps)
            if isinstance(result_list, list) and all(
                isinstance(item, str) for item in result_list
            ):
                steps = result_list
        except Exception:
            pass

        return steps

    def getvisualiztioncode(self, instructions, user_input=None):
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
                        instructions=instructions,
                    ),
                    "role": "user",
                },
            ]
        )

        model_response = self.model.run(temperature=0.1).choices[0].message.content

        pattern = r"```python\n(.*?)\n```"
        python_code_blocks = re.findall(pattern, model_response, re.DOTALL)

        try:
            python_code = python_code_blocks[0]
        except Exception:
            python_code = model_response

        return python_code

    def correctcode(self, python_code, error_message):
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

        pattern = r"```python\n(.*?)\n```"
        python_code_blocks = re.findall(pattern, model_response, re.DOTALL)

        try:
            corrected_python_code = python_code_blocks[0]
        except Exception:
            corrected_python_code = model_response

        return corrected_python_code

    def getanalysisoutput(self, user_input=None):
        self.user_input = user_input or self.user_input
        if self.user_input is None:
            raise MissingValueError(["user_input"])
        # print("getting steps")
        steps = self.getanalysissteps()

        # print("getting code")
        analysis_python_code = self.getanalysiscode(steps)
        # print(f"code received:\n{analysis_python_code}\n\n")

        # print(f"Running code:\n{analysis_python_code}\n\n")
        with CapturePrints() as c:
            exec_scope = {"df": self.df, "sns": sns}
            try:
                exec(analysis_python_code, exec_scope)
            except Exception as e:
                error_message = str(e)
                print(f"Error. {type(e).__name__}: {error_message}")
                analysis_python_code = self.correctcode(
                    analysis_python_code, error_message
                )
                exec(analysis_python_code, exec_scope)

        output = c.getvalue()
        # print(f"Output:\n{output}\n\n")
        # print("sending output")
        return output

    def getanalysis(self, user_input=None):
        self.user_input = user_input or self.user_input
        if self.user_input is None:
            raise MissingValueError(["user_input"])
        analysis = ""

        analysis_output = self.getanalysisoutput()

        if len(analysis_output) > 6000:
            analysis_output = analysis_output[0:3000] + "..." + analysis_output[-3000:]

        # print("analysis output: ", analysis_output, "\n\n")

        self.model.set_messages(
            [
                {
                    "prompt": Prompt("system_analysis_pt"),
                    "role": "system",
                },
                {
                    "prompt": Prompt("analysis_output_pt").format(
                        user_input=self.user_input,
                        analysis_output=analysis_output,
                    ),
                    "role": "user",
                },
            ]
        )

        analysis = (
            self.model.run(
                temperature=0.3, top_p=1, frequency_penalty=0, presence_penalty=0
            )
            .choices[0]
            .message.content
        )

        return analysis

    def getvisualizations(self, image_dir=None, user_input=None):
        self.user_input = user_input or self.user_input
        if self.user_input is None:
            raise MissingValueError(["user_input"])
        visualization_steps = self.getvisualizationsteps()
        visualization_python_code = self.getvisualiztioncode(visualization_steps)

        exec_scope = {"df": self.df}
        try:
            exec(visualization_python_code, exec_scope)
        except Exception as e:
            error_message = str(e)
            visualization_python_code = self.correctcode(
                visualization_python_code, error_message
            )
            exec(visualization_python_code, exec_scope)

        image_dir = image_dir or os.path.join(os.getcwd(), "generated_images")
        try:
            os.makedirs(image_dir, exist_ok=True)
        except Exception:
            # print("Error creating directory")
            pass

        image_list = load_images_in_current_directory()
        png_files = [f for f in os.listdir(os.getcwd()) if f.endswith(".png")]
        for filename in png_files:
            shutil.move(filename, image_dir)
        return image_list

    def run(self, user_input=None):
        self.user_input = user_input or self.user_input
        if self.user_input is None:
            raise MissingValueError(["user_input"])
        analysis = self.getanalysis()
        images = self.getvisualizations()
        return analysis, images


def load_images_in_current_directory():
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
