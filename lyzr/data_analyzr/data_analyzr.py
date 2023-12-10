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

from lyzr.base.prompt import Prompt
from lyzr.base.llms import LLM, get_model
from lyzr.base.errors import MissingValueError

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
    def __init__(self, df=None, user_input=None, gpt_model=None):
        if df is None:
            raise MissingValueError(["dataframe"])
        self.df = self.cleandf(df)
        self.user_input = user_input
        self.gpt_model = gpt_model or os.environ.get("GPT_MODEL", "gpt-3.5-turbo")
        self.df_columns = self.df.columns.tolist()
        self.df_head = self.df.head(5)
        self.client = OpenAI()

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
        system_prompt = Prompt("system_recommendations_pt").format(
            number_of_recommendations=number_of_recommendations,
        )
        user_prompt = Prompt("analysis_recommendations_pt").format(
            number_of_recommendations=number_of_recommendations,
            df_head=self.df_head,
            df_columns=self.df_columns,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
            model=self.gpt_model, messages=messages, temperature=0.2
        )

        steps = completion.choices[0].message.content

        return steps

    def getanalysissteps(self, user_input=None):
        self.user_input = user_input or self.user_input
        if self.user_input is None:
            raise MissingValueError(["user_input"])

        system_prompt = Prompt("system_analysis_pt")
        user_prompt = Prompt("analysis_steps_pt").format(
            user_input=self.user_input,
            df_head=self.df_head,
            df_columns=self.df_columns,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
            model=self.gpt_model, messages=messages, temperature=0.1
        )

        steps = completion.choices[0].message.content

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

        system_prompt = Prompt("system_code_pt")
        user_prompt = Prompt("analysis_code_pt").format(self.user_input, instructions)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
            model=self.gpt_model, messages=messages, temperature=0.1
        )

        model_response = completion.choices[0].message.content

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
        system_prompt = Prompt("system_visualization_pt")

        user_prompt = Prompt("visualization_steps_pt").format(
            self.user_input, self.df_head, self.df_columns
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
            model=self.gpt_model, messages=messages, temperature=0.1
        )

        steps = completion.choices[0].message.content

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

        system_prompt = Prompt("system_vis_code_pt")
        user_prompt = Prompt("visualization_code_pt").format(
            self.user_input,
            instructions,
            self.df_head,
            self.df_columns,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
            model=self.gpt_model, messages=messages, temperature=0.1
        )

        model_response = completion.choices[0].message.content

        pattern = r"```python\n(.*?)\n```"
        python_code_blocks = re.findall(pattern, model_response, re.DOTALL)

        try:
            python_code = python_code_blocks[0]
        except Exception:
            python_code = model_response

        return python_code

    def correctcode(self, python_code, error_message):
        corrected_python_code = ""

        system_prompt = Prompt("system_correct_code_pt")
        user_prompt = Prompt("correct_code_pt").format(
            python_code, error_message, self.df_columns, self.df_head
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
            model=self.gpt_model, messages=messages, temperature=0.1
        )

        model_response = completion.choices[0].message.content

        pattern = r"```python\n(.*?)\n```"
        python_code_blocks = re.findall(pattern, model_response, re.DOTALL)

        try:
            corrected_python_code = python_code_blocks[0]
        except Exception:
            corrected_python_code = model_response

        return corrected_python_code

    def getanalysisoutput(self):
        # print("getting steps")
        steps = self.getanalysissteps()

        # print("getting code")
        analysis_python_code = self.getanalysiscode(steps)

        suppress_warning = """
            import warnings
            warnings.filterwarnings("ignore")
            import pandas as pd
            pd.set_option('display.max_rows', 5)
            """

        analysis_python_code = suppress_warning + analysis_python_code

        # print("Running code")
        with CapturePrints() as c:
            exec_scope = {"df": self.df, "sns": sns}
            try:
                exec(analysis_python_code, exec_scope)
            except Exception as e:
                error_message = str(e)
                analysis_python_code = self.correctcode(
                    analysis_python_code, error_message
                )
                exec(analysis_python_code, exec_scope)

        output = c.getvalue()

        # print("sending output")
        return output

    def getanalysis(self):
        analysis = ""

        analysis_output = self.getanalysisoutput()

        if len(analysis_output) > 6000:
            analysis_output = analysis_output[0:3000] + "..." + analysis_output[-3000:]

        # print("analysis output: ", analysis_output, "\n\n")

        system_prompt = Prompt("system_analysis_pt")
        user_prompt = Prompt("analysis_output_pt").format(
            self.user_input, analysis_output
        )

        analysis_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=analysis_messages,
            temperature=0.3,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        analysis = completion.choices[0].message.content

        return analysis

    def getvisualizations(self):
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

        def load_images_in_current_directory():
            current_directory = os.getcwd()
            image_data = {}

            for file in os.listdir(current_directory):
                if file.lower().endswith(".png"):
                    image_path = os.path.join(current_directory, file)
                    with Image.open(image_path) as img:
                        byte_arr = io.BytesIO()
                        img.save(byte_arr, format="PNG")
                        image_data[file] = byte_arr.getvalue()

            return image_data

        image_list = load_images_in_current_directory()

        if not os.path.exists("generated_images"):
            os.makedirs("generated_images")

        destination_directory = os.path.join(os.getcwd(), "generated_images")

        png_files = [f for f in os.listdir(os.getcwd()) if f.endswith(".png")]

        for file_name in png_files:
            shutil.move(file_name, destination_directory)

        return image_list

    def run(self):
        analysis = self.getanalysis()
        images = self.getvisualizations()
        return analysis, images
