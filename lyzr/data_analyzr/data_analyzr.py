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
    def __init__(self, df, user_input, gpt_model="gpt-3.5-turbo"):
        self.df = self.cleanDf(df)
        self.user_input = user_input
        self.gpt_model = gpt_model
        self.df_columns = self.df.columns.tolist()
        self.df_head = self.df.head(5)
        self.client = OpenAI()

    def cleanDf(self, df):
        # Removing columns having more than 50% of missing values
        df = df[df.columns[df.isnull().mean() < .5]]

        # Getting the columns which are categorical
        cat_columns = df.select_dtypes(include=['object']).columns

        # Getting the columns which are numerical
        num_columns = df.select_dtypes(include=[np.number]).columns

        # Replacing missing categorical values with the most frequent value(mode)
        df[cat_columns] = df[cat_columns].apply(lambda x: x.fillna(x.mode()[0]))

        # Replacing missing numerical values with the mean
        df[num_columns] = df[num_columns].apply(lambda x: x.fillna(x.mean()))
        
        # Removing "Unnamed:" columns if any
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Removing duplicates
        df = df.drop_duplicates(keep='first')
        
        return df


    def getRecommendations(self, number_of_recommendations=4):

        system_prompt = f"""You are a Senior Data Scientist and intelligent strategic advisor with 10+ Years of Experience. This is a Critical Scenerio. The CEO has given you a dataframe, your job is to list {number_of_recommendations} high level recommendations to Analyze data and get deep insights from the data"""

        user_prompt = f"""On the given dataframe give {number_of_recommendations} advanced and high quality recommendations for analysis to get deep insights from the data.

        Dataframe Head:
        ```python
        {self.df_head}
        ```

        1. Do Not provide shallow and obvious recommendations like "Do Data Cleaning", "Do Data Preprocessing", "Do Data Visualization" etc.
        2. You have to keep the recommendations concise and precise.
        3. Only include recommendations that can be answered by running python code on the dataframe.

        This is an extremely critical scenerio, so only include important and high quality recommendations related to the dataframe. 

        Create Recommendations that provide value and are understandable to the CEO and other CXO's. 
        
        Important: Output the recommendations in Python List format example: 
        ```
        [
        "Recommendation 1", 
        "Recommendation 2",
        ...
        ]
        ```

        Dataframe Coloumns:
        ```python
        {self.df_columns}
        ```

        Now, Write down clear, precise and concise {number_of_recommendations} recommendations to Analyze the Data
        Keep the recommendation advanced yet concise. Only the Recommendations in one short line each.
        """

        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
        model=self.gpt_model,
        messages=messages,
        temperature=0.2
        )

        steps = completion.choices[0].message.content

        try:
            result_list = ast.literal_eval(steps)
            if isinstance(result_list, list) and all(isinstance(item, str) for item in result_list):
                steps = result_list
        except:
            pass

        return steps    


    def getAnalysisSteps(self):

        system_prompt = """You are a Senior Data Scientist with 10+ Years of Experience. This is a Critical Scenerio. The CEO has asked you a question on a dataframe, your job is to list down steps to Analyze the Data and answer the CEO's question. """

        user_prompt = f"""CEO: "{self.user_input}"

Dataframe Head:
```python
{self.df_head}
```

1. The Dataset is already Preprocessed, Cleaned and converted to pandas dataframe.
2. You have to describe on a advanced level what type of analysis needs to be done and on which coloumn and what is the expected output.
3. The steps should be only related to analysis and do NOT include any visualization steps.
4. All the analysis results be printed with `print()` function.
5. The analysis results should be short, simple and human readable, so that the CEO and other CXO's can understand it easily.

This is an extremely critical scenerio, so only include important steps related to CEO's Question and do NOT include any unnecessary steps like imports and data cleaning.

Dataframe Coloumns:
```python
{self.df_columns}
```
Important: Output the steps in python list format eg. ["Step 1", "Step 2", "Step 3"]
Now, Write down clear and precise steps to Analyze the Data and answer the CEO's question: "{self.user_input}"
The CEO's question should be answered in the first line itself.

Write list of steps:
"""

        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
        model=self.gpt_model,
        messages=messages,
        temperature=0.1
        )

        steps = completion.choices[0].message.content

        try:
            result_list = ast.literal_eval(steps)
            if isinstance(result_list, list) and all(isinstance(item, str) for item in result_list):
                steps = result_list
        except:
            pass

        return steps

    
    def getAnalysisCode(self, instructions):

        system_prompt = """You Write Python Function. You are a Senior Data Analyst with 10+ Years of Experience. This is a Critical Scenerio. The CEO has asked you to write Python Function to answer a question on a given data, based on the instructions given by Senior Data Scientist"""

        user_prompt = f"""CEO: {self.user_input}

Dataframe Head: 
{self.df_head}

Data Scientist's Instructions:
{instructions}

Here is a sample output for the Python Function and Code:
```python
import pandas as pd
import <necessory_libraries> # import ALL the necessory libraries

def function_name(dataframe):
    # Write your Python Function here that does the required analysis and answer's CEO's Question
    # Print all the analysis results with `print()` function
                
# Assume `df` is already defined
function_name(df) # Call the function that you wrote with the dataframe as the argument
```

Now, Write down python function to print the answer the CEO's question: {self.user_input}

Important: You are writing code only for analysis and not for visualization. So, do NOT include any visualization code in the function. 

The code should always print the analysis results with `print()` function.

The CEO's question should be answered in the first line that is printed. The analysis results should be clear and concise. You can go into detail once you answer CEO's question.

Just Write the Python code in markdown format, that's it.
        """

        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
        model=self.gpt_model,
        messages=messages,
        temperature=0.1
        )

        model_response = completion.choices[0].message.content

        pattern = r'```python\n(.*?)\n```'
        python_code_blocks = re.findall(pattern, model_response, re.DOTALL)

        try:
            python_code = python_code_blocks[0]
        except:
            python_code = model_response    

        return python_code


    def getVisualizationSteps(self):

        system_prompt = """You are a Senior Data Scientist with 10+ Years of Experience. This is a Critical Scenerio. The CEO has asked you a question on a dataframe, your job is to list down steps to Visualize the Data and create visualization to answer the CEO's question. """

        user_prompt = f"""CEO: "{self.user_input}"

Dataframe Head:
```python
{self.df_head}
```

1. The Dataset is already Preprocessed, Cleaned and converted to pandas dataframe.
2. You have to describe on a advanced level what type of visualization needs to be done and on which coloumn and what is the expected visualization.
3. Save the visualizations with proper labels and names.
4. Create at least one or more visualizations

This is an extremely critical scenerio, so only include important steps related to CEO's Question and do NOT include any unnecessary steps like imports and data cleaning.

Dataframe Coloumns:
```python
{self.df_columns}
```
Important: Output the steps in python list format eg. ["Step 1", "Step 2", "Step 3"]
If the CEO's question does not need a visualization to answer, then create visualization that is related to CEO's question and provides deep insights from the data.

Now, Write down clear and precise steps to create visualization answer the CEO's question: "{self.user_input}"
Write list of steps:
"""

        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
        model=self.gpt_model,
        messages=messages,
        temperature=0.1
        )

        steps = completion.choices[0].message.content

        try:
            result_list = ast.literal_eval(steps)
            if isinstance(result_list, list) and all(isinstance(item, str) for item in result_list):
                steps = result_list
        except:
            pass

        return steps

    
    def getVisualiztionCode(self, instructions):

        system_prompt = """You are a Senior Data Analyst with 10+ Years of Experience. You write reliable python code to create a save visualizations for Data Analysis"""

        user_prompt = f"""This is a Critical Scenerio. The CEO has asked you to write Python code to create visualization that answers his question on a given dataframe, based on the instructions given by Senior Data Scientist
        
CEO: "{self.user_input}"

Dataframe Head: 
{self.df_head}

Data Scientist's Instructions:
{instructions}

Dataframe coloumns:
{self.df_columns}

Here is a sample output for the Python Code:
```python
import pandas as pd
import <necessory_libraries> # import ALL the necessory libraries

def function_name(dataframe):
    # Write your Python Function here that creates visualizations that answer's CEO's Question
    # Save the visualizations with proper labels, names and extensions(in .png format)
    # Do not display the visualization just save it.

# Assume `df` is already defined
function_name(df) # Call the function that you wrote with the dataframe as the argument
```
Now, Write down python code to create Visualizations answer the CEO's question: {self.user_input}
        """

        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
        model=self.gpt_model,
        messages=messages,
        temperature=0.1
        )

        model_response = completion.choices[0].message.content

        pattern = r'```python\n(.*?)\n```'
        python_code_blocks = re.findall(pattern, model_response, re.DOTALL)

        try:
            python_code = python_code_blocks[0]
        except:
            python_code = model_response    

        return python_code


    def correctCode(self, python_code, error_message):
        
        corrected_python_code = ""

        system_prompt = """You are an Expert Python Programmer with more than 10 years of experience. You have to fix the erroneous Python Code written by the Data Scientist. And output the working Python Code."""

        user_prompt = f"""CEO asked the follwoing question: {self.user_input}

        Dataframe Head:
        ```python
        {self.df_head}
        ```
        
        Data Scientist wrote the following python code:
        ```python
        {python_code}
        ```

        Upon runninng the python code, resulted in the following error:
        ```error
        {error_message}
        ```
        
        Dataframe Coloumns:
        ```python
        {self.df_columns}
        ```
        
        Do NOT generate anything other than the corrected Python Code and keep the structure of the corrected code same as the input code.

        Take a deep breath and think step by step and only Write the complete corrected Python code in markdown format.
        """

        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
        model=self.gpt_model,
        messages=messages,
        temperature=0.1
        )

        model_response = completion.choices[0].message.content

        pattern = r'```python\n(.*?)\n```'
        python_code_blocks = re.findall(pattern, model_response, re.DOTALL)

        try:
            corrected_python_code = python_code_blocks[0]
        except:
            corrected_python_code = model_response

        return corrected_python_code


    def getAnalysisOutput(self):
        # print("getting steps")
        steps = self.getAnalysisSteps()
        
        # print("getting code")
        analysis_python_code = self.getAnalysisCode(steps)

        suppress_warning = """
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('display.max_rows', 5)
"""

        analysis_python_code = suppress_warning + analysis_python_code

        # print("Running code")
        with CapturePrints() as c:
            exec_scope = {'df': self.df, 'sns': sns}
            try:
                exec(analysis_python_code, exec_scope)
            except Exception as e:
                error_message = str(e)
                analysis_python_code = self.correctCode(analysis_python_code, error_message)
                exec(analysis_python_code, exec_scope)

        output = c.getvalue()        

        # print("sending output")
        return output


    def getAnalysis(self):
        analysis = ""

        analysis_output = self.getAnalysisOutput() 

        if len(analysis_output) > 6000:
            analysis_output = analysis_output[0:3000] + "..." + analysis_output[-3000:]        

        # print("analysis output: ", analysis_output, "\n\n")

        system_prompt = """You are an intelligent data analyst capable of understanding an analytics output result and share them in simple understandable language catered to business users and data analysts.\n\nYou will be provided with the user_query and the analysis_output. You will have to understand the analysis results and generate clear simplified explanations along with corresponding data points.\n\nGenerate 3 analysis explanations, limiting the overall response to 100 words. \n\nPresent the output as bullet points.\n\nRank all your insights and only share the top 3 ones.  Focus on clarity and conciseness.\n\nDo not describe the dataset or the prompt.\nDo not speak about charts.\nDo not share any title. \n\n"""

        user_prompt = f'''User Query: {self.user_input}\n For the above user query, after analysing with python we get the following output \n Analysis Output: \n {analysis_output}'''

        analysis_messages=[
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


    def getVisualizations(self):
        visualization_steps = self.getVisualizationSteps()
        visualization_python_code = self.getVisualiztionCode(visualization_steps)

        exec_scope = {'df': self.df}
        try:
            exec(visualization_python_code, exec_scope)
        except Exception as e:
            error_message = str(e)
            visualization_python_code = self.correctCode(visualization_python_code, error_message)
            exec(visualization_python_code, exec_scope)

        def load_images_in_current_directory():
            current_directory = os.getcwd()
            image_data = {}

            for file in os.listdir(current_directory):
                if file.lower().endswith('.png'):
                    image_path = os.path.join(current_directory, file)
                    with Image.open(image_path) as img:
                        byte_arr = io.BytesIO()
                        img.save(byte_arr, format='PNG')
                        image_data[file] = byte_arr.getvalue()

            return image_data

        image_list = load_images_in_current_directory()
        
        if not os.path.exists('generated_images'):
            os.makedirs('generated_images')

        destination_directory = os.path.join(os.getcwd(), "generated_images")

        png_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.png')]

        for file_name in png_files:
            shutil.move(file_name, destination_directory)           
            
        return image_list           


    def run(self):        
        analysis = self.getAnalysis()
        images = self.getVisualizations()
        return analysis, images    
        