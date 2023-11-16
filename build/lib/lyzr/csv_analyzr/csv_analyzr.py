import pandas as pd
import openai
import re

class CsvAnalyzr:
    def __init__(self, csv_path, user_input, gpt_model="gpt-3.5-turbo"):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.user_input = user_input
        self.gpt_model = gpt_model
        self.df_columns = self.df.columns.tolist()
        self.df_head = self.df.head(5)


    def getSteps(self):

        system_prompt = """You are a Senior Data Scientist with 10+ Years of Experience. This is a Critical Scenerio. The CEO has asked you a question on a given data, your job is to list down steps to Analyze the Data and answer the CEO's question. """

        user_prompt = f"""CEO: {self.user_input}

        Dataframe Head: 
        {self.df_head}

        
        1. The Dataset is already Preprocessed, Cleaned and converted to pandas dataframe.
        2. You have to describe on a advanced level what type of analysis needs to be done and on which coloumn and what is the expected output.

        This is an extremely critical scenerio, so only include important steps related to CEO's Question and do NOT include any unnecessary steps like imports and data cleaning.

        Now, Write down the steps to Analyze the Data and answer the CEO's question: {self.user_input}
        """

        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]

        completion = openai.ChatCompletion.create(model=self.gpt_model, temperature = 0, messages=messages)

        steps = completion.choices[0].message.content


        return steps


    def correctCode(self, python_code, error_message):
        
        corrected_python_code = ""

        system_prompt = """You are an Expert Python Programmer with more than 10 years of experience. You have to fix the erroneous Python Code written by the Data Scientist. And output the working Python Code."""

        user_prompt = f"""CEO asked the follwoing question: {self.user_input}

        Dataframe Head:
        ```
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
        
        Do NOT generate anything other than the corrected Python Code and keep the structure of the corrected code same as the input code.

        Take a deep breath and think step by step and only Write the corrected Python code in markdown format.
        """

        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]

        completion = openai.ChatCompletion.create(model=self.gpt_model, temperature = 0, messages=messages)

        model_response = completion.choices[0].message.content

        pattern = r'```python\n(.*?)\n```'
        python_code_blocks = re.findall(pattern, model_response, re.DOTALL)

        try:
            corrected_python_code = python_code_blocks[0]
        except:
            corrected_python_code = model_response

        return corrected_python_code


    def getCode(self, instructions):

        system_prompt = """You Write Python Function. You are a Senior Data Analyst with 10+ Years of Experience. This is a Critical Scenerio. The CEO has asked you to write Python Function to answer a question on a given data, based on the instructions given by Senior Data Scientist"""

        user_prompt = f"""CEO: {self.user_input}

        Dataframe Head: 
        {self.df_head}
        
        Data Scientist's Instructions:{instructions}

        Here is a sample output for the Python Function:
        ```python
        import pandas as pd
        import <necessory_libraries> # import ALL the necessory libraries

        def function_name(dataframe):
            # Write your Python Function here that does the required analysis and answer's CEO's Question
           
        if __name__ == "__main__":
            df = pandas.read_csv("{self.csv_path}") # Do NOT change this line
            function_name(df) # Call the function that you wrote with the dataframe as the argument
        ```

        Now, Write down python function to answer the CEO's question: {self.user_input}

        Just Write the Python Function in markdown format, that's it.
        """

        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]

        completion = openai.ChatCompletion.create(model=self.gpt_model, temperature = 0, messages=messages)

        model_response = completion.choices[0].message.content

        pattern = r'```python\n(.*?)\n```'
        python_code_blocks = re.findall(pattern, model_response, re.DOTALL)

        try:
            python_code = python_code_blocks[0]
        except:
            python_code = model_response    

        return python_code



    def run(self, data_scientist_instructions=None):
        if data_scientist_instructions is None:
            data_scientist_instructions = self.getSteps()

        python_code = self.getCode(instructions=data_scientist_instructions)

        try:
            exec(python_code)
            return python_code
        except Exception as e:
            print("Failed to Analyze CSV")
            error_message = f"Error: {str(e)}"
            return "print(error_message)"