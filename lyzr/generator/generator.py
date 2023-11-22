from openai import OpenAI
import pandas as pd

class Generator:
    def __init__(self, user_input=None, df=None, gpt_model="gpt-3.5-turbo"):
        if user_input is None or df is None:
            raise ValueError("Please provide user query and dataframe")
        
        self.client = OpenAI()
        self.user_query = user_input
        self.df = df
        self.gpt_model = gpt_model

    def describe_dataset(self):

        if self.df is None:
            raise ValueError("Please provide a dataframe")

        system_prompt = f""" You are a Senior Data Scientist and intelligent strategic advisor with 10+ Years of Experience. This is a Critical Scenerio. The CEO has given you a Dataset Sample, your job is to create a detailed description about the data for CXO's"""

        user_prompt = f"""Dataset Sample (with top five row including the coloumn names): \n{self.df.head(5)} 

The description should be at most one paragraph long and should be understandable by business users and other data scientists. The description should be deeply insightful yet simple for the readers.         
Now Write a detailed description of the data:
"""

        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
        model=self.gpt_model,
        messages=messages,
        temperature=1,
        top_p=0.3,
        frequency_penalty=0.7,
        presence_penalty=0.3
        )

        description = completion.choices[0].message.content

        return description

    def ai_queries_df(self):
        
        if self.df is None:
            raise ValueError("Please provide a dataframe")
        
        system_prompt = f"""You are an expert python programmer with good working knowledge on Pandas, Scikit Learn libraries.\n\nYour task is to come up with 20 Natural Language Queries to analyse the provided dataset which could be executed using pandas and/or scikit learn libraries.\n\nYou will be provided the dataset sample including the dataset sample and dataset description for you to understand the context.\n\nThe natural language queries should not explicitly mention the statistical model or chart type.\n\nYour queries fall in below categories,\nExploratory Analysis\nRegression Analysis\nCorrelation Analysis\nClustering Analysis\nTime Series Analysis\n\n"""

        dataset_description = self.describe_dataset()
        df_head = self.df.head(5)

        user_prompt = f""" Dataset Description: {dataset_description}\n\nDataset Sample: {df_head} """

        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
        model=self.gpt_model,
        messages=messages,
        temperature=1,
        top_p=0.3,
        frequency_penalty=0.7,
        presence_penalty=0.3
        )

        ai_queries_df = completion.choices[0].message.content

        return ai_queries_df

    def insights(self, analysis_output=None):
        if analysis_output is None:
            raise ValueError("Please provide analysis output")

        system_prompt = f"""You are an intelligent data analyst capable of understanding an analytics output result and share them in simple understandable language catered to business users and data analysts.\n\nYou will be provided with the user_query and the analysis_output. You will have to understand the analysis results and generate clear simplified explanations along with corresponding data points.\n\nGenerate 3 analysis explanations, limiting the overall response to 100 words. \n\nPresent the output as bullet points.\n\nRank all your insights and only share the top 3 ones.  Focus on clarity and conciseness.\n\nDo not describe the dataset or the prompt.\nDo not speak about charts.\nDo not share any title. \n\n"""

        user_prompt = f"""User Query: {self.user_query}\nAnalysis Output: {analysis_output}"""

        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
        model=self.gpt_model,
        messages=messages,
        temperature=0.3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )

        insights = completion.choices[0].message.content
        return insights

    def recommendations(self, insights=None):
        if insights is None:
            raise ValueError("Please provide insights")
        
        system_prompt = f"""You are an intelligent strategic advisor who is good at understanding the analysis results and providing intelligent recommendations to the user.\n\nYou will have the original query asked by the user and the analysis results generated as inputs.\n\nYou will have to develop 3 recommendations that the user can implement to solve the problem stated in their query.\n\nIf it is not a problem, then your 3 recommendations should be designed to help the user improve the outcome.\n\nFor each recommendation, also provide ‘Basis of the recommendation’ and ‘Impact of the recommendation if implemented’.\n\nPresent the output as bullet points.\n\nRank all your insights and only share the top 3 ones.  Focus on clarity and succinctness.\n\nDo not describe the dataset or the prompt.\nDo not speak about charts.\nDo not share any title. \n\n"""

        user_prompt = f"""User Query: {self.user_query}\nAnalysis results & insights: {insights}"""

        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
        model=self.gpt_model,
        messages=messages,
        temperature=1,
        top_p=0.3,
        frequency_penalty=0.7,
        presence_penalty=0.3
        )

        recommendations = completion.choices[0].message.content        
        return recommendations

    def tasks(self, insights=None, recommendations=None):
        if insights  is None or recommendations is None:
            raise ValueError("Please provide insights and recommendations")
        
        system_prompt = f"""You are the world’s best to-do creator and assigner. All the to-dos are executable in 60 minutes or less.\n\nYou can understand any recommendation or set of recommendations, break it down into a list of to-do’s that the user could do in 60 minutes or less, which will help the user execute the larger recommendations. \n\nThe to-dos should not be complex that would require multiple tasks to be executed at once. It should be sequential and the user should be able to complete the to-dos one at a time.\n\nGenerate to-dos that are specific and quantifiable, ensuring that each task mentions a clear numeric target. Each task should be feasible to complete within a 2 to 3-hour timeframe. For example, instead of saying ‘Speak to customers’, the task should be ‘Speak to 10 customers’. Likewise, instead of ‘Create a list of documents’, it should specify ‘Create a list of 30 documents’.\n\nEach to-do should be between 15 and 25 words. And generate not more than 5 to-dos. \n\nThe to-dos should always start with one of the below:\nCreate a list,\nWrite down, \nSpeak to,\nSetup 30 minutes to dive deep and analyze,\nPlan to,\nDo,\nReview,\nDraft,\nComplete,\nBegin,\nDiscuss,\nSchedule,\nConfirm,\nReach out to,\nTest,\nAttend,\nAllocate time for"""   

        user_prompt = f"""User Query: {self.user_query}\n\nAnalysis results & insights: {insights}\n\nRecommendations: {recommendations}\n"""

        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        ]

        completion = self.client.chat.completions.create(
        model=self.gpt_model,
        messages=messages,
        temperature=1,
        top_p=0.3,
        frequency_penalty=0.7,
        presence_penalty=0.3
        )

        tasks = completion.choices[0].message.content
        return tasks
