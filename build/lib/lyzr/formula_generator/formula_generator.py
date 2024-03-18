from openai import OpenAI


class FormulaGen:
    def spreadsheets(self, initial_prompt):
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in excel and google sheets. Your only job is to guide me in solving my problem using Excel formulas. Please provide me with the exact solution a formula I need for my problem. nothing but the formula.",
                },
                {"role": "user", "content": initial_prompt},
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        initial_response = response.choices[0].message.content

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert in excel and google sheets. Here is my situation, {initial_prompt}. The solution I got from you was {initial_response}.",
                },
                {
                    "role": "user",
                    "content": "Showcase the solution and briefly explain the solution to me, use an example if suitable. Enclose any formulas in ```formula```.",
                },
            ],
            temperature=1,
            max_tokens=800,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        final_response = response.choices[0].message.content
        return final_response

    def regular_expression(self, initial_prompt):
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Regular Expressions. Your only job is to guide me in solving my problem using Regular Expressions. Please provide me with the exact Regular Expression I need for my problem. print nothing but the Regular Expression.",
                },
                {"role": "user", "content": initial_prompt},
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        initial_response = response.choices[0].message.content

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert in Regular Expressions. Here is my situation, {initial_prompt}. The solution I got from you was {initial_response}.",
                },
                {
                    "role": "user",
                    "content": "Showcase the solution and briefly explain the solution to me. Enclose any formulas in ```formula```.",
                },
            ],
            temperature=1,
            max_tokens=800,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        final_response = response.choices[0].message.content
        return final_response

    def text_to_sql(self, initial_prompt):
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in SQL, especially MySQL. Your only job is to guide me in solving my problem using SQL queries. Please provide me with the exact SQL I need to solve my problem. print nothing but the SQL Querry.",
                },
                {"role": "user", "content": initial_prompt},
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        initial_response = response.choices[0].message.content

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert in SQL, especially MySQL. Here is my situation, {initial_prompt}. The solution I got from you was {initial_response}.",
                },
                {
                    "role": "user",
                    "content": "Showcase the query and briefly explain the query to me with an example if suitable. Enclose any formulas in ```formula```.",
                },
            ],
            temperature=1,
            max_tokens=800,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        final_response = response.choices[0].message.content
        return final_response
