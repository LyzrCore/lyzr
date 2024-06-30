import os
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI



def get_files_in_directory(directory):
    # This function help us to get the file path along with filename.
    files_list = []

    if os.path.exists(directory) and os.path.isdir(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                files_list.append(file_path)

    return files_list


def story_generator(prompt):
    API_KEY = os.getenv('OPENAI_API_KEY')
    ai = OpenAI(api_key=API_KEY)
   
    response = ai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.1,
        max_tokens=1000)

    story = response.choices[0].text.strip()
    return story

def prompt(user_input):
    prompt = f"""You are an expert to create kid's stories, create a complete story on this {user_input}. 
    Make sure story obeys these points: 
     1. Story should be short and precise.
     2. Story will cover from introduction to climax in 500-700 words. 
     3. Story will proivde valuable learning's for children's.
    """

    return prompt
