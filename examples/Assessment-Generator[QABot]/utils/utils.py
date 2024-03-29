import os
import shutil
import streamlit as st
from pathlib import Path
import pandas as pd

def remove_existing_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            st.error(f"Error while removing existing files: {e}")


def get_files_in_directory(directory):
    # This function help us to get the file path along with filename.
    files_list = []

    if os.path.exists(directory) and os.path.isdir(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                files_list.append(file_path)

    return files_list

def save_uploaded_file(uploaded_file):
    # Function to save uploaded file
    remove_existing_files('data')

    file_path = os.path.join('data', uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.read())
    st.success("File uploaded successfully")

    rev = uploaded_file.name[::-1]
    rev = rev[4:]
    filename = rev[::-1]
    
    return filename[:3]


def user_subject_topics(agent, subject, topics_lst):
        resposne_flash = {}
        for topic in topics_lst:
            prompt = f"""You are an expert of this {subject}, Can you write down 3-5 important questions on this {subject} and its topics: {topic} """
            response = agent.query(prompt)
            if response is not None:
                if response.response == 'Empty Response':
                    st.warning('Please provide valid pdf')

                elif response.response != 'Empty Response':
                    # st.subheader("These are the Important Questions, you should prepare")
                    # st.write(response.response)    
                    resposne_flash[topic] = response.response.split('?')
        
        return resposne_flash


def flashcard_viewer(response:dict):
    for topic, questions in response.items():
        st.subheader(topic)
        for question in questions:
            st.write(question)
        st.markdown("---")  

