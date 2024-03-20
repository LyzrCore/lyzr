import streamlit as st
import os
from lyzr import ChatBot
import weaviate
import openai

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Define the Weaviate vector store parameters
vector_store_params = {
    "vector_store_type": "WeaviateVectorStore",
    "url": "https://newytchatbot-tr5zy9ee.weaviate.network",
    "api_key": "8C9HovD2PvtgREyGn77YGPyLnCSWB5vu8t7L",
    "index_name": "Akshay"  # First letter should be capital
}

# Alternatively, for a local Embedded Weaviate vector store
# vector_store_params = {
#     "vector_store_type": "WeaviateVectorStore",
#     "index_name": "IndexName"  # First letter should be capital
# }

# Function to initialize the Chatbot with the provided video URL
def initialize_chatbot(video_url):
    return ChatBot.youtube_chat(urls=[video_url], vector_store_params=vector_store_params)

# Main function to create the Streamlit app
def main():
    st.title("YouTube Chatbot")
    
    # Input field for entering the YouTube video URL
    video_url = st.text_input("Enter YouTube Video URL:")
    
    # Initialize the Chatbot if the video URL is provided
    if video_url:
        chatbot = initialize_chatbot(video_url)
        
        # Input field for asking questions
        question = st.text_input("Ask a question related to the video content:")
        
        # Button to submit the question and get the response
        if st.button("Ask"):
            if question:
                response = chatbot.chat(question)
                st.write("Chatbot's Response:")
                st.write(response.response)
            else:
                st.warning("Please enter a question.")
    
# Run the Streamlit app
if __name__ == "__main__":
    main()
