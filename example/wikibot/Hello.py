import os
os.system("playwright install")
from PIL import Image
import streamlit as st
from lyzr import ChatBot
import nest_asyncio
#from typing import Union
import openai

os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

# Apply nest_asyncio
nest_asyncio.apply()

# Custom function to style the app
def style_app():
    # You can put your CSS styles here
    st.markdown("""
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    .button-group { display: flex; justify-content: space-between; }
    .button { flex: 1; margin: 0 0.5rem; background-color: #4CAF50; color: white; border: none; padding: 15px 32px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; transition-duration: 0.4s; cursor: pointer; border-radius: 12px; }
    .button:hover { background-color: #45a049; }
    .button:active { background-color: #3e8e41; transform: translateY(2px); }
    .input-text { padding: 10px; border-radius: 12px; border: 1px solid #ccc; width: 100%; box-sizing: border-box; margin-bottom: 10px; }
    .input-text:focus { border-color: #4CAF50; }
    </style>
    """, unsafe_allow_html=True)

# Call the function to apply the styles
style_app()

# Load and display the logo
image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("WikiBot")
st.markdown("### Welcome to WikiBot! 🤖")
st.markdown("Interact with Wikipedia Through Chatbots (built using LYZR SDK)")

# Define function to initialize chatbot
def initialize_chatbot(url):
    # Replace these parameters with your actual Weaviate Vector Store parameters
    vector_store_params = {
        "vector_store_type": "WeaviateVectorStore",
        "url": "https://wikibot-zn4f2emj.weaviate.network",
        "api_key": "JKkfz6V0kxYa3WX40FdyiFf2DDTM3GlAIyTT",
        "index_name": "Akshay"
    }
    # Initialize the Webpage Chatbot with the provided URL
    return ChatBot.webpage_chat(
        url=url,
        vector_store_params=vector_store_params
    )

# Main function to run the Streamlit app
def main():
    # User input for URL
    url = st.text_input("Enter the URL of the webpage:")
    
    # Check if URL is provided
    if url:
        # Initialize the chatbot with the provided URL
        chatbot = initialize_chatbot(url)
      
        # Pre-defined prompts
        prompts = [
            "What is the summary of this page?",
            "Can you explain the history of this topic?",
            "Who are the notable figures related to this topic?",
            "What are the controversies surrounding this topic?"
        ]
        
        # Display pre-defined prompts as buttons
        col1, col2 = st.columns(2)
        for i, prompt in enumerate(prompts):
            if i % 2 == 0:
                button = col1.button(prompt, key=f"button_{i}")
            else:
                button = col2.button(prompt, key=f"button_{i}")
            
            # Check if button is clicked
            if button:
                # Chat with the chatbot
                response = chatbot.chat(prompt)
                
                # Display chatbot's response
                st.write("Chatbot's Response:")
                st.write(response.response)
                
            
        # User's own question input field
        user_question = st.text_input("Enter your own question:")
        
        # Chat with the chatbot if user provides a question
        if user_question:
            response = chatbot.chat(user_question)
            
            # Display chatbot's response
            st.write("Chatbot's Response:")
            st.write(response.response)
            
# Run the Streamlit app
if __name__ == "__main__":
    main()

# Footer or any additional information
with st.expander("ℹ️ - About this App"):
    st.markdown("""
    WikiBot enables seamless interaction with Wikipedia through chatbots powered by the LYZR SDK. Effortlessly explore topics, utilize pre-defined prompts for summaries, and pose inquiries directly within your browser. For inquiries or assistance, reach out to Lyzr.
    """)
    
    
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width = True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width = True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width = True)
    st.link_button("Slack", url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw', use_container_width = True)
