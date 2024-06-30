import os
from PIL import Image
from utils import utils
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv; load_dotenv()
from lyzr import VoiceBot

# Setup your config
st.set_page_config(
    page_title="Interactive Audiobook",
    layout="centered",  # or "wide" 
    initial_sidebar_state="auto",
    page_icon="./logo/lyzr-logo-cut.png"
)

# Load and display the logo
image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Interactive Audiobook by Lyzr")
st.markdown("### Welcome to the Interactive Audiobook!")
st.markdown("Interactive Audiobook by Lyzr will convert children's stories into interactive audiobook")

# Custom function to style the app
def style_app():
    # You can put your CSS styles here
    st.markdown("""
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# Interactive Audiobook Application

audio_directory = 'audio'
os.makedirs(audio_directory, exist_ok=True)
original_directory = os.getcwd()
    
# replace this with your openai api key or create an environment variable for storing the key.
API_KEY = os.getenv('OPENAI_API_KEY')

 
def audiobook_agent(user_story:str):
    vb = VoiceBot(api_key=API_KEY)
    try:
        os.chdir(audio_directory)
        vb.text_to_speech(user_story)
    finally:
        os.chdir(original_directory)
    


if __name__ == "__main__":
    style_app() 
    topic = st.text_input('Write breif about the story')
    if st.button('Create'):
        if topic:
            prompt = utils.prompt(user_input=topic)
            story = utils.story_generator(prompt=prompt)
            st.subheader('Glimpse of Story')
            shorten_story = story[:450]
            st.write(shorten_story)
            st.markdown('---')
            st.subheader('Story into audiobook')
            audiobook_agent(user_story=story)
            files = utils.get_files_in_directory(audio_directory)
            audio_file = files[0]
            st.audio(audio_file)             
        else:
            st.warning("Provide the content for story, don't keep it blank")

    
    with st.expander("ℹ️ - About this App"):
        st.markdown("""
        This app uses Lyzr Voice Bot agent to convert books into interactive audiobooks. The QABot agent is built on the powerful Retrieval-Augmented Generation (RAG) model, enhancing your ability to extract valuable insights. For any inquiries or issues, please contact Lyzr.
        
        """)
        st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width = True)
        st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width = True)
        st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width = True)
        st.link_button("Slack", url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw', use_container_width = True)