import os
from PIL import Image
import streamlit as st
from pathlib import Path
from utils import utils
from dotenv import load_dotenv; load_dotenv()
from lyzr import QABot

# Setup your config
st.set_page_config(
    page_title="Event Planner",
    layout="centered",  # or "wide" 
    initial_sidebar_state="auto",
    page_icon="./logo/lyzr-logo-cut.png"
)

# Load and display the logo
image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Event Planner QnA by Lyzr")
st.markdown("### Welcome to the Event Planner QnA!")
st.markdown("Event planners is build on a QABot Agent to answer attendee questions. Attendees could ask questions about schedules, locations, speakers, or logistics")

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

# Event Planner QnA Application
    
# replace this with your openai api key or create an environment variable for storing the key.
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY') 


def event_planner_qa():
    path = utils.get_files_in_directory('data')
    planner_qa = QABot.docx_qa(
        input_files=[Path(str(path[1]))]
    )

    return planner_qa

def file_checker():
    file = []
    for filename in os.listdir('data'):
        file_path = os.path.join('data', filename)
        file.append(file_path)

    return file

if __name__ == "__main__":
    style_app()
    file = file_checker()
    if file is not None:
        st.subheader('Plan your dream event!')
        question = st.text_input('Write you query')
        if st.button('Submit'):
            if question is not None:
                event_agent = event_planner_qa()
                response = event_agent.query(question)
                st.markdown('---')
                st.subheader('Response')
                st.write(response.response)
            else:
                st.warning("Ask question, don't keep it blank")

    with st.expander("ℹ️ - About this App"):
        st.markdown("""
        This app uses Lyzr QABot agent to give answer attendee questions. The QABot agent is built on the powerful Retrieval-Augmented Generation (RAG) model, enhancing your ability to extract valuable insights. For any inquiries or issues, please contact Lyzr.
        
        """)
        st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width = True)
        st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width = True)
        st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width = True)
        st.link_button("Slack", url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw', use_container_width = True)