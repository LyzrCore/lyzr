import os
from PIL import Image
from utils import utils
import streamlit as st
from dotenv import load_dotenv; load_dotenv()
from lyzr import QABot

# Setup your config
st.set_page_config(
    page_title="Assesement Generator",
    layout="centered",  # or "wide" 
    initial_sidebar_state="auto",
    page_icon="./logo/lyzr-logo-cut.png"
)

# Load and display the logo
image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Assesement Generator by Lyzr")
st.markdown("### Welcome to the Assesement Generator!")
st.markdown("Assesement Generator by Lyzr will provide you 10 insightful questions on the textbook pdf you have uploaded")

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

# Assesement Generator Application
    
# replace this with your openai api key or create an environment variable for storing the key.
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# create directory if it doesn't exist
data = "data"
os.makedirs(data, exist_ok=True)

 

def vector_store_configuration(bookname):
    vector_store_params = {
    "vector_store_type": "WeaviateVectorStore",
    "url": os.getenv('VECTOR_STORE_URL'), # replce the url with your weaviate cluster url
    "api_key": os.getenv('VECTOR_STORE_API'), # replace the api with your weaviate cluster api
    "index_name": f"Book{bookname}IndexName" 
  }
    
    return vector_store_params


def smartstudy_bot(filepath, vector_params):
    "This function will implement the Lyzr's QA agent to summarize the Youtube Video"
    smartstudy = QABot.pdf_qa(
            input_files=[str(filepath)],
            vector_store_params=vector_params
        )
    
    return smartstudy

if __name__ == "__main__":
    style_app()
    uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"])
    if uploaded_file is not None:
        filename = utils.save_uploaded_file(uploaded_file)
        subject_name = st.text_input("Write the subject of your Book")
        topics = st.text_input("Enter topics (by comma seperated)")
        topics_list = [topic.strip() for topic in topics.split(",") if topic.strip()]

        if topics_list is not None:
            if st.button("Generate"):
                path = utils.get_files_in_directory(data)
                filepath = path[0]
                vector_params = vector_store_configuration(filename)
                study_agent = smartstudy_bot(filepath=filepath, vector_params=vector_params)
                if study_agent is not None:
                    topic_response = utils.user_subject_topics(agent=study_agent, subject=subject_name, topics_lst=topics_list)
                    utils.flashcard_viewer(response=topic_response)
    else:
        utils.remove_existing_files(data)
        st.warning('Please Upload pdf file')

    with st.expander("ℹ️ - About this App"):
        st.markdown("""
        This app uses Lyzr QABot agent to 10 qustions from TextBook pdf. The QABot agent is built on the powerful Retrieval-Augmented Generation (RAG) model, enhancing your ability to extract valuable insights. For any inquiries or issues, please contact Lyzr.
        
        """)
        st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width = True)
        st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width = True)
        st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width = True)
        st.link_button("Slack", url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw', use_container_width = True)