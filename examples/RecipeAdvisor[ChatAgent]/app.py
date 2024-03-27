from PIL import Image
import streamlit as st
import openai
import os
import time
from lyzr import ChatBot
from dotenv import load_dotenv; load_dotenv()

st.set_page_config(
    page_title="Recipe Advisor",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="logo\lyzr-logo-cut.png",
)

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load and display the logo
image = Image.open("logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Recipe Advisor")
st.markdown("### Welcome to the Lyzr Recipe Bot!")


# Initialize openai api key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# Generate a unique index name based on the current timestamp
unique_index_name = f"IndexName_{int(time.time())}"
vector_store_params = {"index_name": unique_index_name}
st.session_state["chatbot"] = ChatBot.pdf_chat(
    input_files=["dinnerRecipe.pdf"], vector_store_params=vector_store_params
)

# # Inform the user that the files have been uploaded and processed
# st.success("PDFs uploaded and processed. You can now interact with the chatbot.")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "chatbot" in st.session_state:
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.session_state["chatbot"].chat(prompt)
            chat_response = response.response
            response = st.write(chat_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": chat_response}
        )
else:
    st.warning("Please upload PDF files to continue.")


# Footer or any additional information
with st.expander("ℹ️ - About this App"):
    st.markdown(
        """
    This app uses Lyzr ChatBot. It gives you multiple suggestion for the recipes that can be cooked using available ingredients you have. For any inquiries or issues, please contact Lyzr.

    """
    )
    st.link_button("Lyzr", url="https://www.lyzr.ai/", use_container_width=True)
    st.link_button(
        "Book a Demo", url="https://www.lyzr.ai/book-demo/", use_container_width=True
    )
    st.link_button(
        "Discord", url="https://discord.gg/nm7zSyEFA2", use_container_width=True
    )
    st.link_button(
        "Slack",
        url="https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw",
        use_container_width=True,
    )