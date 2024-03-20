import streamlit as st
import os
from PIL import Image
from utils import utils
from lyzr import QABot 
import openai
from dotenv import load_dotenv; load_dotenv()


st.set_page_config(
    page_title="Lyzr Employee-HR Q&A",
    layout="centered", 
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)


image = Image.open("lyzr-logo.png")
st.image(image, width=150)


st.title("Employee-HR Q&A by Lyzr")
st.markdown("### Welcome to the Employee-HR QABot!")


openai.api_key = os.getenv('OPENAI_API_KEY')


# create directory if it doesn't exist
company_documents = "company_documents"
os.makedirs(company_documents, exist_ok=True)


# HR Page
def hr_page():
    st.title("HR Page")
    st.subheader("Upload Company Documents")
    # Upload PDF file
    uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"])
    if uploaded_file is not None:
        utils.save_uploaded_file(uploaded_file)
    else:
        utils.remove_existing_files(company_documents)



def hr_rag():
    # This function will implement the HR Bot.
    path = utils.get_files_in_directory(company_documents)
    path = path[0]

    rag = QABot.pdf_qa(
        input_files=[str(path)],
        llm_params={"model": "gpt-3.5-turbo"},
    )

    return rag

    

# Employee Page
def employee_page():
    file = []
    for filename in os.listdir(company_documents):
        file_path = os.path.join(company_documents, filename)
        file.append(file_path)

    if len(file) > 0:
        st.title("Employee Page")
        st.subheader("Ask your questions")
        question = st.text_input("What is your questions?")

        if st.button("Get Answer"):
            rag = hr_rag()
            response = rag.query(question)
            st.markdown(f"""{response.response}""")

    else:
        st.error('HR has not uploaded the documents')


def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["HR Page", "Employee Page"])

    if selection == "HR Page":
        hr_page()
    elif selection == "Employee Page":
        employee_page()

    with st.expander("ℹ️ - About this App"):
        st.markdown(
            """
        This app uses Lyzr Core to generate answers from the PDF document uploaded by HR. This an Employee-HR QnA app where employees directly ask questions and this app will give the answers related to the document which is provided by the HR. For any inquiries or issues, please contact Lyzr.

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

if __name__ == "__main__":
    main()
