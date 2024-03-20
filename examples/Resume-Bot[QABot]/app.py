import os
import streamlit as st
import shutil
from PIL import Image
from lyzr import QABot

st.set_page_config(
    page_title="Lyzr QA Bot",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

# Load and display the logo
image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Lyzr CV QA-Bot Demo")
st.markdown("### Welcome to the Lyzr QA-Bot!")
st.markdown("Upload your Resume and Ask your queries.")


# Input for API key
# api_key = st.sidebar.text_input("API Key", type="password")
# if api_key:
#     os.environ["OPENAI_API_KEY"] = api_key
# else:
#     # Prompt for API key if not provided
#     st.sidebar.warning("Please enter your API key to proceed.")


# vector_store_params = {
#     "vector_store_type": "WeaviateVectorStore",
#     "index_name": "IndexName" # first letter should be capital
# }


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


# Set the local directory
data_directory = "data"

# Create the data directory if it doesn't exist
os.makedirs(data_directory, exist_ok=True)

# Remove existing files in the data directory
remove_existing_files(data_directory)


# File upload widget
uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded PDF file to the data directory
    file_path = os.path.join(data_directory, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getvalue())
    
    # Display the path of the stored file
    st.success(f"File successfully saved")


def get_files_in_directory(directory="data"):
    # This function help us to get the file path along with filename.
    files_list = []

    # Ensure the directory exists
    if os.path.exists(directory) and os.path.isdir(directory):
        # Iterate through all files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Check if the path points to a file (not a directory)
            if os.path.isfile(file_path):
                files_list.append(file_path)

    return files_list


def rag_implementation():
    # This function will implement RAG Lyzr QA bot
    path = get_files_in_directory()
    path = path[0]

    rag = QABot.pdf_qa(
        input_files=[str(path)],
        llm_params={"model": "gpt-3.5-turbo"},
        # vector_store_params=vector_store_params
    )

    return rag



def resume_response():
    rag = rag_implementation()
    prompt = """ Give the descrition of the given resume in 3 bullet points"""
    
    response = rag.query(prompt)
    return response.response

if uploaded_file is not None:
    automatice_response = resume_response()
    st.markdown(f"""{automatice_response}""")


    question = st.text_input("Ask a question about the resume:")
    
    if st.button("Get Answer"):
        rag = rag_implementation()
        response = rag.query(question)
        st.markdown(f"""{response.response}""")


# Footer or any additional information
with st.expander("ℹ️ - About this App"):
    st.markdown(
        """
    This app uses Lyzr QABot agent to implement the RAG functionality, where users can upload their resumes and ask questions. For any inquiries or issues, please contact Lyzr.

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
