# Resume_QnA

## Purpose

This Streamlit app serves as a demonstration of a Question and Answer (QA) bot for resumes, implemented using the [Lyzr SDK](https://www.lyzr.ai/). The app allows users to upload a PDF resume, ask questions related to the resume, and receive automated responses generated by the Lyzr QA bot which uses the RAG functionality.

## Features

- **Resume Upload:** Users can upload a PDF file containing a resume.
- **Automated Responses:** The app automatically generates responses for a predefined prompt about the resume, displaying key information like personal details, technical skills, and areas of interest.
- **User Queries:** Users can ask specific questions about the resume, and the Lyzr QA bot provides responses based on the input.

## Getting Started

Follow these steps to run the Lyzr CV QA-Bot Streamlit App locally:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

2. **Run the App:**
    ```bash
    streamlit run app.py

3. **Provide API Key:**
Enter your OpenAI API key in the sidebar to enable Lyzr SDK functionality.

4. **Upload Resume:**
Upload a PDF file containing a resume using the file upload widget.

5. **Get Responses:**

- **User Queries:** Enter specific questions about the resume, and click the "Get Answer" button to receive responses.

## Resume QABot App
[Application](https://lyzr-resume-qna.streamlit.app/)

## About Lyzr
Lyzr is the simplest agent framework to help customers build and launch Generative AI apps faster. It is a low-code agent framework that follows an **‘agentic’** way to build LLM apps, contrary to Langchain’s ‘functions and chains’ way and DSPy’s ‘programmatic’ way of building LLM apps. For more information, visit [Lyzr website](https://www.lyzr.ai/) .
