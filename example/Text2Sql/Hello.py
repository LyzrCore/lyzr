import os
from PIL import Image
import streamlit as st
from lyzr import FormulaGen

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

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
st.title("QueryGenius (Built using Lyzr SDK)")
st.markdown("### Welcome to QueryGenius!⌨️")
st.markdown("Transform your descriptions into SQL queries effortlessly and efficiently.")
# Define FormulaGen instance
generate = FormulaGen()

# Define the Streamlit app layout
def app():

    # Text input for user to input the text
    text_input = st.text_area("Enter your texts:")

    # Button to trigger the conversion
    if st.button("Convert to SQL"):
        # Text to SQL conversion
        result = generate.text_to_sql(text_input)

        # Display the result
        st.subheader("Generated SQL Query:")
        st.code(result, language="sql")


# Run the app
if __name__ == "__main__":
    app()

# Footer or any additional information
with st.expander("ℹ️ - About this App"):
    st.markdown("""
   QueryGenius is an intuitive web application that effortlessly converts natural language descriptions into SQL queries, streamlining the querying process and eliminating manual construction.
    """)
    
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack", url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw', use_container_width=True)
