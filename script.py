import os
import streamlit as st
import pandas as pd
import warnings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_experimental.agents import create_pandas_dataframe_agent

from dotenv import load_dotenv
import os

# Load the hidden .env file
load_dotenv()

# Get the key
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Global SSL and Proxy Bypass
# This tells ALL python libraries to ignore SSL errors and the KPIT proxy
os.environ["no_proxy"] = "*"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Data Agent", layout="wide")
st.title("Data Analyst")

# INITIALIZE NVIDIA LLM
try:
    llm = ChatNVIDIA(
        model="meta/llama-3.1-70b-instruct",
        api_key=NVIDIA_API_KEY,
        temperature=0.1
    )
except Exception as e:
    st.error(f"Initialization Error: {e}")

# FILE UPLOAD
uploaded_file = st.file_uploader("Upload your Project CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head(3))

    # INSTRUCTIONS FOR THE AGENT
    universal_prefix = """
    You are an expert data analyst working with a pandas dataframe named 'df'.
    Follow these rules strictly:
    1. Use the 'python_repl_ast' tool to perform any calculations or data retrieval.
    2. Once you have the result from the tool, STOP immediately.
    3. Your very next step MUST be to provide a 'Final Answer:' followed by the clear, concise result.
    4. Do not double-check or repeat your steps.
    5. If an error occurs in your code, fix it once; if it fails again, tell the user you couldn't find the answer.

    You have access to the following dataframe:
    {df_head}
    """

    # CREATE THE PERMANENT AGENT
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        allow_dangerous_code=True,
        agent_type="zero-shot-react-description",
        handle_parsing_errors=True,
        max_iterations=5,
        prefix=universal_prefix.format(df_head=df.head(3).to_string()) # Feeds data structure to LLM
    )

    # PANDAS QUERY
    user_question = st.text_input("Ask a question about this CSV:")
    
    if user_question:
        with st.spinner("NVIDIA NIM is processing..."):
            try:
                # Use invoke
                response = agent.invoke({"input": user_question})
                st.subheader("Final Answer:")
                st.success(response["output"])
            except Exception as e:
                # Specific error handling for the KPIT network
                if "SSL" in str(e) or "EOF" in str(e):
                    st.error("Network Security Block: The KPIT Firewall is still blocking the API.")
                    st.info("Try this: Disconnect from VPN, run the query, then reconnect.")
                else:
                    st.error(f"Analysis Error: {e}")
else:
    st.info("Awaiting CSV upload...")
