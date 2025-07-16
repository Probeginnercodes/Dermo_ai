import os
import streamlit as st

# Set Hugging Face-compatible config
st.set_page_config(layout="wide")

def main():
    # Your existing app code here
    st.title("DermoAI")

if __name__ == "__main__":
    if os.environ.get('IS_HF_SPACE') == '1':  # Detect Hugging Face
        os.system(f"streamlit run {__file__} --server.port=$PORT --server.address=0.0.0.0")
    else:
        main()  # Local run