import os, streamlit as st

from llama_index.core import GPTVectorStoreIndex, SimpleDirectoryReader, PromptHelper
# from langchain_community.llms import OpenAI
# from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint  # Updated import for Hugging Face integration

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

os.environ['API_KEY']= "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"

# Provide openai key from the frontend if you are not using the above line of code to seet the key
# openai_api_key = st.sidebar.text_input(
#     label="#### Your OpenAI API key 👇",
#     placeholder="Paste your openAI API key, sk-",
#     type="password")

directory_path = st.sidebar.text_input(
    label="#### Your data directory path 👇",
    placeholder="C:\data",
    type="default")

def get_response(query,directory_path,api_key):
    
    # llm_predictor = OpenAI(openai_api_key=openai_api_key, temperature=0, model_name="text-davinci-003")
    llm_predictor = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-base",
        huggingfacehub_api_token=api_key
    )

    # Configure prompt parameters and initialise helper
    max_input_size = 4096
    num_output = 256
    max_chunk_overlap = 1

    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    if os.path.isdir(directory_path): 
        # Load documents from the 'data' directory
        documents = SimpleDirectoryReader(directory_path).load_data()
        # service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        # settings = Settings(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        
        response = index.query(query)
        if response is None:
            st.error("Oops! No result found")
        else:
            st.success(response)
    else:
        st.error(f"Not a valid directory: {directory_path}")

# Define a simple Streamlit app
st.title("ChatMATE")
query = st.text_input("What would you like to ask?", "")

# If the 'Submit' button is clicked
if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            if len(os.environ['API_KEY']) > 0:
                get_response(query,directory_path,os.environ['API_KEY'])
            else:
                st.error(f"Enter a valid api key")
        except Exception as e:
            st.error(f"An error occurred: {e}")