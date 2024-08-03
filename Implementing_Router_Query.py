import os
import streamlit as st
from dotenv import load_dotenv
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
import tempfile

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Apply nest_asyncio
nest_asyncio.apply()

# Streamlit app
st.markdown("<h1 style='text-align: center; color: Black;'>InsightPDF</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_and_process_data(file_path):
    # Load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    
    # Split documents into nodes
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Set up LLM and embedding models
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    Settings.embedding = OpenAIEmbedding(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)
    
    # Create indices
    summary_index = SummaryIndex(nodes=nodes)
    vector_index = VectorStoreIndex(nodes=nodes)
    
    # Set up query engines
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    vector_query_engine = vector_index.as_query_engine()
    
    # Create tools
    summary_tool = QueryEngineTool(
        query_engine=summary_query_engine,
        metadata=ToolMetadata(
            description="Useful for summarizing the uploaded PDF."
        )
    )
    vector_tool = QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            description="Useful for finding similar sentences in the uploaded PDF."
        )
    )
    
    # Set up router query engine
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[summary_tool, vector_tool],
        verbose=True,
    )
    
    return query_engine

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load and process data
    query_engine = load_and_process_data(tmp_file_path)

    # User input
    user_query = st.text_input("Enter your question about the uploaded PDF:", "What is this document about?")

    if st.button("Submit Query"):
        with st.spinner("Processing your query..."):
            response = query_engine.query(user_query)
            st.write("Response:")
            st.markdown(f"""
            <div style="background-color: #f0f2f6; border-radius: 10px; padding: 20px; margin: 10px 0;">
                {str(response)}
            </div>
            """, unsafe_allow_html=True)

    # Clean up the temporary file
    os.unlink(tmp_file_path)

else:
    st.info("Please upload a PDF file to begin.")

# Display general information about the app
st.sidebar.title("About PDF Query App")
st.sidebar.info(
    "This application allows you to query information from any uploaded PDF document. "
    "You can ask questions about its content, and the AI will analyze the document to provide answers."
)

# Add a brief explanation of how to use the app
st.sidebar.title("How to Use")
st.sidebar.info(
    "1. Upload a PDF file using the file uploader.\n"
    "2. Enter your question in the text box.\n"
    "3. Click 'Submit Query' to get a response.\n"
    "4. The app will use AI to analyze the PDF and provide an answer."
)