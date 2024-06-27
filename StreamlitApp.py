import os
import streamlit as st
import logging
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
# Set up logging
logging.basicConfig(
    filename='logs/pdf_chat.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('script_logger')

# Set API key for OpenAI
os.environ["OPENAI_API_KEY"] = "sk-proj-yzo3nGH4mg2Iit1H2O5RT3BlbkFJ3VMBoPY0q90GqWYY6M2Z"

# Streamlit app starts here
st.title("PDF Document Query App")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file is not None:
    pdfreader = PdfReader(uploaded_file)

    # Extract text from PDF
    raw_text = ''
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)

    query = st.text_input("Enter your question:")
    if query:
        docs = document_search.similarity_search(query)
        response = load_qa_chain(OpenAI(), chain_type="stuff").run(input_documents=docs, question=query)
        st.write("Answer:", response)