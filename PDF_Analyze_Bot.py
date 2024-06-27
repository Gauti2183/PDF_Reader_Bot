
import os
import logging
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

## Add logging of activitues
logging.basicConfig(
    filename='logs/telegram_script_surveillance.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('script_logger')
print(logger)
os.environ["OPENAI_API_KEY"] = "sk-proj-yzo3nGH4mg2Iit1H2O5RT3BlbkFJ3VMBoPY0q90GqWYY6M2Z"

## UDF ------------------------------------------------------------------------------------------------------
chathistory_path = "/home/surveillance/Documents/SSKLAI-AGENTS/SSKLGPT/ChatHistory/ChatHistory_surveillance.json"


if __name__ == '__main__':

    file_path = (
        "/home/surveillance/Documents/LangChainBot/pdf_files/gs4dhdaa2a9f352b0445bafbc79ca799dce4d.pdf"
    )
    pdfreader = PdfReader(file_path)

    from typing_extensions import Concatenate

    # read text from pdf
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    # print(raw_text)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # print(len(texts))

    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)
    print(document_search)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    query = "What is the subject of this document?"
    docs = document_search.similarity_search(query)
    x=chain.run(input_documents=docs, question=query)
    print(x)