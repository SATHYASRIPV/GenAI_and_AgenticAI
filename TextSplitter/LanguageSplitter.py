from dotenv import load_dotenv
from operator import itemgetter
# from langchain_community.chains import RetrievalQA

from langchain_ollama import OllamaLLM
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter, 
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from colorama import Fore
import os
import warnings 

warnings.filterwarnings("ignore")

load_dotenv()

llm = OllamaLLM(model="tinyllama", temperature=0.2)

loader = WebBaseLoader(
    "https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/code_splitter/"
)
documents = loader.load()

# prompt templates
retriever_template = "Extract code snippet from the web page based on the {programming_language} language and the {context} of the documents retrieved. Do not include other languages"

rag_template = """You are a senior developer who can answer {question} in natural language based on {context} /
generate a detailed answer to explain in bullet points (steps) and return the formatted answer /
add any imports statements or any other requirements if necessary 
"""

retriever_prompt = PromptTemplate(
    template=retriever_template,
    input_variables=["context", "programming_language"])

rag_prompt = PromptTemplate(
    template=rag_template,
    input_variables=["context", "question"])

def split_code(code_snippet):
    python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0)
    return python_splitter.create_documents([code_snippet])


# INDEXING - Load embeddings and create a vector store 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 1})


# RETRIEVE code snippet from web page
def get_code_snippets(language):
    """ get the code snippet given a language """
    # chain
    retriever_chain = (
        {"context": retriever, "programming_language": RunnablePassthrough()}
        | retriever_prompt
        | llm
        | StrOutputParser()
    )
    code_snippets = retriever_chain.invoke(language)
    print(f"{Fore.BLUE}=== CODE SNIPPET:===\n")
    print(f"{code_snippets}{Fore.RESET}")
    return code_snippets


def generate(query: str, vector_store):
    """ generate the response given a query and language """
    rag_chain = (
        {
        "context": vector_store,
        "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(query)

def query(query=""):
    query = "How to split code in Python?"
    code_snippet = get_code_snippets("Python")  
    docs = split_code(code_snippet)
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    return generate(query, retriever)

