
import bs4
from dotenv import load_dotenv
from langsmith import Client
# from langchain_core.tools import Client
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from colorama import Fore
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

llm = ChatOllama(model="tinyllama", temperature=0)

#### INDEXING ####
urls = [
    "http://blog.langchain.dev/deconstructing-rag",
    "https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/",
    "https://blog.langchain.dev/graph-based-metadata-filtering-for-improving-vector-search-in-rag-applications/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
)
retriever = vectorstore.as_retriever()


#### Retrieval Grader : Retrieval Evaluator ####
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

    def get_score(self) -> str:
        """Return the binary score as a string."""
        return self.binary_score


def get_score(self) -> str:
    """Return the binary score as a string."""
    return self.binary_score

# LLM with function call 
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt 
system_template = """You are an evaluator determining the relevance of a retrieved {documents} to a user's query {question}.If the document contains keyword(s) or semantic meaning related to the question, mark it as relevant.Assign a binary score of 'yes' or 'no' to indicate the document's relevance to the question."""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["documents", "question"],
    template="{question}",
)
grader_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

### Question Re-writer - Knowledge Refinement ####
# Prompt 
prompt_template = """Given a user input {question}, your task is re-write or rephrase the question to optimize the query in order to imprive the content generation"""

system_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
human_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["question"],
    template="{question}",
)
re_write_prompt = ChatPromptTemplate.from_messages(
    [system_prompt, human_prompt]
)

### Web Search Tool - Knowledge Searching ####
web_search_tool = TavilySearchResults(k=3) 

#### Generate Answer  ####
# Prompt
prompt = Client().pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)

# Retrieve and assess
def assess_retrieved_docs(query, documents=[]):
    """Retrieve and assess the relevance of documents to a given query."""
    retrieval_grader = grader_prompt | structured_llm_grader | get_score
    docs = retriever.invoke(query) 
    doc_txt = docs[0].page_content
    binary_score = retrieval_grader.invoke({"question": query, "documents": doc_txt})
    return binary_score, docs

# Rewrite and optimize 
def rewrite_query(query):
    """Rewrite and optimize a given user query for the model."""
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter.invoke({"question": query})

# Search the web
def search_web(query):
    """Search the web for complimentary information."""
    docs = web_search_tool.invoke({"query": query})
    print(docs)
    web_docs = "\n".join([d["content"] for d in docs])
    return Document(page_content=web_docs)

def generate_answer(docs, query):
    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    # Run
    return rag_chain.invoke({"context": docs, "question": query})

def query(query):
    """Query the model with a question and assess the relevance of retrieved documents."""
    # question = "RAG"
    binary_score, docs = assess_retrieved_docs(query)
    if binary_score == "no":
        print(f"{Fore.MAGENTA}Retrieved documents are irrelevant. Searching the web for additional information.")
        docs = search_web(query)
    # Rewrite and optimize the query
    print(f"{Fore.YELLOW}Rewriting the query for content generation.")
    optimized_query = rewrite_query(query)
    print(f"Relevance score: {binary_score}")
    return generate_answer(docs, optimized_query)