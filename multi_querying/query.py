
import bs4
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.load import dumps, loads

from colorama import Fore
import warnings
warnings.filterwarnings("ignore")

load_dotenv()


# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm = OllamaLLM(model="tinyllama", temperature=0.2)

# RAG Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
multi_query_prompt = ChatPromptTemplate.from_template(template)


# GENERATION
rag_template = """Answer the following question based on this context:
{context}
Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

# INDEXING

# Load blog
loader = WebBaseLoader(
    web_paths=("http://blog.langchain.dev/deconstructing-rag/",),
    # bs_kwargs=dict(
    #     parse_only=bs4.SoupStrainer(
    #         class_=("article-header", "section", "article-header__content", "article-header__footer")
    #     )
    # ),
)
blog_docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

# Index
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

retriever = vectorstore.as_retriever()


# RETRIEVAL Multi-Query
def generate_multi_queries():
    """ generate multiple queries given a user input """
    return (
        multi_query_prompt
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

# GENERATION 
def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string_
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    
    # Return
    return [loads(doc) for doc in unique_docs]
    


# Query
def query(query):
    question = "how can we make retrieval robust to variability in user input?"
    # Retrieval Multi-query Chain
    retrieval_chain = generate_multi_queries() | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question": query})
    print(f"{Fore.CYAN}{len(docs)} documents retrieved {Fore.RESET}")


    # Generation Chain
    final_rag_chain = (
        {"context": retrieval_chain, "question": RunnablePassthrough()} 
        | rag_prompt 
        | llm 
        | StrOutputParser()
    )

    return final_rag_chain.invoke(query)

