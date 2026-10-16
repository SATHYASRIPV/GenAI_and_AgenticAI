import bs4
from dotenv import load_dotenv
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
from langchain_huggingface import HuggingFaceEmbeddings

from colorama import Fore
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

llm = OllamaLLM(model="tinyllama", temperature=0.2)

# Multi-Query Generation & Re-ranking
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)


# GENERATION
promtp_template = """Answer the following question based on this context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template( promtp_template)

#### INDEXING ####

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


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

# Index
vectorstore = Chroma.from_documents(documents=splits, 
                                    # embedding=CohereEmbeddings()
                                    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))


retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####
generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)


# https://gist.github.com/srcecde/eec6c5dda268f9a58473e1c14735c7bb
def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse = True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results


# Query
def query(query):
 
    question = "how can we make retrieval robust to variability in user input?"
    # RAG Chain

    retrieval_chain_rag_fusion = (
        generate_queries 
        | retriever.map() 
        | reciprocal_rank_fusion
    )

    docs = retrieval_chain_rag_fusion.invoke({"question": query})
    print(f"{Fore.CYAN}{len(docs)} documents retrieved {Fore.RESET}")
    
    # GENERATION Chain
    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion, "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )

    return final_rag_chain.invoke({"question": query})

