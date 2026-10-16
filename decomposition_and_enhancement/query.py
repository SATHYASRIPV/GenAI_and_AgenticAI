
import bs4
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langsmith import Client
from operator import itemgetter
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from utils import format_qa_pair, format_qa_pairs

from colorama import Fore
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
# LLM

print(Fore.YELLOW + "Initializing Hugging Face Chat Engine...")
# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     task="conversational",
#     huggingfacehub_api_token=hf_token,
#     temperature=0.2,
#     max_new_tokens=250
# )

llm = OllamaLLM(model="tinyllama", temperature=0.2)

# model = ChatHuggingFace(llm=llm)

# load documents (data source: https://blog.langchain.dev/)
loader = WebBaseLoader("https://blog.langchain.dev/deconstructing-rag/")
docs = loader.load()

loader = WebBaseLoader("https://blog.langchain.dev/reflection-agents/")
docs.extend(loader.load())

#  split documents, create vector store and load embeddings
loader = WebBaseLoader(
    web_paths=("https://blog.langchain.dev/reflection-agents/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("article-header__section", "article-header__content", "article-header__footer")
        )
    ),
)
# loader = WebBaseLoader(web_paths=("https://www.langchain.com/blog",))
blog_docs = docs

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)
splits = text_splitter.split_documents(blog_docs)

# Add this safety check:
print(f"Successfully generated {len(splits)} text chunks for embedding.")
if not splits:
    raise ValueError("Splits array is empty. The web page failed to extract usable text content.")

# Index and load embeddings
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# Create the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 1. DECOMPOSITION
template = """You are a helpful assistant trained to generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)


def generate_sub_questions(query):
    """ generate sub questions based on user query"""
    # Chain
    generate_queries_decomposition = (
        prompt_decomposition 
        | llm 
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    ) 

    # Run
    sub_questions = generate_queries_decomposition.invoke({"question": query})
    questions_str = "\n".join(sub_questions)
    print(Fore.MAGENTA + "=====  SUBQUESTIONS: =====" + Fore.RESET)
    print(Fore.WHITE + questions_str + Fore.RESET + "\n")
    return sub_questions


# 2. ANSWER SUBQUESTIONS RECURSIVELY 
template = """You are a helpful assistant. Answer this question ONLY using the provided context.
 
QUESTION: {sub_question}
 
CONTEXT FROM DOCUMENTS:
{context}
 
INSTRUCTIONS:
- Answer ONLY based on the context above
- If the context doesn't contain the answer, say "Not found in documents"
- Be specific and cite the context
- Keep answer concise (2-3 sentences)
 
ANSWER:"""
prompt_qa = ChatPromptTemplate.from_template(template)


def generate_qa_pairs(sub_questions):
    """ ask the LLM to generate a pair of question and answer based on the original user query """
    q_a_pairs = ""
    q_a_pairs_list = []
    
    for sub_question in sub_questions:
        # chain
        # recent_qa = "\n".join(q_a_pairs.split("\n")[-4:]) if q_a_pairs else ""
        generate_qa = (
            {"context": itemgetter("sub_question") | retriever, "sub_question": itemgetter("sub_question"), "q_a_pairs": itemgetter("q_a_pairs")}
            | prompt_qa 
            | llm 
            | StrOutputParser()
        )
        answer = generate_qa.invoke({"sub_question": sub_question, "q_a_pairs": q_a_pairs})
        q_a_pair = format_qa_pair(sub_question, answer)
        q_a_pairs_list.append(q_a_pair)
        q_a_pairs = q_a_pairs + "\n --- \n" + q_a_pair
    return "\n".join(q_a_pairs_list)
    # print(Fore.BLUE + "=====  QUESTION/ANSWER PAIRS: =====" + Fore.RESET + q_a_pairs + "\n")

# 3. ANSWER INDIVIDUALLY

# RAG prompt = https://smith.langchain.com/hub/rlm/rag-prompt
prompt_rag = Client().pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)


def retrieve_and_rag(prompt_rag, sub_questions):
    """RAG on each sub-question"""
    rag_results = []
    for sub_question in sub_questions:
        retrieved_docs = retriever.invoke(sub_question)
    
        answer_chain = (
            prompt_rag
            | llm
            | StrOutputParser()
        )
        answer = answer_chain.invoke({"question": sub_question, "context": retrieved_docs})
        rag_results.append(answer)
    
    return rag_results, sub_questions



# SUMMARIZE AND ANSWER 

# Prompt
template = """Here is a set of Q+A pairs:

{context}

to use these to synthesize an answer to the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


# Query
def query(query):
    # generate optimized answer for a given query using the improved subqueries
    question = "What are the main components of an LLM-powered autonomous agent system?"
    queries = [
        "How is context improving AI systems",
        "What are the two main components involved in Basic Reflection",
        "Explain the steps involved in the Reflexion loop"
    ]
    sub_questions = generate_sub_questions(query)
    q_a_pairs = generate_qa_pairs(sub_questions)
    # answers, questions = retrieve_and_rag(prompt_rag, sub_questions)
    # context = format_qa_pairs(questions, answers)
    final_rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    return final_rag_chain.invoke({"question": question, "context": q_a_pairs})
