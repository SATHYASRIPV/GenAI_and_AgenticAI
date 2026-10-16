from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from colorama import Fore
import os   
import warnings

warnings.filterwarnings("ignore")

# LOAD ENV VARIABLES
load_dotenv()

# Load the model
print(Fore.YELLOW + "Initializing Hugging Face Chat Engine...")
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="conversational",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    temperature=0.2,
    max_new_tokens=250
)

# 2. Wrap it in ChatHuggingFace so it cleanly accepts ChatPromptTemplate
model = ChatHuggingFace(llm=llm)


# Load the documents = https://python.langchain.com/docs/get_started/introduction
print(Fore.YELLOW + "Scraping webpage documentation...")
loader = WebBaseLoader("https://python.langchain.com/docs/get_started/introduction")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
documents = text_splitter.split_documents(documents)
print(Fore.GREEN + f"✓ Successfully sliced text into {len(documents)} clean chunks.")

# prompt templates
template = (
    "you are senior developer who answers questions based on the given context information.\n\n" 
    "If you do not know the answer based on the context, say 'I do not find that in the documentation.'\n\n" 
    "Context:\n{context}")

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("user", "{input}")
])


# RETRIEVER - Load embeddings and create a vector store 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={'normalize_embeddings': True}
)
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# GENERATE - Define the function to generate the response
def generate(query): 
    combine_docs_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    # Execute the unified input syntax sequence
    response = rag_chain.invoke({"input": query})
    return response["answer"]



def start():
    print("HF_TOKEN", os.getenv("HF_TOKEN"))
    # print("documents", documents)
    instructions = (
        "Type your question and press ENTER. Type 'x' to go back to the MAIN menu.\n"
    )
    print(Fore.BLUE + "\n\x1B[3m" + instructions + "\x1B[0m" + Fore.RESET)

    print("MENU")
    print("====")
    print("[1]- Ask a question")
    print("[2]- Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        ask()
    elif choice == "2":
        print("Goodbye!")
        exit()
    else:
        print("Invalid choice")
        start()


def ask():
    while True:
        user_input = input("Q: ")
        # Exit
        if user_input == "x":
            start()
        else:
            response = generate(user_input)

            print(Fore.BLUE + f"A: " + response + Fore.RESET)
            print(Fore.WHITE + "\n-------------------------------------------------")


if __name__ == "__main__":
    start()
