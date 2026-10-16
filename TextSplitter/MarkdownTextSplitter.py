import os
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from colorama import Fore
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# https://python.langchain.com/docs/modules/data_connection/vectorstores/

template: str = """/
    You are a senior developer who can answer {question} in natural language about markdown files  and {context} /
    respond in bullet points. /
    """
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["question", "context"],
    template="{question}",
)
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

model = OllamaLLM(model="tinyllama", temperature=0.2)

chunk_size = 60

# md loader
markdown_path = "./README.md"
data = TextLoader(markdown_path, encoding="utf-8").load()
markdown_text = ",".join([doc.page_content for doc in data])

def load_documents():
    md_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return md_splitter.create_documents([markdown_text])


documents = load_documents()
db = Chroma.from_documents(documents, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
retriever = db.as_retriever()

def print_chunks():
    _ = [print(f"{index + 1}/{len(documents)} - {document.page_content}\n") for index, document in enumerate(documents)]

    print(f"{Fore.GREEN}- Chunk Size: {chunk_size}")
    print(f"- Number of chunks: {len(documents)}")

 
def main(query):
    print_chunks()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_prompt_template
        | model
        | StrOutputParser()
    )
    response = chain.invoke(query)
    print(f"{Fore.CYAN}{response}")
   

if __name__ == "__main__":  
    query = "explain how to create a virtual environment for mac users"
    main(query)
