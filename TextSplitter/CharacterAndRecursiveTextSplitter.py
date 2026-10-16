import os
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from colorama import Fore
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

template: str = "You are a helpful assistant who answers {question} based on {context}"

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["question", "context"],
    template="{question}",
)
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

model = OllamaLLM(model="tinyllama", temperature=0.2)

chunk_size = 200

state_of_the_union = TextLoader("./docs/state_of_the_union.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=0
)

recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=0,
)

def recursive_load_documents():
    return recursive_text_splitter.split_documents(state_of_the_union)


def load_documents():
    return text_splitter.split_documents(state_of_the_union)


# documents = load_documents()
documents = recursive_load_documents()
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
    # query = "What did the president say about Ketanji Brown Jackson"
    query = "What happened to Heath"
    main(query)
