import os
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from unstructured_client.models import shared
from unstructured.documents.elements import CompositeElement, Table
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from pdfminer.high_level import extract_text
from langchain.chat_models import ChatOpenAI
from unstructured_client import UnstructuredClient
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from utils import image_to_base64, load_images_from_directory, load_img_to_image_str   
from colorama import Fore
from PIL import Image
import base64
import io

load_dotenv()

filepaths = ["./docs/text-splitters.pdf"]
pdf_elements = []
documents = []
UNSTRUCTURED_API_KEY = os.environ.get("UNSTRUCTURED_API_KEY")
model = ChatOpenAI()
gpt4 = ChatOpenAI(model="gpt-4-turbo", max_tokens=1024)

#init client

def element_to_document(element):
    if isinstance(element, CompositeElement):
        content = element.text  # Assuming CompositeElement has a text attribute
    elif isinstance(element, Table):
        content = str(element.to_dict())  # Assuming Table has a to_list method to convert to a list
    else:
        content = str(element)  # Fallback to string representation for unknown types
    return Document(page_content=content)


# Extracts the elements from the PDF
def extract_elements_from_pdf(filepaths):
    """ extract elements from pdf with partition_pdf"""
    for filepath in filepaths:
        pdf_elements = partition_pdf(
            filename=filepath,
            strategy="hi_res", 
            model="yolox",
            infer_table_structure=True,
            extract_images_in_pdf=True, 
        )

    documents.append(pdf_elements)


def describe_images_from_pdf():
    """Describe images from PDFs."""
    images = load_img_to_image_str()
    if not images:
        print("No images!")
        return False

    for base64_image in images:
        msg = gpt4.invoke(
            [
                HumanMessage(
                   content=[
                    {"type": "text", "text": "Please give a short description of the image provided."}, 
                    {
                    "type": "image_url",
                    "image_url": {
                       "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                    },
                ],
                )
            ]
        )
        print(Fore.GREEN + f"Image description: {msg.content}")
        documents.append(Document(page_content=msg.content))
           
        

def process_pdf_documents():
    """Process the PDF documents and extract elements."""
    print("Processing PDF documents...")  
    extract_elements_from_pdf(filepaths)

    print("Extracting images from PDFs:")
    images = load_img_to_image_str()

    print("Describing images from PDFs:")
    describe_images_from_pdf()


    flattened_elements_list = [element for sublist in documents for element in sublist]
    docs = [element_to_document(element) for element in flattened_elements_list]
    _ = [print(doc.page_content) for doc in docs]
    print(f"number PDF documents processed. {len(docs)}")
    return docs



