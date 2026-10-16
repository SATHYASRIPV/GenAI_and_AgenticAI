import os
import sys
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document

from unstructured_client import UnstructuredClient
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from langchain_core.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from colorama import Fore
from PIL import Image
import pytesseract
import base64
import io

load_dotenv()

TESSERACT_PATH = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
try:
    version = pytesseract.get_tesseract_version()
    print(f"✓ Tesseract configured: {version}")
except pytesseract.TesseractNotFoundError as e:
    print(f"❌ Tesseract configuration failed: {e}")
    sys.exit(1)

from unstructured.documents.elements import CompositeElement, Table

pytesseract.pytesseract.pytesseract_cmd  = TESSERACT_PATH
filepaths = ["./docs/text-splitters.pdf"]
pdf_elements = []
documents = []
UNSTRUCTURED_API_KEY = os.environ.get("UNSTRUCTURED_API_KEY")
client = UnstructuredClient(api_key=UNSTRUCTURED_API_KEY)
gpt4 = OllamaLLM(model="tinyllama", max_tokens=1024)

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
        pdf_elements = client.general.partition(
            filename=filepath,
            strategy="layout", 
            model="yolox",
            infer_table_structure=True,
        )

    documents.append(pdf_elements)


def process_pdf_documents():
    """Process the PDF documents and extract elements."""
    print("Processing PDF documents...")  
    extract_elements_from_pdf(filepaths)

    flattened_elements_list = [element for sublist in documents for element in sublist]
    docs = [element_to_document(element) for element in flattened_elements_list]
    _ = [print(doc.page_content) for doc in docs]
    return docs



