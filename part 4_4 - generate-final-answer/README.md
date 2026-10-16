# Project : Advanced RAG - Query Transformation: Decomposition (ChromaDB vector store)

---
Query decomposition is a strategy to improve question-answering using the power of LLM by re-writing and rephrasing the user input to improve retrieval process by breaking down a question into well-written sub-questions.


- **OpenAI** : [OpenAI](https://python.langchain.com/docs/integrations/platforms/openai) is a Python library that provides a simple interface to the OpenAI API. It also provides a command-line interface (CLI) for interacting with the API.

- **python-dotenv** : [python-dotenv](https://pypi.org/project/python-dotenv/) is a Python library that loads environment variables from a .env file. It is used to load the OpenAI API key from the .env file.

- **colorama** : [python-dotenv](https://pypi.org/project/colorama/) is a Python library to produce colored terminal text and outputs

- **BeautifulSoup** : [bs4](https://python.langchain.com/v0.2/docs/integrations/document_transformers/beautiful_soup/) Beautiful Soup is a Python package for parsing HTML and XML documents

- **ChromaDB** : [chromaDB](https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/) : Chroma is a AI-native open-source vector database focused on developer productivity


## **Install Python** 

- A [Quick Guide for Installing](https://github.com/PackeTsar/Install-Python/blob/master/README.md#install-python-) Python on Common Operating Systems
- Download the latest version of [Python 3.12](https://www.python.org/downloads/)

## Create a virtual environment :

(Windows)
```
python -m venv env
```

(MacOS)
```
python3 -m venv env
```

## Activate the virtual environment :

```
source env/bin/activate
```

## Installation:
(Windows)
```
pip install -r requirements.txt
pip install langchainhub
```

(MacOS)

```
pip3 install -r requirements.txt
```

## [get an API key](https://platform.openai.com/account/api-keys)

.env file

OPENAI_API_KEY=sk-brHeh...A39v5iXsM2

`export OPENAI_API_KEY='sk-brHeh...A39v5iXsM2'`


## Run the script:

(Windows)
```
python main.py
```

(MacOS)
```
python3 main.py
```
