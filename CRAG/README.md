# Project : CRAG - Corrective RAG

---
CRAG enhances the traditional RA pipeline by introducing a retrieval evaluator to assess the relationship between the retrieved documents and the query. 


- **OpenAI** : [OpenAI](https://python.langchain.com/docs/integrations/platforms/openai) is a Python library that provides a simple interface to the OpenAI API. It also provides a command-line interface (CLI) for interacting with the API.

- **python-dotenv** : [python-dotenv](https://pypi.org/project/python-dotenv/) is a Python library that loads environment variables from a .env file. It is used to load the OpenAI API key from the .env file.

- **colorama** : [python-dotenv](https://pypi.org/project/colorama/) is a Python library to produce colored terminal text and outputs

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
pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain langgraph tavily-python
```

(MacOS)

```
pip3 install -r requirements.txt
pip install langchain chromadb sentence-transformers
```

## [OpenAI - Get an API key](https://platform.openai.com/account/api-keys)

.env file

OPENAI_API_KEY=sk-brHeh...A39v5iXsM2

`export OPENAI_API_KEY='sk-brHeh...A39v5iXsM2'`

## [Tavily Search API](https://app.tavily.com/documentation/apis)
Tavily's Search API is a search engine built specifically for AI agents (LLMs), delivering real-time, accurate, and factual results at speed.

- [LangChain documentation](https://js.langchain.com/v0.1/docs/integrations/retrievers/tavily/)

## Run the script:

(Windows)
```
python main.py
```

(MacOS)
```
python3 main.py
```
