# Project : CRAG - Corrective RAG as Graph

---
CRAG enhances the traditional RA pipeline by introducing a retrieval evaluator to assess the relationship between the retrieved documents and the query. 

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
pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain langgraph tavily-python streamlit
streamlit-chat
```

(MacOS)

```
pip3 install -r requirements.txt
pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain langgraph tavily-python
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

## Start Streamlit App:

`streamlit run app.py`