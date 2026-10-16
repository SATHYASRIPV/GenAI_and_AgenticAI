## Project : Text splitters with LangChain

....


- **OpenAI** : [OpenAI](https://python.langchain.com/docs/integrations/platforms/openai) is a Python library that provides a simple interface to the OpenAI API. It also provides a command-line interface (CLI) for interacting with the API.

- **python-dotenv** : [python-dotenv](https://pypi.org/project/python-dotenv/) is a Python library that loads environment variables from a .env file. It is used to load the OpenAI API key from the .env file.

- **colorama** : [python-dotenv](https://pypi.org/project/colorama/) is a Python library to produce colored terminal text and outputs

- **BeautifulSoup** : [bs4](https://python.langchain.com/v0.2/docs/integrations/document_transformers/beautiful_soup/) Beautiful Soup is a Python package for parsing HTML and XML documents

- **ChromaDB** : [chromaDB](https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/) : Chroma is a AI-native open-source vector database focused on developer productivity



## 🛠️ Requirements : Installation & Setup

### Python 3.10.0

`brew install pyenv`
`pyenv install 3.10.0`

switch to python 3.10.0

`pyenv local 3.10.0`

### packages

- **LangChain** :[LangChain](https://www.langchain.com/) is a Python library that translates text to and from any language. It uses the Google Translate API to translate text. It also uses the Google Cloud Text-to-Speech API to convert text to speech.
- **Chroma** : [Chroma](https://www.trychroma.com/) is a Python library that converts text to speech. It uses the Google Cloud Text-to-Speech API to convert text to speech.
- **OpenAI** : [OpenAI](https://python.langchain.com/docs/integrations/platforms/openai) is a Python library that provides a simple interface to the OpenAI API. It also provides a command-line interface (CLI) for interacting with the API.
- **python-dotenv** : [python-dotenv](https://pypi.org/project/python-dotenv/) is a Python library that loads environment variables from a .env file. It is used to load the OpenAI API key from the .env file.


## 🌐 Create a virtual environment & activate the virtual environment :

**MacOS/Linux**:

```
python3 -m venv env
source env/bin/activate

```

**Windows**:

```
python -m venv env
source env/bin/activate
```

## 🏗️ Installation:

### Install Python 3.6 or higher

use pip3 on a Mac or Linux and pip on Windows

```
pip install -r requirements.txt
pip install --upgrade langchain
```

## [Get an API key](https://platform.openai.com/account/api-keys)

### Set the key as an environment variable:

`export OPENAI_API_KEY='sk-brHeh...A39v5iXsM2'`

.env file:

```
OPENAI_API_KEY=sk-brHeh...A39v5iXsM2
```

## ▶️ Start the app:

`python main.py`

## ▶️ start streamlit app on localhost:8501:

`streamlit run app.py`
