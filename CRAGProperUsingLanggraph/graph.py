from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from pprint import pprint
from typing import List
from colorama import Fore
from query import retriever, assess_retrieved_docs, rewrite_query, search_web, generate_answer
# https://github.com/langchain-ai/langgraph/tree/main?ref=blog.langchain.dev
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents 
    """
    question : str
    generation : str
    web_search : str
    documents : List[str]

workflow = StateGraph(GraphState)

# NODES
def retrieve_documents(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print(f"{Fore.MAGENTA}---RETRIEVE---{Fore.RESET}")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print(f"{Fore.CYAN}---ASSESS RELEVANCE TO QUESTION---{Fore.RESET}")
    question = state["question"]
    documents = state["documents"]
    
    # Score each doc
    filtered_documents = []
    web_search = "No"
    for d in documents:
        score = assess_retrieved_docs(question, documents)
        if score == "yes":
            filtered_documents.append(d)
            print(f"{Fore.GREEN}--- document is relevant ✅ ---{Fore.RESET}")
            filtered_documents.append(d)
        else:
            print(f"{Fore.RED}--- document is not relevant ❌ ---{Fore.RESET}")
            web_search = "Yes"
            continue
    return {"documents": filtered_documents, "question": question, "web_search": web_search}

def rewrite_query(state):
    """
    rewrite the query to produce an optimized question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    print(f"{Fore.BLUE}---REVISE QUERY---{Fore.RESET}")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    optimized_question = rewrite_query(question)
    return {"documents": documents, "question": optimized_question}
  
    
def web_search_node(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print(f"{Fore.MAGENTA}---WEB SEARCH---{Fore.RESET}")
    question = state["question"]
    documents = state["documents"]

    # Web search
    web_results = search_web(question)
    documents.extend(web_results)

    return {"documents": documents, "question": question}

def generation_answer(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print(f"{Fore.GREEN}---GENERATE---{Fore.RESET}")
    question = state["question"]
    documents = state["documents"]
    
    # RAG generation
    answer = generate_answer(documents, question)
    return {"documents": documents, "question": question, "generation": answer}


# EDGES
def decide_to_search_the_web(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # search the web
        print("---DECISION: SEARCH THE WEB ---")
        return "web_search_node"
        
    else:
        # rewrite query
        print("---DECISION: REWRITE QUERY---")
        return "rewrite_query"


def generate(state):
    return "generate"


# Define the nodes
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("generate", generation_answer)

# Build graph
workflow.set_entry_point("retrieve_documents")
workflow.add_edge("retrieve_documents", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents", 
    decide_to_search_the_web, {
        "web_search_node": "web_search_node",
        "rewrite_query": "rewrite_query"
    })

workflow.add_edge("rewrite_query", "generate")
workflow.add_edge("web_search_node", "generate")

# Compile
app = workflow.compile()


def query(query):
    # Run
    inputs = {"question": query}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node ''{key}''")
        pprint("\n---\n")
    # Final generation
    pprint(value)
    return value["generation"]
 

