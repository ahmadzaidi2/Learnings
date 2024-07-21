
# Give example of all 3 questions in Linked in description
#Linked in post write that we will run this using llama3, running locally using ollama
# Write in linked in write up that how important it is to have correct placement of architecteral component of adaptib=ve RAG, this can then be simpley translated into working code using langgraph
# Commit to git

# Remove this section


######################################################################
# Imports
######################################################################
import os
from typing import Literal
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from pprint import pprint

######################################################################
# API
######################################################################
from dotenv import load_dotenv
load_dotenv()
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

######################################################################
# Models
######################################################################
# Set embeddings
embed = OpenAIEmbeddings()
#LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

######################################################################
# Simple RAG
######################################################################
# Docs to index
urls = [
    "https://cobusgreyling.medium.com/the-anatomy-of-chain-of-thought-prompting-cot-b7489c925402"
]
# Load
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embed,
)
retriever = vectorstore.as_retriever()
######################################################################
# Router
######################################################################
# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search", "Model"] = Field(
        ...,
        description="Given a user question choose to route it to web search , vectorstore or Model",
    )
structured_llm_router = llm.with_structured_output(RouteQuery)
# Prompt
system = """You are an expert at routing a user question to a vectorstore, web search or Model.
For queries related to Chain of Thoughts, prompt engineering strictly Use the vectorstore as your response. 
For topics were you do not have knowledge and performing websearch is essential use web-search,
If the query been asked to you is simple and you are confident that you can reply then use Model"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
print(
    question_router.invoke(
        {"question": "Who is player of the match of T20 world cup final?"}
    )
)
######################################################################
# Retrieval Grader
######################################################################
# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
######################################################################
# Simple LLM Response
######################################################################
system = """You are a helpful assistant. You take the user's query and reply with a helpful answer"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Query: \n\n {question}"),
    ]
)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
######################################################################
# Generate
######################################################################
# Prompt
prompt = hub.pull("rlm/rag-prompt")
# Post-processing
def format_docs(docs):
    """Format documents

    Args:
        docs (document): individual docs

    Returns:
        string: formatted docs
    """
    return "\n\n".join(doc.page_content for doc in docs)
# Chain
rag_chain = prompt | llm | StrOutputParser()
######################################################################
# Hallucination Grader
######################################################################
# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)
# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded and is supported by the set of facts. """
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
hallucination_grader = hallucination_prompt | structured_llm_grader
######################################################################
# Answer Grader
######################################################################
# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
structured_llm_grader = llm.with_structured_output(GradeAnswer)
# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)
answer_grader = answer_prompt | structured_llm_grader
######################################################################
# Question Re-writer
######################################################################
# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)
question_rewriter = re_write_prompt | llm | StrOutputParser()
######################################################################
# Web Search Tool
######################################################################
web_search_tool = TavilySearchResults(k=3)
######################################################################
# Graph State
######################################################################
class GraphState(TypedDict):
    """
    Represents the state of our graph.Each graph execution creates a state that is passed 
    between nodes in the graph as they execute, and each node updates this internal state 
    with its return value after it executes

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: str
    documents: List[str]
######################################################################
# Graph flow
######################################################################
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("Getting into Retrieve")
    question = state["question"]
    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def llm_fallback(state):
    """
    Generate answer in case of LLM fallback

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("Getting into LLm Fallback")
    question = state["question"]

    generation = chain.invoke({"question": question})
    print(generation)
    return {"question": question, "generation": generation}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("Getting into Generate")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("Checking Document relevance to question")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("Graded document is relevant")
            filtered_docs.append(d)
        else:
            print("Graded document is not relevant")
            continue
    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("Transforming the question been asked")
    question = state["question"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"question": better_question}

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("Performing web search")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}

### Edges ###

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("Routing the Question")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("Routing to web search")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("Routing for RAG")
        return "vectorstore"
    elif source.datasource == "Model":
        print("Routing for LLM Fallback")
        return "Model"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("Accessing Graded Document")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "All the retrieved documents are not relevant for generation, transform the question"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("Perform Generation")
        return "generate"
    
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("Check for hallucinations")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score
    # Check hallucination
    if grade == "yes":
        print("Generation is grounded in documents, No hallucination")
        # Check question-answering
        print("Getting in Generation check")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("Generation Address the question")
            return "useful"
        else:
            print("Generation do not address the question")
            return "not useful"
    else:
        print("Generation is not grounded in documents, retry generation")
        return "not supported"
    

######################################################################
# Graph building
######################################################################
workflow = StateGraph(GraphState)
# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("llm_fallback", llm_fallback)  # LLM_Fallback

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "Model": "llm_fallback",
    },
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)


# Add an edge from llm_fallback to END
workflow.add_edge("llm_fallback", END)
workflow.set_finish_point("llm_fallback")

# Compile
app = workflow.compile()
######################################################################
# Example 1: websearch
######################################################################
# Run
inputs = {
    "question": "Who was Player of the Match for Twenty 20 world cup final?"
}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
    pprint("\n---\n")
# Final generation
pprint(value["generation"])
######################################################################
# Example 2: RAG
######################################################################
# Run
inputs = {"question": "What is Chain of Thoughts?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
    pprint("\n---\n")
# Final generation
pprint(value["generation"])
######################################################################
# Example 3: LLM
######################################################################
# Run
inputs = {"question": "Hey!"}
app.invoke(inputs)


