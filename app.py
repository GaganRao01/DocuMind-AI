import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Any

# Load environment variables (including LangSmith settings)
load_dotenv()

# Initialize Gemini components
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# State definition
class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    thoughts: List[str]
    pdf_db: Any

# Process PDF and extract text
def process_pdf(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(pages)
    
    full_text = " ".join([page.page_content for page in pages])
    return FAISS.from_documents(texts, embeddings), full_text

# Extract questions using LLM
def extract_questions(text):
    prompt = (
        "Extract all questions from the following text. A question is a sentence that ends with a question mark "
        "or is phrased as an inquiry (e.g., starting with what, how, why, when, where, who, which). "
        "List each question on a new line.\n\n" + text
    )
    response = llm.invoke(prompt)
    questions = [q.strip() for q in response.content.split('\n') if q.strip()]
    return questions

# Define nodes
def retrieve_from_pdf(state: AgentState):
    docs = state["pdf_db"].similarity_search(state["question"], k=3)
    context = [doc.page_content for doc in docs]
    thoughts = state["thoughts"] + [f"Retrieved {len(docs)} relevant passages from the PDF."]
    return {"context": context, "thoughts": thoughts}

def web_search(state: AgentState):
    context = ["No relevant information found in PDF. Consider using web search."]
    thoughts = state["thoughts"] + ["PDF context insufficient, suggesting web search."]
    return {"context": context, "thoughts": thoughts}

def generate_answer(state: AgentState):
    context_str = "\n".join(state["context"])
    prompt = (
        f"Based on the following context:\n{context_str}\n\nQuestion: {state['question']}\n\n"
        "Please provide your thought process step by step, explaining how you arrive at the answer. "
        "If the context is insufficient, suggest searching the internet. "
        "After your thought process, write '---' followed by your final answer."
    )
    response = llm.invoke(prompt)
    full_response = response.content
    
    if '---' in full_response:
        llm_thoughts, answer = full_response.split('---', 1)
        llm_thoughts = llm_thoughts.strip()
        answer = answer.strip()
    else:
        llm_thoughts = "LLM did not provide thought process."
        answer = full_response
    
    thoughts = state["thoughts"] + ["LLM reasoning:\n" + llm_thoughts]
    return {"answer": answer, "thoughts": thoughts}

# Build LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("retrieve_pdf", retrieve_from_pdf)
workflow.add_node("web_search", web_search)
workflow.add_node("generate_answer", generate_answer)

# Define edges
workflow.add_edge(START, "retrieve_pdf")
workflow.add_conditional_edges(
    "retrieve_pdf",
    lambda state: "generate_answer" if state["context"] and len(state["context"]) > 0 else "web_search",
)
workflow.add_edge("web_search", "generate_answer")
workflow.add_edge("generate_answer", END)

# Compile workflow
app = workflow.compile()

# Streamlit UI
st.title("PDF QA Assistant")
st.caption("Upload a PDF and get automatic answers to questions found in the document")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_questions" not in st.session_state:
    st.session_state.processed_questions = False

uploaded_file = st.file_uploader("Choose PDF file", type="pdf")

if uploaded_file and not st.session_state.processed_questions:
    with st.spinner("Processing PDF and extracting questions..."):
        try:
            vector_db, full_text = process_pdf(uploaded_file)
            st.session_state.pdf_db = vector_db
            questions = extract_questions(full_text)
            st.session_state.questions = questions
            st.session_state.current_question_index = 0
            st.session_state.processed_questions = True
            
            if questions:
                st.success(f"Found {len(questions)} questions in the document")
            else:
                st.warning("No questions detected in the document")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "thoughts" in msg and msg["thoughts"]:
            with st.expander("Thought process"):
                st.markdown("\n\n".join(msg["thoughts"]))

# Process extracted questions
if "processed_questions" in st.session_state and st.session_state.processed_questions:
    if "questions" in st.session_state and st.session_state.questions:
        if st.session_state.current_question_index < len(st.session_state.questions):
            question = st.session_state.questions[st.session_state.current_question_index]
            
            if not any(msg["content"] == question and msg["role"] == "user" for msg in st.session_state.messages):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                
                with st.spinner("Thinking..."):
                    try:
                        result = app.invoke({
                            "question": question,
                            "context": [],
                            "answer": "",
                            "thoughts": [],
                            "pdf_db": st.session_state.pdf_db
                        })
                        with st.chat_message("assistant"):
                            st.markdown(result["answer"])
                            with st.expander("Thought process"):
                                st.markdown("\n\n".join(result["thoughts"]))
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "thoughts": result["thoughts"]
                        })
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Sorry, I encountered an error: {str(e)}",
                            "thoughts": ["Error occurred during processing"]
                        })
                
                st.session_state.current_question_index += 1
                if st.session_state.current_question_index < len(st.session_state.questions):
                    st.rerun()
            
            if st.session_state.current_question_index >= len(st.session_state.questions):
                st.success("All questions from the document have been answered!")

# Allow manual questions
if "pdf_db" in st.session_state:
    if prompt := st.chat_input("Ask additional questions about the PDF"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Thinking..."):
            try:
                result = app.invoke({
                    "question": prompt,
                    "context": [],
                    "answer": "",
                    "thoughts": [],
                    "pdf_db": st.session_state.pdf_db
                })
                with st.chat_message("assistant"):
                    st.markdown(result["answer"])
                    with st.expander("Thought process"):
                        st.markdown("\n\n".join(result["thoughts"]))
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "thoughts": result["thoughts"]
                })
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {str(e)}",
                    "thoughts": ["Error occurred during processing"]
                })

# Reset button
if st.session_state.get("processed_questions", False):
    if st.button("Reset and Process New PDF"):
        st.session_state.messages = []
        st.session_state.processed_questions = False
        st.session_state.questions = []
        st.session_state.current_question_index = 0
        if "pdf_db" in st.session_state:
            del st.session_state.pdf_db
        st.rerun()