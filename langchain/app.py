import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DOC = "medical_book.pdf"
PERSIST_DIR = "vectorStore"
UPLOAD_DIR = "uploadedVectorStore"
URL_DIR = "urlVectorStore"


os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(URL_DIR, exist_ok=True)


groq_api_key = os.environ.get('GROQ_API_KEY')
if not groq_api_key:
    st.error("API Key not found! Please set the GROQ_API_KEY environment variable.")
    st.stop()

st.set_page_config(page_title="Health Chatbot", layout="wide")
st.title("Health Medical Chatbot")

if "docs" not in st.session_state:
    st.session_state.docs = None
if "final_documents" not in st.session_state:
    st.session_state.final_documents = None
if "text_splitter" not in st.session_state:
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=60)
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Options")
    choice = st.radio("Choose an option:", ["Insert Document", "Insert URL", "Set Default Document"])

def initialize_vectorstore(documents=None, directory=PERSIST_DIR):
    if documents:
        vectors = st.session_state.embeddings.embed_documents([doc.page_content for doc in documents])
        vectorstore = Chroma.from_documents(documents, st.session_state.embeddings, persist_directory=directory)
        vectorstore.persist()
        return vectorstore
    else:
        return Chroma(persist_directory=directory, embedding_function=st.session_state.embeddings)

def load_or_initialize_vectorstore(directory=PERSIST_DIR, documents=None):
    if os.path.exists(os.path.join(directory, "index")):
        return Chroma(persist_directory=directory, embedding_function=st.session_state.embeddings)
    else:
        return initialize_vectorstore(documents, directory)


if choice == "Insert Document":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_file:
        try:
            
            st.session_state.chat_history = []

            temp_file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            st.session_state.loader = PyPDFLoader(temp_file_path)
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vectorstore = load_or_initialize_vectorstore(UPLOAD_DIR, st.session_state.final_documents)
            st.sidebar.success("Document uploaded and vectors stored successfully.")
        except Exception as e:
            st.sidebar.error(f"Error processing document: {e}")

elif choice == "Insert URL":
    url = st.sidebar.text_input("Enter the URL")
    if url:
        try:
            
            st.session_state.chat_history = []

            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            loader = WebBaseLoader(web_paths=(url,))
            st.session_state.docs = loader.load()
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vectorstore = load_or_initialize_vectorstore(URL_DIR, st.session_state.final_documents)
            st.sidebar.success("URL content processed and vectors stored successfully.")
        except Exception as e:
            st.sidebar.error(f"Error processing URL: {e}")

else:
    try:
        
        st.session_state.chat_history = []

        st.session_state.loader = PyPDFLoader(DEFAULT_DOC)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectorstore = load_or_initialize_vectorstore(PERSIST_DIR, st.session_state.final_documents)
        st.sidebar.success("Default document loaded and vectors stored successfully.")
    except Exception as e:
        st.sidebar.error(f"Error loading default document: {e}")

if st.session_state.vectorstore is not None:
    try:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        prompt_template = ChatPromptTemplate.from_template(""" 
        You are an expert assistant specializing in health-related questions. Use the context below to answer questions:
        1. Answer **only** based on the provided context.
        2. Do not make up answers or provide unrelated information.
        <context>{context}</context>
        Question: {input}
        """)
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = st.session_state.vectorstore.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)

       
        st.write("### Chat Area")

       
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.write(f"🤔 **You:** {chat['question']}")
                st.write(f"🤖 **Assistant:** {chat['answer']}")
                st.write("---")

       
        user_query = st.text_input("Ask your question here", key="user_input", placeholder="Type your question...")
        if user_query:
            response = retriever_chain.invoke({"input": user_query})
            if 'answer' in response and response['answer']:
                answer = response['answer']
            else:
                answer = "No relevant information found."

            st.session_state.chat_history.append({"question": user_query, "answer": answer})
            st.write(f"🤔 **You:** {user_query}")
            st.write(f"🤖 **Assistant:** {answer}")
            st.write("---")

            with st.expander("Document Similarity Search"):
                if "context" in response:
                    for doc in response["context"]:
                        st.write(doc.page_content)
                        st.write("--------------------------------")
    except Exception as e:
        st.error(f"Error in chat system: {e}")
else:
    st.error("Please upload a document or set a default document before asking a question.")
