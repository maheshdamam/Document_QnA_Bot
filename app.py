from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
import streamlit as st 
from time import sleep

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)


if "vector_db" not in st.session_state:
    st.session_state.vector_db = None 

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

def document_process(path):
    # Document Loading
    loader = PyPDFLoader(path)
    docs = loader.load()

    # Splitting 
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)

    # Embeddings and vector store 
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vector_db =  InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True


# Document Upload 
st.subheader("📃 Document QnA Chatbot - Ask Anything")

# FIX 1: Wrap upload in a clean interface that clears out properly
if not st.session_state.document_uploaded:
    file = st.file_uploader(label="Select Your PDF File", type="pdf")
    if file:
        with open("uploaded_document.pdf", "wb") as f:
            f.write(file.getvalue())
        
        with st.spinner("Processing..."):
            document_process("./uploaded_document.pdf")

        st.success("Document Processed Successfully.")
        sleep(1.5)
        st.rerun()

# FIX 2: Only show chat features if the document data state exists
if st.session_state.document_uploaded and st.session_state.vector_db:
    
    # Render historical chat records FIRST on top-down reruns
    for oneMessage in st.session_state.messages:
        with st.chat_message(oneMessage["role"]):
            st.markdown(oneMessage["content"])

    # Await chat input from user
    query = st.chat_input("Ask Anything...")
    
    if query:
        # Display user message instantly
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
    
        # Perform retrieval and invocation
        with st.spinner("Thinking..."):
            documents = st.session_state.vector_db.similarity_search(query, k=2)
            context = ""
            for doc in documents:
                context += doc.page_content + "\n\n"

            prompt = f"You are a helpful assistant and you provide answers for user questions based on the provided context. context: {context} and question is: {query}" 
            result = llm.invoke(prompt)

        clean_text = result.content if isinstance(result.content, str) else result.content[0]['text']


        st.session_state.messages.append({"role": "ai", "content": clean_text})
        with st.chat_message("ai"):
            st.markdown(clean_text)

