import os
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")


# Initialize chat history in Streamlit session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def translate_role_for_streamlit(user_role):
    """
    Translate user roles for Streamlit chat display.
    """
    return "assistant" if user_role == "model" else user_role

def save_uploaded_files(uploaded_files):
    """
    Save uploaded files to a temporary directory and return the file paths.
    """
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    return file_paths

def get_pdf_text(file_paths):
    """
    Extract text from PDF files using LlamaParse.
    """
    parser = LlamaParse(api_key=LLAMA_CLOUD_API_KEY, result_type="markdown", verbose=True)
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=file_paths, file_extractor=file_extractor).load_data()
    concatenated_text = "\n".join([doc.text for doc in documents])
    return concatenated_text

def get_text_chunks(text):
    """
    Split text into chunks to process with FAISS.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """
    Create and save a vector store for the text chunks using FAISS.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """
    Create and return a conversational chain for handling user questions.
    """
    prompt_template = """
        You are an intelligent assistant. First, try to answer the question based on the context provided below. If the context does not contain the answer or if the question is general, use your general knowledge and available information to respond.

        Context (if available):
        {context}

        User's Question:
        {question}

        Assistant's Response:
        """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    """
    Process user input to find the answer from the vector store and generate a response.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    st.header("Gemini PDF Chatbot 🤖")

    # Display the chat history
    for message in st.session_state.chat_history:
        role, content = message
        with st.chat_message(role):
            st.markdown(content)

    user_question = st.chat_input("Ask a Question from the PDF Files")

    if user_question:
        # Display the user's question
        st.chat_message("user").markdown(user_question)
        st.session_state.chat_history.append(("user", user_question))

        # Process the user's question
        response_text = user_input(user_question)
        
        # Display the assistant's response
        st.chat_message("assistant").markdown(response_text)
        st.session_state.chat_history.append(("assistant", response_text))

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    file_paths = save_uploaded_files(pdf_docs)
                    raw_text = get_pdf_text(file_paths)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing Complete!")
            else:
                st.warning("Please upload PDF files.")


if __name__ == "__main__":
    main()
