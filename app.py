from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token 
        self.message_box.markdown(self.message)


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)
st.title("DocumentGPT")

st.markdown("""
Welcome!

Use this chatbot to ask questions to an AI about your files!
            
Upload your files on the sidebar.
""")

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""
        
with st.sidebar:
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")

    if st.button("Save API Key"):
        if openai_api_key:
            st.session_state["openai_api_key"] = openai_api_key
            
            st.success("API Key saved successfully!")
        else:
            st.error("Please enter a valid API Key.")

    file = st.file_uploader("Upload a .txt .pdf or  .docx file", type =["pdf", "docx", "txt"])

llm = ChatOpenAI(openai_api_key=st.session_state["openai_api_key"], temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()])

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(f"./.cache/files/{file.name}")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message,role)

def save_message(message, role):
    st.session_state["messages"].append({"messages": message, "role": role})

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["messages"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """
    Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up

    Context: {context}
    """),
    ("human", "{question}")
    ])

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!","ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about you file...")
    if message:
        send_message(message, "human")
        chain = ({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } | prompt | llm)
        with st.chat_message("ai"):
            response = chain.invoke(message)

else:
    st.session_state["messages"] = []