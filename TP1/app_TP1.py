import os
import streamlit as st
import sys
import asyncio
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# ===== Configuración del Event Loop =====
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Crear un nuevo loop si no existe
def get_event_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()

loop = get_event_loop()
asyncio.set_event_loop(loop)

# ===== PARTE 1: Carga de documentos =====
st.title("Sistema RAG con Pinecone y Groq 🚀")
st.subheader("Cargando y procesando documentos...")

dir_cv = r'C:\Users\Gerardo Vilcamiza\Documents\Python Scripts\Master IA\LLMs\CVs\Gerardo Vilcamiza'
file_loader = PyPDFDirectoryLoader(dir_cv)
doc_cv = file_loader.load()

st.success(f"Documentos PDF cargados: {len(doc_cv)}")

# ===== PARTE 2: División en fragmentos =====
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
documents_cv = text_splitter.split_documents(doc_cv)

st.success(f"Documentos divididos en {len(documents_cv)} fragmentos")

# ===== PARTE 3: Configuración de Pinecone =====
PINECONE_API_KEY = "pcsk_5o862d_Lphz9e8ANb6jeZTA2w7DHpjUHoGZKCLjCZWKEMKb1sVa7ofgvg9WFpjgWDpsp5Y"
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Configuración con cloud y región
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "eguins"

if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=1024, metric="cosine", spec=spec)
    st.success(f"Índice {index_name} creado exitosamente.")

# ===== PARTE 4: Generación de embeddings y carga en Pinecone =====
st.subheader("Insertando fragmentos en Pinecone...")

# Inicialización asíncrona de PineconeEmbeddings
async def init_pinecone_embeddings():
    return PineconeEmbeddings(model="multilingual-e5-large")

embed_model = loop.run_until_complete(init_pinecone_embeddings())

namespace = "espacio"

docsearch = PineconeVectorStore.from_documents(
    documents=documents_cv,
    index_name=index_name,
    embedding=embed_model,
    namespace=namespace
)
st.success(f"Fragmentos insertados en el índice '{index_name}'")

# ===== PARTE 5: Configuración de Groq =====
os.environ["GROQ_API_KEY"] = "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"
groq_api_key = os.environ["GROQ_API_KEY"]

llm = ChatGroq(model="llama3-70b-8192", temperature=0.2)
st.success("Modelo LLaMA 3 de Groq configurado.")

# ===== PARTE 6: Configuración del Chat =====
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Inicializando la session_state para el historial del chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Se muestra el historial de mensajes
st.subheader("Chat interactivo 📜")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Entrada del usuario
query = st.chat_input("Escribe tu pregunta aquí:")

# Se procesa la entrada del usuario y se actualiza el historial del chat
if query:
    # Añade pregunta al historial
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Mostrando respuesta del modelo
    with st.spinner("Buscando respuesta..."):
        response = qa_chain.invoke({"query": query})["result"]

    # Añade respuesta al historial
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
