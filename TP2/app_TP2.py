import os
import streamlit as st
import re
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

# Crear un loop manualmente
def get_event_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()

loop = get_event_loop()
asyncio.set_event_loop(loop)

# ===== Configuración inicial =====
PINECONE_API_KEY = "pcsk_5o862d_Lphz9e8ANb6jeZTA2w7DHpjUHoGZKCLjCZWKEMKb1sVa7ofgvg9WFpjgWDpsp5Y"
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Configuración Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "curriculums"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=1024, metric="cosine", spec=spec)

# Ejecutar la inicialización de PineconeEmbeddings en el event loop
async def init_pinecone_embeddings():
    return PineconeEmbeddings(model="multilingual-e5-large")

embed_model = loop.run_until_complete(init_pinecone_embeddings())

# ===== Configuración de agentes =====
def configure_agent(name, path, namespace):
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    
    vectorstore = PineconeVectorStore.from_documents(
        documents=split_docs,
        index_name=index_name,
        embedding=embed_model,
        namespace=namespace
    )
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff")
    return qa_chain

# Configurar agentes
agent_a = configure_agent("Gerardo Vilcamiza", r'C:\Users\Gerardo Vilcamiza\Documents\Python Scripts\Master IA\LLMs\CVs\Gerardo Vilcamiza', "namespace_a")
agent_b = configure_agent("Ruben Vidal", r'C:\Users\Gerardo Vilcamiza\Documents\Python Scripts\Master IA\LLMs\CVs\Ruben Vidal', "namespace_b")

# ===== Lógica de selección de agentes =====
def detect_agents(query):
    """
    Detecta qué agentes deben responder la consulta.
    - Si ambos nombres están presentes: devuelve ambos.
    - Si solo uno está presente: devuelve ese agente.
    - Si no hay nombres, devuelve el agente por defecto (Persona A -> gerardo vilcamiza).
    """
    has_persona_a = bool(re.search(r'\b(gerardo vilcamiza|persona a)\b', query, re.IGNORECASE))
    has_persona_b = bool(re.search(r'\b(ruben vidal|persona b)\b', query, re.IGNORECASE))

    if has_persona_a and has_persona_b:
        return ["agent_a", "agent_b"]
    elif has_persona_b:
        return ["agent_b"]
    return ["agent_a"]  # Por defecto

# ===== Interfaz de chat =====
st.title("Sistema de Agentes con Avatares 🤖")

# Historial de mensajes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar", None)):
        st.write(message["content"])

# Entrada del usuario
query = st.chat_input("Escribe tu pregunta aquí:")
if query:
    st.session_state.messages.append({"role": "user", "content": query, "avatar": "🧑‍💼"})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.write(query)
    
    # Detectar agentes relevantes
    agents_to_use = detect_agents(query)
    responses = []

    # Responder con cada agente correspondiente
    for agent_name in agents_to_use:
        if agent_name == "agent_a":
            response = agent_a.invoke({"query": query})["result"]
            responses.append({"name": "Gerardo Vilcamiza", "avatar": "👨‍💻", "response": response})
        elif agent_name == "agent_b":
            response = agent_b.invoke({"query": query})["result"]
            responses.append({"name": "Ruben Vidal", "avatar": "🤓", "response": response})

    # Mostrando respuestas de los agentes
    for res in responses:
        st.session_state.messages.append({"role": "assistant", "content": f"**{res['name']}**: {res['response']}", "avatar": res['avatar']})
        with st.chat_message("assistant", avatar=res['avatar']):
            st.write(f"**{res['name']}**: {res['response']}")
