from ollama import embeddings
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
import os
import hashlib



custom_template = """

Eres un asistente experto en análisis de libros en PDF.

Tu tarea:
- Leer el contenido proporcionado.
- Responder únicamente lo que se te pregunte, basándote solo en ese contenido.
- Si el contenido no es suficiente para responder con certeza, di que no tienes suficiente información.

Reglas:

Para responder debes seguir lo siguente:

Contexto: {context}
Pregunta: {question}

- No inventes, no agregues conocimiento externo.
- No asumas nada que no esté explícitamente en el texto.
- Siempre responde en español.

Tipos de pedidos posibles:
- Preguntas sobre el contenido del libro.
- Solicitudes de resumen de capítulos o fragmentos.

Importante:
- Si el usuario pide un resumen, genera uno breve y claro del contenido proporcionado.
- Si el usuario hace una pregunta, respóndela directamente, sin rodeos ni explicaciones innecesarias.

"""

pdfs_directory = "./pdfs"
db_directory = "./vector_db"
if not os.path.exists(db_directory):
    os.makedirs(db_directory)
if not os.path.exists(pdfs_directory):
    os.makedirs(pdfs_directory)
    
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(persist_directory=db_directory, embedding_function=embeddings)
model = OllamaLLM(model="phi4-mini:3.8b-q4_K_M", temperature=0.4, max_tokens=2000, stream = True)


def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


#funcion para. dividir el contenido
def text_splitter(documents, book_name=None):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    if book_name:
        for chunk in chunks:
            chunk.page_content = f"Curso: {book_name}\n{chunk.page_content}"
    return chunks

#Funcion para guardar en la memoria.
def index_docs(documents):
    vector_store.add_documents(documents)
    vector_store.persist()
    
#Funcion para devolver contenido relacionado a la pregunta del usuario
def retrieve_docs(query, book_name=None):
    docs = vector_store.similarity_search(query, k=10) 
    if book_name:
        docs = [doc for doc in docs if doc.metadata.get("book_name") == book_name]
    return docs


#Funcion para generar la respuesta.

def generate_response_stream(contexto, book_name):
    prompt = ChatPromptTemplate.from_template(custom_template)
    chain = prompt | model

    response_stream = chain.stream({
        "contexto": contexto,
        "book_name":book_name,
    })

    return response_stream


def get_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def is_pdf_already_indexed(file_hash):
    result = vector_store.similarity_search(file_hash, k=1)
    if result:
        for doc in result:
            if doc.metadata.get("file_hash") == file_hash:
                return True
    return False




uploaded_file = st.file_uploader("Sube un PDF", type="pdf", accept_multiple_files=False)
book_name = st.text_input("Nombre del libro")

#Verfiicar si el pdf ya fue indexado
if uploaded_file and book_name:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)

    file_hash = get_file_hash(pdfs_directory + uploaded_file.name)
    if is_pdf_already_indexed(file_hash):
        st.warning("Este PDF ya ha sido indexado.")
    else:
        st.success("PDF subido y procesado correctamente.")
        


    chunked_documents = text_splitter(documents)
    for doc in chunked_documents:
        doc.metadata["file_hash"] = file_hash
        doc.metadata["book_name"] = book_name

    index_docs(chunked_documents)

question = st.text_input("Pregunta del usuario")
    

if question != "":
    st.chat_message("user").write(f"Título del libro: {question}")
    st.markdown("---")

    # 2. Recuperar contexto relevante (si existe)
    related_documents = retrieve_docs( question)

    # Extraer el texto de cada Document
    contexto = "\n".join(doc.page_content for doc in related_documents) if related_documents else ""
    # 3. Armar el prompt rellenando el template
    prompt = custom_template.format(
        book_name= book_name,
        contexto=contexto,
    )

    # 4. Generar respuesta
    message_placeholder = st.chat_message("assistant").empty()
    full_response = ""

    for chunk in generate_response_stream(contexto, book_name):
        full_response += chunk  # cada chunk trae parte del texto
        message_placeholder.markdown(full_response)  # vamos actualizando
