# 🧠 RAG PDF Assistant

Este proyecto es una aplicación de RAG (*Retrieval-Augmented Generation*) que permite subir libros en PDF, analizarlos y hacer preguntas sobre su contenido o pedir resúmenes. Utiliza modelos locales a través de [Ollama](https://ollama.com), y una interfaz interactiva construida con Streamlit.

---

## 🚀 Características

- 📄 Carga e indexado de libros en PDF.
- 🧠 Búsqueda semántica de contenido usando **Chroma**.
- 🤖 Generación de respuestas o resúmenes usando un modelo LLM local (ej: `phi4-mini`).
- 🌐 Interfaz simple con Streamlit para interacción rápida.
- 🔒 Evita respuestas inventadas gracias al uso de contexto real del libro.

---

## 🧰 Tecnologías utilizadas

- [Ollama](https://ollama.com)
- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [PDFPlumber](https://github.com/jsvine/pdfplumber)
- [Streamlit](https://streamlit.io/)

---

## ⚙️ Requisitos

- Python 3.10 o superior
- Tener instalado [Ollama](https://ollama.com)

---

## 🧩 Instalación de Ollama

1. Descargá e instalá Ollama desde: [https://ollama.com/download](https://ollama.com/download)
2. Abrí una terminal y ejecutá:

```bash
ollama run phi4-mini
```

Esto descargará el modelo phi4-mini.

También necesitás el modelo de embeddings:

```bash
ollama run nomic-embed-text
```

## 🛠 Instalación del proyecto

1. Cloná el repositorio:
```bash
git clone https://github.com/gabykap29/rag_example.git
cd tu-repo
```

2. Instalá las dependencias (se recomienda usar un entorno virtual):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📦 Requisitos del requirements.txt

```txt
streamlit
langchain
langchain-community
langchain-core
langchain-ollama
langchain-text-splitters
chromadb
pdfplumber
ollama
```

## ▶️ Cómo ejecutar la app

1. Asegurate de tener corriendo los modelos de Ollama:

```bash
ollama run phi4-mini
ollama run nomic-embed-text

```
2. Ejecutá la aplicación:
```bash
streamlit run app.py
```

3. Abrí tu navegador en: http://localhost:8501

📁 Estructura de carpetas

```graphql
📦 https://github.com/gabykap29/rag_example.git/
 ┣ 📂 pdfs/              # PDFs subidos por el usuario
 ┣ 📂 vector_db/         # Base vectorial persistente (ChromaDB)
 ┣ 📄 main.py             # Código principal del asistente RAG
 ┣ 📄 demo.py             # Código dividido y explicado del asistente RAG en formato notebook
 ┣ 📄 requirements.txt   # Dependencias
 ┗ 📄 README.md          # Este archivo
```