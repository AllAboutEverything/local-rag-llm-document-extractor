import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "data/"
DB_PATH = "db/"

def main():
    # 0. Alte Datenbank löschen
    if os.path.exists(DB_PATH):
        print(f"Lösche alte Datenbank in {DB_PATH}...")
        shutil.rmtree(DB_PATH)

    # 1. Dokumente laden
    print(f"Lade PDFs aus {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    raw_documents = loader.load()
    print(f"{len(raw_documents)} PDF-Seiten geladen.")

    # 2. Chunking mit Überlappung
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600, 
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"{len(chunks)} Chunks erstellt.")

    # 3. Embeddings mit Qwen3 via Ollama
    print("Starte Vektorisierung mit qwen3-embedding:8b (lokal)...")
    embeddings = OllamaEmbeddings(model="qwen3-embedding:latest")

    # 4. Vektordatenbank erstellen
    print("Erstelle neue ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    
    # Test-Check
    print("\n--- DEBUG-CHECK ---")
    test_query = "Fehlerbehandlung"
    test_docs = vectorstore.similarity_search(test_query, k=1)
    if test_docs:
        print(f"Erfolgreicher Test-Abruf! Beispiel-Inhalt:\n{test_docs[0].page_content[:300]}...")

    print("\nFertig! Dein Wissen ist nun mit Qwen3-Power indiziert.")

if __name__ == "__main__":
    main()