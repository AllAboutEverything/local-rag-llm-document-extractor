import streamlit as st
from langchain_chroma import Chroma 
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time

# --- SEITEN-KONFIGURATION ---
st.set_page_config(page_title="DevKnowledge AI Assistant", layout="wide")
st.title("DevKnowledge AI")

# --- SIDEBAR: RETRIEVAL-EINSTELLUNGEN ---
with st.sidebar:
    st.header("Retrieval-Einstellungen")
    k_docs = st.slider("Anzahl abgerufener Chunks (k)", min_value=1, max_value=20, value=3)
    score_threshold = st.slider("Min. Similarity Score (0 = alles)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    search_type = st.selectbox("Such-Strategie", ["similarity", "mmr"])
    show_debug = st.toggle("Debug-Output anzeigen", value=True)
    
    st.divider()
    st.header("Direkte Vektorsuche")
    test_query = st.text_input("Test-Abfrage (ohne LLM):")
    if st.button("Nur Chunks abrufen") and test_query:
        st.session_state["run_test_query"] = test_query

# --- 1. SYSTEM LADEN ---
@st.cache_resource
def load_system():
    embeddings = OllamaEmbeddings(model="qwen3-embedding:latest")
    vectorstore = Chroma(persist_directory="db/", embedding_function=embeddings)
    llm = ChatOllama(model="qwen3:8b", temperature=0.1)
    return vectorstore, llm, embeddings

vectorstore, llm, embeddings = load_system()

# Datenbank-Info
with st.sidebar:
    st.divider()
    st.caption(f"Dokumente in DB: {vectorstore._collection.count()}")
    st.caption("Embedding: qwen3-embedding:latest")

# --- 2. DIREKTE VEKTORSUCHE (Test ohne LLM) ---
if "run_test_query" in st.session_state:
    query = st.session_state.pop("run_test_query")
    st.subheader(f"Direkte Chunk-Suche für: *{query}*")
    
    results_with_scores = vectorstore.similarity_search_with_score(query, k=k_docs)
    
    for i, (doc, score) in enumerate(results_with_scores):
        with st.expander(f"Chunk {i+1} | Score: {score:.4f} | {doc.metadata.get('source', 'unbekannt')}"):
            st.text(doc.page_content)
            st.caption(f"Metadata: {doc.metadata}")
    st.divider()

# --- 3. PROMPT TEMPLATE ---
template = """You are a helpful technical assistant. Answer the question in english based on the given context.

If the context contains relevant information, answer the question in detail.
If the context does not contain relevant information, clearly say: "I could not find this information in the context."

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate.from_template(template)

# --- 4. RETRIEVER ---
def get_retriever():
    if search_type == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k_docs, "fetch_k": k_docs * 3}
        )
    else:
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_docs}
        )

# --- 5. CHAT-HISTORIE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. CHAT-INPUT & LOGIK ---
if prompt := st.chat_input("Stelle eine Frage zu deinen Dokumenten..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        # === SCHRITT 1: RETRIEVAL ===
        if show_debug:
            st.markdown("---")
            st.markdown("### Schritt 1: Retrieval")
        
        retriever = get_retriever()
        t_start = time.time()
        retrieved_docs = retriever.invoke(prompt)
        t_retrieval = time.time() - t_start
        
        if show_debug:
            st.caption(f"Retrieval: {t_retrieval:.2f}s | {len(retrieved_docs)} Chunks gefunden")
            
            for i, doc in enumerate(retrieved_docs):
                # Similarity Score einzeln berechnen
                score_result = vectorstore.similarity_search_with_score(prompt, k=1)
                
                with st.expander(f"Chunk {i+1}: {doc.metadata.get('source', 'unbekannt')} | Seite: {doc.metadata.get('page', '?')}"):
                    st.text_area(f"Inhalt Chunk {i+1}", doc.page_content, height=150, key=f"chunk_{i}_{prompt[:20]}")
                    st.caption(f"Zeichen: {len(doc.page_content)} | Metadata: {doc.metadata}")

        # === SCHRITT 2: KONTEXT ZUSAMMENSTELLEN ===
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        if show_debug:
            st.markdown("### Schritt 2: Zusammengestellter Kontext")
            with st.expander(f"Vollständiger Kontext ({len(context)} Zeichen)"):
                st.text(context)

        # === SCHRITT 3: PROMPT AUFBAUEN ===
        final_prompt = QA_PROMPT.format(context=context, question=prompt)
        
        if show_debug:
            st.markdown("### Schritt 3: Finaler Prompt ans LLM")
            with st.expander(f"Prompt ({len(final_prompt)} Zeichen)"):
                st.text(final_prompt)

        # === SCHRITT 4: LLM ANTWORT ===
        if show_debug:
            st.markdown("### Schritt 4: LLM Antwort")
        
        t_llm_start = time.time()
        
        # Streaming
        response_container = st.empty()
        full_response = ""
        
        for chunk in llm.stream(final_prompt):
            full_response += chunk.content
            response_container.markdown(full_response + "▌")
        
        response_container.markdown(full_response)
        t_llm = time.time() - t_llm_start
        
        if show_debug:
            st.caption(f"LLM Generierung: {t_llm:.2f}s")
            st.markdown("---")

        # === QUELLEN ===
        with st.expander("Verwendete Quellen"):
            sources = {}
            for doc in retrieved_docs:
                src = doc.metadata.get('source', 'unbekannt')
                page = doc.metadata.get('page', '?')
                if src not in sources:
                    sources[src] = []
                sources[src].append(str(page))
            
            for src, pages in sources.items():
                st.write(f"- **{src}** (Seiten: {', '.join(set(pages))})")
        
        # Similarity Score für alle Chunks separat zeigen
        if show_debug:
            with st.expander("Similarity Scores (alle Chunks)"):
                scores_results = vectorstore.similarity_search_with_score(prompt, k=k_docs)
                for doc, score in scores_results:
                    bar_val = max(0.0, min(1.0, 1.0 - score))  # Chroma gibt L2-Distanz zurück
                    st.write(f"Score: `{score:.4f}` (L2-Distanz) | {doc.metadata.get('source','?')} S.{doc.metadata.get('page','?')}")
                    st.progress(bar_val)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
