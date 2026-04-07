import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
# from langchain_community.document_compressors import FlashrankRerank
from langchain_cohere import CohereRerank
from langchain_core.messages import HumanMessage, AIMessage
from html_templates import css, bot_template, user_template
from prompts import qa_prompt, rewrite_prompt
import tempfile
import config

def get_pdf_text(pdf_docs):
    documents = []

    for doc in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(doc.read())
            temp_path = temp_file.name
        
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = doc.name

        documents.extend(docs)
    return documents

def get_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = config.CHUNK_SIZE,
        chunk_overlap = config.CHUNK_OVERLAP
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def create_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        embedding=embeddings,
        documents=text_chunks
    )
    return vector_store

def get_llm():
    return ChatOpenAI(model=config.QA_MODEL)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def get_sources(docs):
    sources = []
    seen = set()

    for doc in docs:
        score = doc.metadata.get("relevance_score", 0)

        if score < config.RERANKER_SCORE_THRESHOLD:
            continue

        filename = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        filename = filename.split("/")[-1]
        key = (filename, page)

        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": filename,
                "page": page + 1,
            })

    return sources

def format_chat_history(chat_history):
    formatted = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            formatted.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"AI: {msg.content}")
    return "\n".join(formatted)

def create_hybrid_retriever(vector_store, text_chunks):
    # semantic retriever
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k" : config.RETRIEVER_K})

    # bm25 keyword retriever
    bm25_retriever = BM25Retriever.from_documents(text_chunks)
    bm25_retriever.k = config.RETRIEVER_K

    #hybrid retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=config.ENSEMBLE_WEIGHTS
    )

    compressor = CohereRerank(
        model=config.RERANKER_MODEL,
        top_n = config.RERANKER_TOP_N)
    reranking_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid_retriever
    )

    return reranking_retriever

def rewrite_question(user_question, chat_history):
    if not chat_history:
        return user_question

    rewriter = rewrite_prompt | ChatOpenAI(
        model=config.REWRITER_MODEL,
        temperature=config.REWRITER_TEMPERATURE
    ) | StrOutputParser()

    return rewriter.invoke({
        "chat_history": format_chat_history(chat_history),
        "question": user_question
    })

def create_chain(context, llm, chat_history):
    chain = (
        {
            "context": RunnableLambda(lambda x: context),
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(
                lambda x: format_chat_history(chat_history)
            )
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return chain

def handle_user_input(user_question):
    rewritten_question = rewrite_question(user_question, st.session_state.chat_history)

    retrieved_docs = st.session_state.retriever.invoke(rewritten_question)
    sources = get_sources(retrieved_docs)
    context = format_docs(retrieved_docs)

    chain = create_chain(
        context,
        st.session_state.llm,
        st.session_state.chat_history
    )
    print("rewritten question: ", rewritten_question)
    response = chain.invoke(rewritten_question)

    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=response))

    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        elif isinstance(msg, AIMessage):
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

    if sources:
        with st.expander("Sources"):
            for source in sources:
                st.markdown(f"- **{source['filename']}** — Page {source['page']}")

def main():
    load_dotenv()

    st.set_page_config(page_title="Ask My Docs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "llm" not in st.session_state:
        st.session_state.llm = get_llm()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Ask My Docs")
    user_question = st.text_input("Ask a question about your documents")

    if user_question and st.session_state.vector_store:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your pdfs here and click on Process", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):

                documents = get_pdf_text(pdf_docs)

                text_chunks = get_chunks(documents)

                vector_store = create_vector_store(text_chunks)

                retriever = create_hybrid_retriever(vector_store, text_chunks)

                st.session_state.vector_store = vector_store
                st.session_state.retriever = retriever

                st.success("Processing Complete!")

if __name__ == '__main__':
    main()