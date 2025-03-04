from openai import OpenAI
import streamlit as st
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
embeddings = PineconeEmbeddings(
    model=st.secrets["EMBEDDING_MODEL"],
    pinecone_api_key=st.secrets["PINECONE_API_KEY"]
)

docsearch = PineconeVectorStore.from_existing_index(
    index_name=st.secrets["PINECONE_INDEX_NAME"],
    embedding=embeddings,
    namespace=st.secrets["PINECONE_NAMESPACE"],
)
retriever = docsearch.as_retriever()

llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model_name=st.secrets["LLM_MODEL"],
    temperature=0.0,
    streaming=True
)

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

st.title("RAG simple")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "assistant" == message["role"] and isinstance(message["context"], list) and len(message["context"]) > 0:
            with st.expander("Xem thêm thông tin trong tài liệu bổ sung"):
                for doc in message["context"]:
                    st.header(f"{int(doc.metadata['page'])}/{int(doc.metadata['total_pages'])} pages. {doc.metadata['source']}", divider="gray")
                    st.text(doc.page_content)

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        context = None
        response = ""
        placeholder = st.empty()
        for chunk in retrieval_chain.stream({"input": prompt}):
            if "answer" in chunk:
                response += chunk["answer"]
                placeholder.write(response)
            elif "context" in chunk:
                context = chunk["context"]
        if isinstance(context, list) and len(context) > 0:
            with st.expander("Xem thêm thông tin trong tài liệu bổ sung"):
                for doc in context:
                    st.header(f"{int(doc.metadata['page'])}/{int(doc.metadata['total_pages'])} pages. {doc.metadata['source']}", divider="gray")
                    st.text(doc.page_content)
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "context": context})
